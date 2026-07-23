"""SFT2 的完整单批算法：模型前向、目标函数和权重策略。"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nimloth.training.sft2.batch import SFT2Batch
from nimloth.training.sft2.runtime import SFT2ModelRuntime
from nimloth.wm import (
    LatentWMPredictor,
    SequenceSIGReg,
)


def require_sft2_wm_history(
    wm_predictor: LatentWMPredictor,
    *,
    history_size: int,
    source: Path,
) -> None:
    """拒绝加载与当前 SFT2 LeWM 上下文长度不一致的 predictor。"""

    actual = int(wm_predictor.config.history_size)
    expected = int(history_size)
    if actual != expected:
        raise ValueError(
            "SFT2 WM checkpoint history_size does not match config: "
            f"checkpoint={actual}, config={expected}, source={source}"
        )


@dataclass(frozen=True)
class SFT2StepOutput:
    """一次 SFT2 主前向产生的 loss、当前 state 和日志指标。"""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor | None]
    metrics: dict[str, float]
    current_state: torch.Tensor
    sample_count: int


@dataclass(frozen=True)
class SFT2SIGRegStepOutput:
    """主 loss 反传完成后，单独执行的 SIGReg 阶段结果。"""

    loss: torch.Tensor
    raw_loss: torch.Tensor | None
    metrics: dict[str, float]


class _DifferentiableAllGather(torch.autograd.Function):
    """汇聚小型 state，并在 backward 时把全局梯度送回来源 rank。

    每个 rank 都计算同一个全局 loss；backward 先合计所有 rank 对 global state
    的梯度，再取回本 rank 的 slice。随后 DDP 对模型参数做平均，正好得到一次全局
    batch loss 对共享参数的梯度。
    """

    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            ctx.world_size = 1
            ctx.rank = 0
            ctx.local_rows = value.shape[0]
            return value
        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()
        ctx.local_rows = value.shape[0]
        gathered = [torch.empty_like(value) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, value)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, global_gradient: torch.Tensor) -> tuple[torch.Tensor]:
        if ctx.world_size == 1:
            return (global_gradient,)
        summed = global_gradient.contiguous().clone()
        dist.all_reduce(summed, op=dist.ReduceOp.SUM)
        start = ctx.rank * ctx.local_rows
        return (summed.narrow(0, start, ctx.local_rows),)


def gather_global_sigreg_states(
    detached_current_state: torch.Tensor,
    next_state: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """构造跨 rank 的有效 ``(s_t,s_{t+1})`` SIGReg batch。

    current state 始终无梯度；next state 使用可微 gather。不同 rank 的本地 B 可以
    不同，collective 前会补齐到最大 B，再由 global valid mask 删除补齐与 sampler
    padding 行。
    """

    if detached_current_state.requires_grad:
        raise ValueError("SFT2 SIGReg current_state must be detached")
    if detached_current_state.ndim != 2 or next_state.ndim != 2:
        raise ValueError("SFT2 global SIGReg states must have shape (B,D)")
    if detached_current_state.shape != next_state.shape:
        raise ValueError("SFT2 global SIGReg local state shapes do not match")
    if valid_mask.shape != (next_state.shape[0],):
        raise ValueError(
            "SFT2 global SIGReg valid mask must have shape (B,), "
            f"got {tuple(valid_mask.shape)} for B={next_state.shape[0]}"
        )

    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    local_size = torch.tensor(
        [next_state.shape[0]],
        device=next_state.device,
        dtype=torch.int64,
    )
    sizes = [torch.empty_like(local_size) for _ in range(world_size)]
    if distributed:
        dist.all_gather(sizes, local_size)
    else:
        sizes[0].copy_(local_size)
    max_size = max(int(size.item()) for size in sizes)

    def pad_rows(value: torch.Tensor) -> torch.Tensor:
        missing = max_size - value.shape[0]
        if missing <= 0:
            return value.contiguous()
        return torch.cat(
            (value, value.new_zeros((missing, *value.shape[1:]))),
            dim=0,
        )

    current_padded = pad_rows(detached_current_state.detach())
    next_padded = pad_rows(next_state)
    valid_padded = torch.zeros(max_size, device=next_state.device, dtype=torch.int64)
    valid_padded[: valid_mask.shape[0]] = valid_mask.to(
        device=next_state.device,
        dtype=torch.int64,
    )

    gathered_current = [torch.empty_like(current_padded) for _ in range(world_size)]
    gathered_valid = [torch.empty_like(valid_padded) for _ in range(world_size)]
    if distributed:
        dist.all_gather(gathered_current, current_padded)
        dist.all_gather(gathered_valid, valid_padded)
    else:
        gathered_current[0].copy_(current_padded)
        gathered_valid[0].copy_(valid_padded)
    global_current = torch.cat(gathered_current, dim=0)
    global_next = _DifferentiableAllGather.apply(next_padded)
    global_valid = torch.cat(gathered_valid, dim=0).bool()
    valid_count = int(global_valid.sum().item())
    return (
        global_current[global_valid],
        global_next[global_valid],
        valid_count,
    )


@contextlib.contextmanager
def shared_sigreg_rng(seed: int, device: torch.device) -> Iterator[None]:
    """只在 SIGReg 内使用跨 rank 相同的随机投影，并恢复各 rank 原 RNG。"""

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        yield


class SFT2Algorithm:
    """定义 SFT2 一个 batch 的目标函数与两阶段计算图。

    主阶段只计算当前 step 的 CE/WM/value；它反传并释放 Qwen 图后，SIGReg 阶段
    才以 detached ``s_t`` 和在线 ``s_{t+1}`` 计算正则。这样 SIGReg 数值上仍看见
    两个连续状态，但梯度只进入新状态侧，也不会同时保留两份 Qwen 激活。
    """

    def __init__(
        self,
        *,
        history_size: int,
        sigreg: SequenceSIGReg | None,
        sigreg_weight: float,
        value_weight: float,
        ce_weight: float,
        value_rank_margin: float,
        value_rank_weight: float,
        wm_weight_start: float = 0.1,
        wm_weight_end: float = 1.0,
        wm_warmup_fraction: float = 0.3,
    ) -> None:
        self.history_size = int(history_size)
        if self.history_size < 1:
            raise ValueError(
                f"history_size must be positive, got {self.history_size}"
            )
        self.sigreg = sigreg
        self.sigreg_weight = float(sigreg_weight)
        self.value_weight = float(value_weight)
        self.ce_weight = float(ce_weight)
        self.value_rank_margin = float(value_rank_margin)
        self.value_rank_weight = float(value_rank_weight)
        self.wm_weight_start = float(wm_weight_start)
        self.wm_weight_end = float(wm_weight_end)
        self.wm_warmup_fraction = float(wm_warmup_fraction)

    def wm_weight(self, global_step: int, total_steps: int) -> float:
        """在训练前段用 cosine ramp 增加 WM loss 权重。"""

        if total_steps <= 0:
            return self.wm_weight_end
        warmup_steps = max(1, int(total_steps * self.wm_warmup_fraction))
        if global_step >= warmup_steps:
            return self.wm_weight_end
        progress = global_step / warmup_steps
        cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.wm_weight_start + (
            self.wm_weight_end - self.wm_weight_start
        ) * cosine

    @property
    def has_sigreg_stage(self) -> bool:
        """训练是否需要在主 loss 反传后执行独立 SIGReg 阶段。"""

        return self.sigreg is not None and self.sigreg_weight > 0.0

    def training_primary_step(
        self,
        runtime: SFT2ModelRuntime,
        batch: SFT2Batch,
        *,
        wm_weight: float,
    ) -> SFT2StepOutput:
        return self._step(
            runtime,
            batch,
            wm_weight=wm_weight,
            include_lm_loss=True,
            include_value_ranking=True,
        )

    def training_sigreg_step(
        self,
        runtime: SFT2ModelRuntime,
        batch: SFT2Batch,
        *,
        detached_current_state: torch.Tensor,
        sigreg_seed: int,
    ) -> SFT2SIGRegStepOutput:
        """只让在线 ``s_{t+1}`` 接收 SIGReg 梯度。

        调用者必须先完成主 loss backward，再调用本方法。小于两个样本的 rank
        无法估计 SIGReg 分布，但仍用依赖在线 state 的零 loss 参与 DDP backward。
        """

        if not self.has_sigreg_stage:
            raise RuntimeError("SFT2 SIGReg stage is disabled")
        next_state = runtime.agent.encode_state(
            batch.online_tail,
            include_lm_loss=False,
        ).state
        global_current, global_next, global_batch_size = gather_global_sigreg_states(
            detached_current_state,
            next_state,
            batch.sample_weights > 0.0,
        )
        with shared_sigreg_rng(sigreg_seed, global_next.device):
            sigreg_loss = self._sigreg_loss(global_current, global_next)
        if sigreg_loss is None:
            # 保留全局 gather/在线编码图参与，但不伪造 global B<2 的统计量。
            backward_loss = global_next.sum() * 0.0
            metrics = {"sigreg_skipped_small_batch": 1.0}
        else:
            backward_loss = self.sigreg_weight * sigreg_loss
            metrics = {"sigreg_loss": float(sigreg_loss.detach().item())}
        metrics["sigreg_global_batch_size"] = float(global_batch_size)
        return SFT2SIGRegStepOutput(
            loss=backward_loss,
            raw_loss=sigreg_loss,
            metrics=metrics,
        )

    def merge_training_metrics(
        self,
        primary_metrics: dict[str, float],
        sigreg: SFT2SIGRegStepOutput | None,
    ) -> dict[str, float]:
        """把两个显式反传阶段合并为一个 optimizer-step 日志视图。"""

        metrics = dict(primary_metrics)
        metrics["lambda_sigreg"] = self.sigreg_weight if sigreg is not None else 0.0
        if sigreg is None:
            return metrics
        metrics.update(sigreg.metrics)
        metrics["total_loss"] += float(sigreg.loss.detach().item())
        return metrics

    def evaluation_step(
        self,
        runtime: SFT2ModelRuntime,
        batch: SFT2Batch,
    ) -> SFT2StepOutput:
        return self._step(
            runtime,
            batch,
            wm_weight=1.0,
            include_lm_loss=False,
            include_value_ranking=False,
        )

    def _step(
        self,
        runtime: SFT2ModelRuntime,
        batch: SFT2Batch,
        *,
        wm_weight: float,
        include_lm_loss: bool,
        include_value_ranking: bool,
    ) -> SFT2StepOutput:
        """按照 current forward → next target → CE/WM/value 完成主阶段。"""

        if not 1 <= batch.history_size <= self.history_size:
            raise ValueError(
                "SFT2 batch context length exceeds algorithm history_size: "
                f"batch={batch.history_size}, algorithm={self.history_size}"
            )
        # 当前 Qwen 只执行一次；更老 state 来自它们先前 current forward 的 cache。
        current_encoded = runtime.agent.encode_state(
            batch.current,
            include_lm_loss=include_lm_loss,
        )
        cached_history = runtime.history_cache.history(
            batch.history_keys,
            reference=current_encoded.state,
        )
        model_output = runtime.agent.forward_step_from_history(
            batch.action_indices,
            cached_history,
            encoded_current=current_encoded,
        )
        runtime.history_cache.store(
            batch.current_keys,
            model_output.state[:, -1],
            enabled=not batch.is_padding,
        )
        target_states = runtime.target_state(batch.next)
        aligned_targets = target_states[batch.current_next_indices]

        wm_loss = F.mse_loss(
            model_output.predicted_next_state,
            aligned_targets,
        )
        value = self._value_loss(
            model_output.action_values,
            batch.current_action_indices,
            batch.current_value_targets,
            include_ranking=include_value_ranking,
        )
        total = wm_weight * wm_loss + self.value_weight * value["loss"]
        if model_output.lm_loss is not None:
            total = total + self.ce_weight * model_output.lm_loss
        sample_count = 0 if batch.is_padding else batch.batch_size
        if batch.is_padding:
            total = total * 0.0

        metrics = {
            "value_reg": float(value["regression"].detach().item()),
            "value_rank": float(value["ranking"].detach().item()),
            "value_total": float(value["loss"].detach().item()),
            "lambda_wm": float(wm_weight),
            "lambda_sigreg": 0.0,
            "lambda_value": self.value_weight,
            "lambda_ce": self.ce_weight,
            "context_length": float(batch.history_size),
            "current_batch_size": float(sample_count),
            "history_cache_entries": float(runtime.history_cache.count),
            "total_loss": float(total.detach().item()),
        }
        metrics["wm_mse"] = float(wm_loss.detach().item())
        if model_output.lm_loss is not None:
            metrics["lm_ce"] = float(model_output.lm_loss.detach().item())

        return SFT2StepOutput(
            loss=total,
            losses={
                "lm": model_output.lm_loss,
                "wm": wm_loss,
                "value": value["loss"],
            },
            metrics=metrics,
            current_state=model_output.state[:, -1],
            sample_count=sample_count,
        )

    def _value_loss(
        self,
        all_values: torch.Tensor,
        action_indices: torch.Tensor,
        return_targets: torch.Tensor,
        *,
        include_ranking: bool,
    ) -> dict[str, torch.Tensor]:
        """Value head loss"""

        chosen_values = all_values.gather(
            -1,
            action_indices.unsqueeze(-1),
        ).squeeze(-1)
        targets = return_targets.to(device=all_values.device, dtype=all_values.dtype)
        regression = F.mse_loss(chosen_values, targets)
        chosen_mask = F.one_hot(
            action_indices,
            num_classes=all_values.shape[-1],
        ).bool()
        max_other = all_values.masked_fill(
            chosen_mask,
            float("-inf"),
        ).max(dim=-1).values
        ranking = F.relu(
            self.value_rank_margin + max_other - chosen_values
        ).mean()
        rank_weight = self.value_rank_weight if include_ranking else 0.0
        return {
            "loss": regression + rank_weight * ranking,
            "regression": regression,
            "ranking": ranking,
            "chosen_values": chosen_values,
        }

    def _sigreg_loss(
        self,
        current_state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor | None:
        """为当前 transition 的在线 ``(s_t,s_{t+1})`` 计算一次 SIGReg。"""

        if self.sigreg is None or self.sigreg_weight <= 0.0:
            return None
        if current_state.ndim != 2 or next_state.ndim != 2:
            raise ValueError(
                "SFT2 SIGReg expects current_state/next_state=(B,D), "
                f"got {tuple(current_state.shape)} and {tuple(next_state.shape)}"
            )
        if current_state.shape != next_state.shape:
            raise ValueError("SFT2 SIGReg state batch sizes do not match")
        return self.sigreg(torch.stack((current_state, next_state), dim=1))

__all__ = [
    "SFT2Algorithm",
    "SFT2SIGRegStepOutput",
    "SFT2StepOutput",
    "gather_global_sigreg_states",
    "require_sft2_wm_history",
    "shared_sigreg_rng",
]
