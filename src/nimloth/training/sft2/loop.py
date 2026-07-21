"""SFT2 训练循环及其可恢复状态。"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from nimloth.training.sft2.checkpoint import (
    SFT2CheckpointManager,
    read_checkpoint_step,
    resume_epoch_and_micro_step,
)
from nimloth.training.sft2.components import SFT2Components
from nimloth.training.sft2.algorithm import SFT2Algorithm
from nimloth.training.sft2.schedule import wm_loss_weight_schedule
from nimloth.training.sft2.evaluate import evaluate
from nimloth.training.sft2.utils import no_sync_if_needed, seed_training_micro_step
from nimloth.util.csv_log import CSVRecordWriter
from nimloth.util.distributed import is_main
from nimloth.util.metrics import MetricAccumulator
from nimloth.util.optim import qwen_lr_schedule, set_optimizer_group_lr
from nimloth.util.profiling import StepTimer
from nimloth.util.wandb import log_metrics


@dataclass
class SFT2LoopState:
    """训练循环中必须跨 checkpoint 保留的位置与指标。"""

    global_step: int = 0
    best_val_wm_mse: float = float("inf")
    start_epoch: int = 1
    resume_micro_step: int = 0


def load_sft2_loop_state(
    *,
    resume: bool,
    resume_state_path: Path | None,
    resume_checkpoint_dir: Path | None,
    optimizer: torch.optim.Optimizer,
    training_invariants: dict[str, Any],
) -> SFT2LoopState:
    """读取训练位置并校验影响数据顺序和梯度语义的不变量。"""

    loop_state = SFT2LoopState()
    if not resume or resume_state_path is None or not resume_state_path.exists():
        return loop_state

    saved_state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
    loop_state.global_step = int(saved_state.get("step", 0))
    loop_state.best_val_wm_mse = float(
        saved_state.get("best_val_wm_mse", saved_state.get("best_val", float("inf")))
    )

    saved_invariants = saved_state.get("training_invariants")
    if saved_invariants is not None:
        mismatches = {
            key: (saved_invariants.get(key), current_value)
            for key, current_value in training_invariants.items()
            if saved_invariants.get(key) != current_value
        }
        if mismatches:
            raise ValueError(f"resume training invariants mismatch: {mismatches}")

    if "epoch" in saved_state:
        loop_state.start_epoch, loop_state.resume_micro_step = (
            resume_epoch_and_micro_step(saved_state)
        )
    if saved_state.get("optimizer") is not None:
        optimizer.load_state_dict(saved_state["optimizer"])

    if is_main():
        print(
            json.dumps(
                {
                    "resume": True,
                    "resume_ckpt": str(resume_checkpoint_dir),
                    "start_epoch": loop_state.start_epoch,
                    "global_step": loop_state.global_step,
                    "resume_micro_step": loop_state.resume_micro_step,
                    "best_val_wm_mse": loop_state.best_val_wm_mse,
                }
            )
        )
    return loop_state


@dataclass
class SFT2TrainingLoop:
    """执行 SFT2 的微批训练、验证和 checkpoint 生命周期。"""

    args: Any
    rank: int
    world_size: int
    device: torch.device
    train_loader: Any
    val_loader: Any
    train_sampler: Any
    train_batch_sampler: Any
    components: SFT2Components
    algorithm: SFT2Algorithm
    checkpoint_manager: SFT2CheckpointManager
    log_writer: CSVRecordWriter
    wandb_run: Any
    state: SFT2LoopState
    train_wm_predictor: bool
    total_steps: int
    qwen_warmup_steps: int
    llm_tune: str
    vision_tune: str
    step_timer: StepTimer = field(init=False)
    last_periodic_checkpoint_time: float = field(init=False)

    def __post_init__(self) -> None:
        self.step_timer = StepTimer(
            enabled=self.args.step_timing,
            log_interval=self.args.step_timing_interval,
        )
        self.last_periodic_checkpoint_time = time.monotonic()

    def run(self) -> SFT2LoopState:
        """执行剩余 epoch，并返回最终可保存状态。"""

        for epoch in range(self.state.start_epoch, self.args.epochs + 1):
            self._run_epoch(epoch)

        if is_main():
            self.checkpoint_manager.save(
                "final",
                step=self.state.global_step,
                epoch=self.args.epochs,
                best_val_wm_mse=self.state.best_val_wm_mse,
            )
        return self.state

    def _run_epoch(self, epoch: int) -> None:
        """执行一个 epoch，并在全部 rank 完成后统一验证。"""

        self._set_sampler_epoch(epoch)
        optimizer = self.components.optimizer
        optimizer.zero_grad(set_to_none=True)
        accumulator = MetricAccumulator()
        train_iterator, micro_index = self._resume_train_iterator(epoch)
        micro_batch_count = len(self.train_loader)
        agent = self.components.agent
        ddp_modules = [agent.backbone.model, agent.wm.state_proj, agent.wm.value_head]
        if self.train_wm_predictor:
            ddp_modules.append(agent.wm.wm_predictor)

        # PyTorch 2.8 的 static_graph=True 与 no_sync() 组合存在 Reducer
        # 回归。保留重复 Qwen forward 所需的静态图，并在每个微批同步梯度。
        use_ddp_no_sync = (
            self.world_size > 1
            and not self.components.qwen_pair_parallel
            and not self.components.ddp_static_graph
        )
        if (
            is_main()
            and self.world_size > 1
            and self.args.grad_accum > 1
            and not use_ddp_no_sync
        ):
            print(
                json.dumps(
                    {
                        "ddp_gradient_accumulation": "sync_each_microbatch",
                        "reason": "torch_2_8_static_graph_no_sync_regression",
                    }
                )
            )

        while True:
            timer_start = self.step_timer.start("dataloader")
            try:
                batch_samples = next(train_iterator)
            except StopIteration:
                break
            self.step_timer.stop("dataloader", timer_start)

            micro_index += 1
            seed_training_micro_step(self.args.seed, epoch, micro_index, self.rank)
            sync_gradients = (
                micro_index % self.args.grad_accum == 0
                or micro_index == micro_batch_count
            )
            with no_sync_if_needed(
                ddp_modules,
                enabled=not sync_gradients and use_ddp_no_sync,
            ):
                lambda_wm, metrics, loss = self._forward_loss(batch_samples)
                timer_start = self.step_timer.start("backward")
                (loss / self.args.grad_accum).backward()
                self.step_timer.stop("backward", timer_start)
            accumulator.update(metrics)

            if sync_gradients:
                timer_start = self.step_timer.start("optimizer")
                self._optimizer_step(epoch, accumulator, lambda_wm=lambda_wm)
                self.step_timer.stop("optimizer", timer_start)
                self.step_timer.on_optimizer_step(
                    global_step=self.state.global_step,
                    epoch=epoch,
                )
                self._save_periodic_checkpoints(epoch, micro_index)

        self._barrier()
        self._validate_and_checkpoint(epoch)
        self._barrier()

    def _set_sampler_epoch(self, epoch: int) -> None:
        """让分布式 sampler 使用一致的 epoch 随机种子。"""

        if self.train_batch_sampler is not None:
            self.train_batch_sampler.set_epoch(epoch)
        elif self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

    def _resume_train_iterator(self, epoch: int) -> tuple[Any, int]:
        """恢复首个 epoch 内已经消费过的微批位置。"""

        train_iterator = iter(self.train_loader)
        if epoch != self.state.start_epoch or not self.state.resume_micro_step:
            return train_iterator, 0

        consumed = self.state.resume_micro_step
        micro_batch_count = len(self.train_loader)
        if consumed > micro_batch_count:
            raise ValueError(
                "checkpoint micro_step_in_epoch exceeds current DataLoader length: "
                f"{consumed} > {micro_batch_count}"
            )
        if consumed % self.args.grad_accum != 0 and consumed != micro_batch_count:
            raise ValueError(
                "partial-epoch checkpoint was not saved at an optimizer boundary: "
                f"micro_step={consumed}, grad_accum={self.args.grad_accum}"
            )
        for _ in range(consumed):
            next(train_iterator)
        if is_main():
            print(
                json.dumps(
                    {
                        "resume_data_position": {
                            "epoch": epoch,
                            "skipped_micro_batches": consumed,
                            "total_micro_batches": micro_batch_count,
                        }
                    }
                )
            )
        return train_iterator, consumed

    def _forward_loss(self, batch_samples: Any) -> tuple[float, dict[str, float], Any]:
        """运行共享 forward，并按当前训练步组合全部目标函数。"""

        timer_start = self.step_timer.start("forward")
        lambda_wm = wm_loss_weight_schedule(
            self.state.global_step,
            self.total_steps,
            start=self.args.lambda_wm_start,
            end=self.args.lambda_wm_end,
        )
        batch = self.components.batch_builder.prepare(batch_samples)
        step_output = self.algorithm.training_step(
            batch,
            wm_weight=lambda_wm,
        )
        self.step_timer.stop("forward", timer_start)
        return lambda_wm, step_output.metrics, step_output.loss

    def _optimizer_step(
        self,
        epoch: int,
        accumulator: MetricAccumulator,
        *,
        lambda_wm: float,
    ) -> None:
        """更新学习率和所有训练模块，并记录聚合后的微批指标。"""

        optimizer = self.components.optimizer
        qwen_lr = qwen_lr_schedule(
            self.state.global_step,
            warmup_steps=self.qwen_warmup_steps,
            total_steps=self.total_steps,
            start_lr=self.args.lr_qwen_start,
            peak_lr=self.args.lr_qwen_peak,
        )
        set_optimizer_group_lr(optimizer, "qwen", qwen_lr)
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]],
            1.0,
        )
        optimizer.step()
        if self.components.vision_ema is not None:
            self.components.vision_ema.update(self.components.agent.backbone.model)
        optimizer.zero_grad(set_to_none=True)
        self.state.global_step += 1

        averages = accumulator.averages()
        accumulator.reset()
        if not is_main():
            return
        self.log_writer.append(
            {
                "time": time.time(),
                "epoch": epoch,
                "global_step": self.state.global_step,
                **averages,
                "lambda_wm": lambda_wm,
                "lambda_sigreg": self.args.lambda_sigreg,
                "qwen_lr": qwen_lr,
            }
        )
        log_metrics(
            self.wandb_run,
            namespace="train",
            metrics=averages,
            step=self.state.global_step,
            context={"global_step": self.state.global_step},
        )

    def _save_periodic_checkpoints(self, epoch: int, micro_index: int) -> None:
        """在优化器边界保存按步和按时间触发的 checkpoint。"""

        save_step = bool(
            self.args.checkpoint_interval_steps
            and self.args.checkpoint_interval_steps > 0
            and self.state.global_step % self.args.checkpoint_interval_steps == 0
        )
        save_latest = False
        if self.args.checkpoint_interval_minutes > 0:
            if is_main():
                elapsed = time.monotonic() - self.last_periodic_checkpoint_time
                save_latest = elapsed >= self.args.checkpoint_interval_minutes * 60.0
            save_latest = self._broadcast_bool(save_latest)

        if save_latest:
            self._barrier()
            if is_main():
                self.checkpoint_manager.save(
                    "latest",
                    step=self.state.global_step,
                    epoch=epoch,
                    best_val_wm_mse=self.state.best_val_wm_mse,
                    epoch_complete=False,
                    micro_step_in_epoch=micro_index,
                )
                self.last_periodic_checkpoint_time = time.monotonic()
            self._barrier()

        if save_step:
            self._barrier()
            if is_main():
                self.checkpoint_manager.save(
                    f"step_{self.state.global_step:06d}",
                    step=self.state.global_step,
                    epoch=epoch,
                    best_val_wm_mse=self.state.best_val_wm_mse,
                    epoch_complete=False,
                    micro_step_in_epoch=micro_index,
                )
                self._prune_step_checkpoints()
            self._barrier()

    def _prune_step_checkpoints(self) -> None:
        """只保留最近若干个按步保存的 checkpoint。"""

        keep = int(getattr(self.args, "checkpoint_keep_last", 0) or 0)
        if keep <= 0:
            return
        checkpoints = sorted(
            (
                (read_checkpoint_step(path), path)
                for path in self.args.output_dir.glob("step_*")
                if path.is_dir()
                and path.name.startswith("step_")
                and (path / "training_state.pt").is_file()
            ),
            key=lambda item: item[0],
        )
        for _, path in checkpoints[:-keep]:
            shutil.rmtree(path, ignore_errors=True)

    def _validate_and_checkpoint(self, epoch: int) -> None:
        """验证当前模型，并根据 WM MSE 更新 epoch/best checkpoint。"""

        val_metrics = evaluate(
            self.algorithm,
            self.val_loader,
            batch_builder=self.components.batch_builder,
            max_batches=self.args.max_val_batches,
        )
        if not is_main():
            return

        val_wm_mse = val_metrics.get("wm_mse", float("inf"))
        log_metrics(
            self.wandb_run,
            namespace="val",
            metrics=val_metrics,
            step=self.state.global_step,
            context={"epoch": epoch},
        )
        self.log_writer.append(
            {
                "time": time.time(),
                "epoch": epoch,
                "global_step": self.state.global_step,
                **val_metrics,
                "val_wm_mse": val_metrics.get("wm_mse", ""),
            }
        )
        improved = val_wm_mse < self.state.best_val_wm_mse
        if improved:
            self.state.best_val_wm_mse = val_wm_mse
        self.checkpoint_manager.save(
            f"epoch_{epoch:03d}",
            step=self.state.global_step,
            epoch=epoch,
            best_val_wm_mse=self.state.best_val_wm_mse,
        )
        if improved:
            self.checkpoint_manager.save(
                "best",
                step=self.state.global_step,
                epoch=epoch,
                best_val_wm_mse=self.state.best_val_wm_mse,
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "global_step": self.state.global_step,
                    "val_metrics": val_metrics,
                    "best_val_wm_mse": self.state.best_val_wm_mse,
                    "checkpoint_metric": self.args.checkpoint_metric,
                    "llm_tune": self.llm_tune,
                    "vision_tune": self.vision_tune,
                }
            )
        )

    def _broadcast_bool(self, value: bool) -> bool:
        """让 rank 0 决定基于墙钟时间的保存时机。"""

        if not (dist.is_available() and dist.is_initialized()):
            return value
        flag = torch.tensor([1 if value else 0], device=self.device, dtype=torch.int32)
        dist.broadcast(flag, src=0)
        return bool(flag.item())

    @staticmethod
    def _barrier() -> None:
        """仅在分布式进程组已初始化时同步。"""

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
