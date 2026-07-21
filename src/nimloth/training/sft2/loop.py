"""SFT2 训练循环及其可恢复状态。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from nimloth.config.sft2 import SFT2LoopConfig
from nimloth.rollout import TransitionBatchBuilder
from nimloth.training.sft2.checkpoint import (
    SFT2CheckpointRuntime,
    resume_epoch_and_micro_step,
)
from nimloth.training.sft2.algorithm import SFT2Algorithm
from nimloth.training.sft2.evaluate import evaluate
from nimloth.training.sft2.runtime import (
    SFT2ModelRuntime,
    SFT2OptimizationRuntime,
)
from nimloth.training.sft2.reporting import SFT2Reporter
from nimloth.training.sft2.utils import seed_training_micro_step
from nimloth.util.distributed import is_main
from nimloth.util.metrics import MetricAccumulator
from nimloth.util.profiling import StepTimer


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

    config: SFT2LoopConfig
    rank: int
    train_loader: Any
    val_loader: Any
    train_sampler: Any
    train_batch_sampler: Any
    algorithm: SFT2Algorithm
    model_runtime: SFT2ModelRuntime
    optimization_runtime: SFT2OptimizationRuntime
    batch_builder: TransitionBatchBuilder
    checkpoint_runtime: SFT2CheckpointRuntime
    reporter: SFT2Reporter
    state: SFT2LoopState
    total_steps: int
    step_timer: StepTimer = field(init=False)

    def __post_init__(self) -> None:
        self.step_timer = StepTimer(
            enabled=self.config.step_timing,
            log_interval=self.config.step_timing_interval,
        )

    def run(self) -> SFT2LoopState:
        """执行剩余 epoch，并返回最终可保存状态。"""

        for epoch in range(self.state.start_epoch, self.config.epochs + 1):
            self._run_epoch(epoch)

        self.checkpoint_runtime.save_final(
            step=self.state.global_step,
            epoch=self.config.epochs,
            best_val_wm_mse=self.state.best_val_wm_mse,
        )
        return self.state

    def _run_epoch(self, epoch: int) -> None:
        """执行一个 epoch，并在全部 rank 完成后统一验证。"""

        self._set_sampler_epoch(epoch)
        self.optimization_runtime.zero_grad()
        accumulator = MetricAccumulator()
        train_iterator, micro_index = self._resume_train_iterator(epoch)
        micro_batch_count = len(self.train_loader)
        while True:
            timer_start = self.step_timer.start("dataloader")
            try:
                batch_samples = next(train_iterator)
            except StopIteration:
                break
            self.step_timer.stop("dataloader", timer_start)

            micro_index += 1
            seed_training_micro_step(self.config.seed, epoch, micro_index, self.rank)
            sync_gradients = (
                micro_index % self.config.grad_accum == 0
                or micro_index == micro_batch_count
            )
            with self.optimization_runtime.accumulation_context(
                sync_gradients=sync_gradients,
            ):
                lambda_wm, metrics, loss = self._forward_loss(batch_samples)
                timer_start = self.step_timer.start("backward")
                self.optimization_runtime.backward(
                    loss,
                    grad_accum=self.config.grad_accum,
                )
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
                self.checkpoint_runtime.save_periodic(
                    step=self.state.global_step,
                    epoch=epoch,
                    micro_step=micro_index,
                    best_val_wm_mse=self.state.best_val_wm_mse,
                )

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
        if consumed % self.config.grad_accum != 0 and consumed != micro_batch_count:
            raise ValueError(
                "partial-epoch checkpoint was not saved at an optimizer boundary: "
                f"micro_step={consumed}, grad_accum={self.config.grad_accum}"
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
        lambda_wm = self.algorithm.wm_weight(
            self.state.global_step,
            self.total_steps,
        )
        batch = self.batch_builder.prepare(batch_samples)
        step_output = self.algorithm.training_step(
            self.model_runtime,
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

        qwen_lr = self.optimization_runtime.step(
            global_step=self.state.global_step,
        )
        self.state.global_step += 1

        averages = accumulator.averages()
        accumulator.reset()
        self.reporter.log_train_step(
            epoch=epoch,
            global_step=self.state.global_step,
            metrics=averages,
            wm_weight=lambda_wm,
            sigreg_weight=self.config.lambda_sigreg,
            qwen_lr=qwen_lr,
        )

    def _validate_and_checkpoint(self, epoch: int) -> None:
        """验证当前模型，并根据 WM MSE 更新 epoch/best checkpoint。"""

        val_metrics = evaluate(
            self.algorithm,
            self.model_runtime,
            self.val_loader,
            batch_builder=self.batch_builder,
            max_batches=self.config.max_val_batches,
        )
        val_wm_mse = val_metrics.get("wm_mse", float("inf"))
        improved = val_wm_mse < self.state.best_val_wm_mse
        if improved:
            self.state.best_val_wm_mse = val_wm_mse
        self.checkpoint_runtime.save_epoch(
            step=self.state.global_step,
            epoch=epoch,
            best_val_wm_mse=self.state.best_val_wm_mse,
            improved=improved,
        )
        self.reporter.log_validation(
            epoch=epoch,
            global_step=self.state.global_step,
            metrics=val_metrics,
            best_val_wm_mse=self.state.best_val_wm_mse,
            checkpoint_metric=self.config.checkpoint_metric,
        )

    @staticmethod
    def _barrier() -> None:
        """仅在分布式进程组已初始化时同步。"""

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
