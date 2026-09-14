"""SFT2 训练循环及其可恢复状态。"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from nimloth.config.sft2 import SFT2LoopConfig
from nimloth.training.sft.stage3.batch import SFT2BatchBuilder
from nimloth.training.sft.stage3.checkpoint import (
    SFT2CheckpointRuntime,
    resume_epoch_and_micro_step,
)
from nimloth.training.sft.stage3.algorithm import SFT2Algorithm
from nimloth.training.sft.stage3.evaluate import evaluate, distributed_metric_averages
from nimloth.training.sft.stage3.runtime import (
    SFT2ModelRuntime,
    SFT2OptimizationRuntime,
)
from nimloth.training.sft.stage3.reporting import SFT2Reporter
from nimloth.training.sft.stage3.utils import global_sigreg_seed, seed_training_micro_step
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
    stopped: bool = False


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
    if training_invariants.get("outcome_schema") is not None and saved_invariants is None:
        raise ValueError("outcome resume requires saved training invariants")
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
    train_batch_sampler: Any
    algorithm: SFT2Algorithm
    model_runtime: SFT2ModelRuntime
    optimization_runtime: SFT2OptimizationRuntime
    batch_builder: SFT2BatchBuilder
    checkpoint_runtime: SFT2CheckpointRuntime
    reporter: SFT2Reporter
    state: SFT2LoopState
    total_steps: int
    outcome_eval_dir: Path | None = None
    outcome_export_identity: dict | None = None
    step_timer: StepTimer = field(init=False)

    def __post_init__(self) -> None:
        self.step_timer = StepTimer(
            enabled=self.config.step_timing,
            log_interval=self.config.step_timing_interval,
        )

    def run(self) -> SFT2LoopState:
        """执行剩余 epoch，并返回最终可保存状态。"""

        cap = getattr(self.config, "stop_after_steps", 0)
        if cap and self.state.global_step >= cap:
            raise ValueError("stop_after_steps must exceed the restored global step")
        if self.outcome_eval_dir is not None and self.state.global_step == 0:
            self.model_runtime.history_cache.start(epoch=0, phase="val")
            self._evaluate_export(self.val_loader, epoch=0, split="eval")
        for epoch in range(self.state.start_epoch, self.config.epochs + 1):
            self._run_epoch(epoch)
            if self.state.stopped:
                return self.state

        self.checkpoint_runtime.save_final(
            step=self.state.global_step,
            epoch=self.config.epochs,
            best_val_wm_mse=self.state.best_val_wm_mse,
        )
        return self.state

    def _run_epoch(self, epoch: int) -> None:
        """执行一个 epoch，并在全部 rank 完成后统一验证。"""

        self._set_sampler_epoch(epoch)
        resuming_epoch = (
            epoch == self.state.start_epoch and self.state.resume_micro_step > 0
        )
        self.model_runtime.history_cache.start(
            epoch=epoch,
            phase="train",
            resume=resuming_epoch,
        )
        self.optimization_runtime.zero_grad()
        accumulator = MetricAccumulator()
        train_iterator, micro_index = self._resume_train_iterator(epoch)
        micro_batch_count = len(self.train_loader)
        def normalized_batches():
            while group := list(itertools.islice(train_iterator, self.config.grad_accum)):
                counts = [self.batch_builder.supervision_counts(item) for item in group]
                outcome_counts = ([self.batch_builder.outcome_count(item) for item in group]
                                  if getattr(self.algorithm, "outcome_weight", 0) > 0 else [0] * len(group))
                totals = torch.tensor([sum(n for n, _ in counts), sum(n for _, n in counts), sum(outcome_counts)],
                                      device=self.batch_builder.device)
                world = 1
                if dist.is_available() and dist.is_initialized():
                    world = dist.get_world_size()
                    dist.all_reduce(totals)
                for item, (all_count, lm_count), outcome_count in zip(group, counts, outcome_counts, strict=True):
                    yield item, (world * all_count / max(1, int(totals[0])),
                                 world * lm_count / max(1, int(totals[1])), lm_count,
                                 world * outcome_count / max(1, int(totals[2])), outcome_count)

        normalized_iterator = iter(normalized_batches())
        while True:
            timer_start = self.step_timer.start("dataloader")
            try:
                batch_samples, loss_scales = next(normalized_iterator)
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
                lambda_wm, metrics, sample_count = self._train_microbatch(
                    batch_samples,
                    epoch=epoch,
                    micro_step=micro_index,
                    loss_scales=(loss_scales[0], loss_scales[1], loss_scales[3]),
                )
            if sample_count > 0:
                lm_metric = metrics.pop("lm_ce", None)
                if lm_metric is not None and loss_scales[2] > 0:
                    accumulator.update({"lm_ce": lm_metric}, count=loss_scales[2])
                outcome_metric = metrics.pop("outcome_bce", None)
                if outcome_metric is not None:
                    accumulator.update({"outcome_bce": outcome_metric}, count=loss_scales[4])
                accumulator.update(metrics, count=sample_count)

            if sync_gradients:
                timer_start = self.step_timer.start("optimizer")
                self._optimizer_step(epoch, accumulator, lambda_wm=lambda_wm)
                self.step_timer.stop("optimizer", timer_start)
                self.step_timer.on_optimizer_step(
                    global_step=self.state.global_step,
                    epoch=epoch,
                )
                cap = getattr(self.config, "stop_after_steps", 0)
                if cap and self.state.global_step >= cap:
                    self.checkpoint_runtime.save_stopped(
                        step=self.state.global_step, epoch=epoch, micro_step=micro_index,
                        best_val_wm_mse=self.state.best_val_wm_mse,
                    )
                    self.state.stopped = True
                    return
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

        self.train_batch_sampler.set_epoch(epoch)

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

    def _train_microbatch(
        self,
        batch_samples: Any,
        *,
        epoch: int,
        micro_step: int,
        loss_scales: tuple[float, ...] | None = None,
    ) -> tuple[float, dict[str, float], int]:
        """先反传单次 CE/WM/value，再构建并反传单向 SIGReg 图。"""

        timer_start = self.step_timer.start("forward_primary")
        lambda_wm = self.algorithm.wm_weight(
            self.state.global_step,
            self.total_steps,
        )
        batch = self.batch_builder.prepare(batch_samples)
        if (getattr(self.config, "diagnose_outcome_gradients", False)
                and self.state.global_step == 0 and micro_step == 1):
            from nimloth.training.sft.stage3.diagnostics import outcome_gradient_diagnostic
            diagnostic = outcome_gradient_diagnostic(
                self.algorithm, self.model_runtime, batch, wm_weight=lambda_wm,
            )
            diagnostic.update(rank=self.rank, epoch=epoch, micro_step=micro_step)
            path = self.checkpoint_runtime.manager.output_dir / f"outcome_gradients_rank_{self.rank:03d}.json"
            with path.open("x") as stream:
                json.dump(diagnostic, stream, indent=2, allow_nan=False)
            print(json.dumps(diagnostic, allow_nan=False), flush=True)
        primary = self.algorithm.training_primary_step(
            self.model_runtime,
            batch,
            wm_weight=lambda_wm,
        )
        self.step_timer.stop("forward_primary", timer_start)

        detached_current_state = primary.current_state.detach()
        primary_metrics = primary.metrics
        sample_count = primary.sample_count
        timer_start = self.step_timer.start("backward_primary")
        primary_loss = primary.loss
        divisor = self.config.grad_accum
        if loss_scales is not None:
            lm = primary.losses["lm"]
            lm_term = self.algorithm.ce_weight * lm if lm is not None else 0
            outcome = primary.losses.get("outcome")
            outcome_term = self.algorithm.outcome_weight * outcome if outcome is not None else 0
            primary_loss = ((primary_loss - lm_term - outcome_term) * loss_scales[0]
                            + lm_term * loss_scales[1]
                            + outcome_term * (loss_scales[2] if len(loss_scales) > 2 else loss_scales[0]))
            divisor = 1
        self.optimization_runtime.backward(primary_loss, grad_accum=divisor)
        del primary_loss
        if loss_scales is not None:
            del lm, lm_term, outcome, outcome_term

        self.step_timer.stop("backward_primary", timer_start)
        # 不让任何主阶段 Tensor 引用跨入下一次 Qwen forward。
        del primary

        sigreg = None
        if self.algorithm.has_sigreg_stage:
            timer_start = self.step_timer.start("forward_sigreg")
            sigreg = self.algorithm.training_sigreg_step(
                self.model_runtime,
                batch,
                detached_current_state=detached_current_state,
                sigreg_seed=global_sigreg_seed(
                    self.config.seed,
                    epoch,
                    micro_step,
                ),
            )
            self.step_timer.stop("forward_sigreg", timer_start)
            timer_start = self.step_timer.start("backward_sigreg")
            self.optimization_runtime.backward(
                sigreg.loss,
                grad_accum=self.config.grad_accum,
            )
            self.step_timer.stop("backward_sigreg", timer_start)

        metrics = self.algorithm.merge_training_metrics(primary_metrics, sigreg)
        return lambda_wm, metrics, sample_count

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

        averages = distributed_metric_averages(accumulator)
        accumulator.reset()
        averages["total_loss"] = (
            lambda_wm * averages.get("wm_mse", 0.)
            + self.algorithm.value_weight * averages.get("value_total", 0.)
            + self.algorithm.dino_grid_weight * averages.get("dino_grid_mse", 0.)
            + self.algorithm.ce_weight * averages.get("lm_ce", 0.)
            + self.algorithm.sigreg_weight * averages.get("sigreg_loss", 0.)
            + self.algorithm.outcome_weight * averages.get("outcome_bce", 0.)
        )
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

        self.model_runtime.history_cache.start(epoch=epoch, phase="val")
        val_metrics = self._evaluate_export(self.val_loader, epoch=epoch, split="eval")
        if self.outcome_eval_dir is not None:
            self.model_runtime.history_cache.start(epoch=epoch, phase="probe_train")
            self._evaluate_export(self.train_loader, epoch=epoch, split="train")
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
        if self.outcome_eval_dir is not None:
            self._seal_exports(epoch)
        self.reporter.log_validation(
            epoch=epoch,
            global_step=self.state.global_step,
            metrics=val_metrics,
            best_val_wm_mse=self.state.best_val_wm_mse,
            checkpoint_metric=self.config.checkpoint_metric,
        )

    def _evaluate_export(self, loader, *, epoch: int, split: str):
        writer = None
        if self.outcome_eval_dir is not None:
            from nimloth.eval.stage3_outcome import OutcomeRowsWriter
            directory = Path(self.outcome_eval_dir)
            directory.mkdir(parents=True, exist_ok=True)
            writer = OutcomeRowsWriter(
                directory / f"epoch_{epoch:03d}_{split}_rank_{self.rank:03d}.jsonl",
                outcome_available=self.algorithm.outcome_weight > 0,
            )
        try:
            metrics = evaluate(self.algorithm, self.model_runtime, loader,
                            batch_builder=self.batch_builder,
                            max_batches=self.config.max_val_batches if split == "eval" else -1,
                            on_batch=writer)
        finally:
            if writer is not None:
                writer.close()
        if writer is not None:
            self._publish_export_manifest(loader, epoch=epoch, split=split, writer=writer)
        return metrics

    def _publish_export_manifest(self, loader, *, epoch, split, writer):
        from nimloth.eval.stage3_outcome import file_sha256
        expected = int(loader.batch_sampler.window_count) * self.algorithm.prediction_horizon
        local = {"name": Path(writer.stream.name).name, "rows": writer.row_count, "sha256": file_sha256(writer.stream.name)}
        files = [local]
        if dist.is_available() and dist.is_initialized():
            files = [None] * dist.get_world_size()
            dist.all_gather_object(files, local)
        if sum(item["rows"] for item in files) != expected:
            raise ValueError("export did not cover all expected windows")
        if not self.outcome_export_identity:
            raise ValueError("export requires dataset and initialization identity")
        if self.rank == 0:
            manifest = {"status": "pending_checkpoint", "expected_rows": expected, "files": files,
                        "identity": self.outcome_export_identity, "epoch": epoch, "step": self.state.global_step}
            if epoch == 0:
                manifest.update(status="complete", checkpoint_identity={"initialization": self.outcome_export_identity})
            path = Path(self.outcome_eval_dir) / f"epoch_{epoch:03d}_{split}.complete.json"
            with path.open("x") as stream:
                json.dump(manifest, stream, indent=2)

    def _seal_exports(self, epoch):
        if self.rank != 0:
            return
        from nimloth.eval.stage3_outcome import file_sha256
        checkpoint = self.checkpoint_runtime.manager.output_dir / f"epoch_{epoch:03d}"
        state_path = checkpoint / "training_state.pt"
        identity = {"path": str(checkpoint), "training_state_sha256": file_sha256(state_path),
                    "step": self.state.global_step, "epoch": epoch}
        for split in ("train", "eval"):
            path = Path(self.outcome_eval_dir) / f"epoch_{epoch:03d}_{split}.complete.json"
            manifest = json.loads(path.read_text())
            manifest.update(status="complete", checkpoint_identity=identity)
            temporary = path.with_suffix(".tmp")
            with temporary.open("x") as stream:
                json.dump(manifest, stream, indent=2)
            temporary.replace(path)

    @staticmethod
    def _barrier() -> None:
        """仅在分布式进程组已初始化时同步。"""

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
