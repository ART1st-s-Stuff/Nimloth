"""SFT2 训练循环及其可恢复状态。"""

from __future__ import annotations

import itertools
import json
from nimloth.training.sft.stage3.early_stop import initialize_early_stop, update_early_stop
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from nimloth.config.sft2 import SFT2LoopConfig
from nimloth.training.common.activation_offload import saved_activation_context
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
    early_stop_state: dict | None = None
    converged: bool = False


def load_sft2_loop_state(
    *,
    resume: bool,
    resume_state_path: Path | None,
    resume_checkpoint_dir: Path | None,
    optimizer: torch.optim.Optimizer,
    training_invariants: dict[str, Any],
    agent=None,
    legacy_schedule_total_steps: int | None = None,
) -> SFT2LoopState:
    """读取训练位置并校验影响数据顺序和梯度语义的不变量。"""

    loop_state = SFT2LoopState()
    if not resume or resume_state_path is None or not resume_state_path.exists():
        return loop_state

    saved_state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
    loop_state.global_step = int(saved_state.get("step", 0))
    loop_state.early_stop_state = saved_state.get("early_stop_state")
    if "early_stop_contract" in saved_state:
        if not isinstance(loop_state.early_stop_state, dict) or any(
                loop_state.early_stop_state.get(key) != value
                for key, value in saved_state["early_stop_contract"].items()):
            raise ValueError("checkpoint early stopping state missing or inconsistent")
    loop_state.best_val_wm_mse = float(
        saved_state.get("best_val_wm_mse", saved_state.get("best_val", float("inf")))
    )

    saved_invariants = saved_state.get("training_invariants")
    if not isinstance(saved_invariants, dict) or saved_invariants.get("training_unit") != "complete_trajectory_v1":
        raise ValueError("trajectory-native resume requires complete_trajectory_v1 training invariants")
    if saved_invariants is not None:
        # Before this switch existed all WM/value inputs were connected. Preserve
        # those resumes while rejecting either direction of a changed boundary.
        saved_invariants = {"wm_value_backbone_grad": True, **saved_invariants}
        if "schedule_total_steps" not in saved_invariants and "schedule_total_steps" in training_invariants:
            if legacy_schedule_total_steps is None:
                raise ValueError("legacy checkpoint requires explicit --schedule-total-steps for resume")
            saved_invariants["schedule_total_steps"] = legacy_schedule_total_steps
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
        state = saved_state["optimizer"]
        if agent is not None:
            from nimloth.training.sft.stage3.fsdp_checkpoint import optimizer_state_to_load
            state = optimizer_state_to_load(agent, optimizer, state)
        optimizer.load_state_dict(state)

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
    feature_export_dir: Path | None = None
    feature_export_identity: dict | None = None
    feature_export_step: int | None = None
    diagnostic_steps: tuple[int, ...] = ()
    diagnostic_dir: Path | None = None
    diagnostic_identity: dict | None = None
    frozen_wm_cache_dir: Path | None = None
    frozen_wm_cache_split: str | None = None
    frozen_wm_cache_identity: dict | None = None
    step_timer: StepTimer = field(init=False)

    def __post_init__(self) -> None:
        self.step_timer = StepTimer(
            enabled=self.config.step_timing,
            log_interval=self.config.step_timing_interval,
            sample_interval=getattr(self.config, "step_timing_sample_interval", 1),
        )

    def run(self) -> SFT2LoopState:
        """执行剩余 epoch，并返回最终可保存状态。"""

        self.state.early_stop_state = initialize_early_stop(self.config, self.state.early_stop_state)
        if self.state.early_stop_state is not None:
            self.checkpoint_runtime.manager = replace(
                self.checkpoint_runtime.manager, early_stop_state=self.state.early_stop_state)
        if (self.state.early_stop_state is not None and
                self.state.early_stop_state["bad_epochs"] >= self.config.early_stop_patience):
            raise ValueError("checkpoint has already converged; no further updates authorized by this stopping rule")
        cap = getattr(self.config, "stop_after_steps", 0)
        if cap and self.state.global_step >= cap:
            raise ValueError("stop_after_steps must exceed the restored global step")
        self._run_fixed_diagnostic()
        if self.outcome_eval_dir is not None and self.state.global_step == 0:
            self._evaluate_export(self.val_loader, epoch=0, split="eval")
        for epoch in range(self.state.start_epoch, self.config.epochs + 1):
            self._run_epoch(epoch)
            if self.state.stopped:
                return self.state
            if self.state.converged:
                self.checkpoint_runtime.save_final(step=self.state.global_step, epoch=epoch,
                    best_val_wm_mse=self.state.best_val_wm_mse)
                self._write_completion(epoch, "early_stop")
                return self.state

        self.checkpoint_runtime.save_final(
            step=self.state.global_step,
            epoch=self.config.epochs,
            best_val_wm_mse=self.state.best_val_wm_mse,
        )
        self._write_completion(self.config.epochs, "epoch_limit")
        return self.state

    def _write_completion(self, epoch: int, reason: str) -> None:
        if not is_main():
            return
        root = self.checkpoint_runtime.manager.output_dir
        pending = root / "training_complete.json.tmp"
        pending.write_text(json.dumps({"epoch": epoch, "step": self.state.global_step,
            "checkpoint": f"epoch_{epoch:03d}", "reason": reason,
            "early_stop_state": self.state.early_stop_state,
            "schedule_total_steps": self.total_steps}, indent=2) + "\n", encoding="utf-8")
        pending.replace(root / "training_complete.json")

    def evaluate_only(self) -> dict[str, float]:
        """Run the production validation forward without updating or saving weights."""
        if self.feature_export_dir is not None and self.frozen_wm_cache_dir is not None:
            raise ValueError("feature and frozen-WM exports must run separately")
        writer = None
        if self.feature_export_dir is not None:
            from nimloth.training.sft.stage3.diagnostics import DINOFeatureWriter

            writer = DINOFeatureWriter(
                self.feature_export_dir,
                rank=self.rank,
                step=(
                    self.feature_export_step
                    if self.feature_export_step is not None
                    else self.state.global_step
                ),
                identity=self.feature_export_identity,
            )
        elif self.frozen_wm_cache_dir is not None:
            from nimloth.training.sft.stage3.diagnostics import FrozenWMTrajectoryWriter

            writer = FrozenWMTrajectoryWriter(
                self.frozen_wm_cache_dir,
                rank=self.rank,
                identity=self.frozen_wm_cache_identity,
            )
        loader = self.val_loader
        if self.frozen_wm_cache_dir is not None:
            if self.frozen_wm_cache_split not in {"train", "eval"}:
                raise ValueError("frozen-WM cache export requires an explicit train/eval split")
            loader = self.train_loader if self.frozen_wm_cache_split == "train" else self.val_loader
        metrics = evaluate(
            self.algorithm,
            self.model_runtime,
            loader,
            batch_builder=self.batch_builder,
            max_batches=self.config.max_val_batches,
            on_batch=writer,
        )
        if writer is not None and hasattr(writer, "finalize"):
            if self.feature_export_dir is not None:
                writer.finalize(metrics=metrics)
            else:
                writer.finalize()
        if is_main():
            print(json.dumps({"eval_only": True, "metrics": metrics}), flush=True)
        return metrics

    def _run_epoch(self, epoch: int) -> None:
        """执行一个 epoch，并在全部 rank 完成后统一验证。"""

        self.model_runtime.set_training_mode()
        self._set_sampler_epoch(epoch)
        self.optimization_runtime.zero_grad()
        accumulator = MetricAccumulator()
        train_iterator, micro_index = self._resume_train_iterator(epoch)
        micro_batch_count = len(self.train_loader)
        def normalized_batches():
            while group := list(itertools.islice(train_iterator, self.config.grad_accum)):
                counts = [self.batch_builder.supervision_counts(item) for item in group]
                outcome_counts = ([self.batch_builder.outcome_count(item) for item in group]
                                  if getattr(self.algorithm, "outcome_weight", 0) > 0 else [0] * len(group))
                state_counts = [self.batch_builder.observed_state_count(item) for item in group]
                totals = torch.tensor([sum(n for n, _ in counts), sum(n for _, n in counts), sum(outcome_counts), sum(state_counts)],
                                      device=self.batch_builder.device)
                world = 1
                if dist.is_available() and dist.is_initialized():
                    world = dist.get_world_size()
                    dist.all_reduce(totals)
                for item, (all_count, lm_count), outcome_count, state_count in zip(group, counts, outcome_counts, state_counts, strict=True):
                    yield item, (world * all_count / max(1, int(totals[0])),
                                 world * lm_count / max(1, int(totals[1])), lm_count,
                                 world * outcome_count / max(1, int(totals[2])), outcome_count,
                                 1. / len(group), world * state_count / max(1, int(totals[3])), state_count)

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
                    loss_scales=(loss_scales[0], loss_scales[1], loss_scales[3], loss_scales[5], loss_scales[6]),
                )
            # Batch sizes describe a rank-local microbatch, including padding.
            # Weighting these counts by windows would report sum(W**2)/sum(W).
            metrics.pop("current_batch_size", None)
            accumulator.update({
                "current_batch_size": float(sample_count),
                "trajectory_batch_size": float(sum(item.loss_weight > 0 for item in batch_samples)),
            }, count=1)
            # SIGReg is a microbatch statistic, not a window-weighted mean.
            regularizer_metrics = {key: metrics.pop(key) for key in tuple(metrics)
                                   if key.startswith("sigreg_")}
            if regularizer_metrics:
                accumulator.update(regularizer_metrics, count=1)
            if sample_count > 0:
                dino_metric = metrics.pop("dino_grid_mse", None)
                if dino_metric is not None:
                    accumulator.update({"dino_grid_mse": dino_metric}, count=loss_scales[7])
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
        """Encode each complete trajectory once and backpropagate one combined loss."""
        lambda_wm = self.algorithm.wm_weight(self.state.global_step, self.total_steps)
        batch = self.batch_builder.prepare(batch_samples)
        if (getattr(self.config, "diagnose_outcome_gradients", False)
                and self.state.global_step == 0 and micro_step == 1):
            from nimloth.training.sft.stage3.diagnostics import outcome_gradient_diagnostic
            diagnostic = outcome_gradient_diagnostic(self.algorithm, self.model_runtime, batch, wm_weight=lambda_wm)
            diagnostic.update(rank=self.rank, epoch=epoch, micro_step=micro_step)
            path = self.checkpoint_runtime.manager.output_dir / f"outcome_gradients_rank_{self.rank:03d}.json"
            with path.open("x") as stream:
                json.dump(diagnostic, stream, indent=2, allow_nan=False)
            print(json.dumps(diagnostic, allow_nan=False), flush=True)
        timer_start = self.step_timer.start("forward")
        with saved_activation_context(self.config.activation_offload):
            primary = self.algorithm.training_primary_step(self.model_runtime, batch, wm_weight=lambda_wm)
            sigreg = None
            if self.algorithm.has_sigreg_stage:
                sigreg = self.algorithm.training_sigreg_step(
                    self.model_runtime, batch, online_states=primary.online_states,
                    sigreg_seed=global_sigreg_seed(self.config.seed, epoch, micro_step),
                )
            loss = primary.loss
            divisor = self.config.grad_accum
            if loss_scales is not None:
                lm = primary.losses.get("lm")
                lm_term = self.algorithm.ce_weight * lm if lm is not None else 0
                outcome = primary.losses.get("outcome")
                outcome_term = self.algorithm.outcome_weight * outcome if outcome is not None else 0
                dino = primary.losses.get("dino")
                dino_term = self.algorithm.dino_grid_weight * dino if dino is not None else 0
                loss = ((loss - lm_term - outcome_term - dino_term) * loss_scales[0]
                        + lm_term * loss_scales[1]
                        + outcome_term * loss_scales[2]
                        + dino_term * loss_scales[4])
                divisor = 1
            if sigreg is not None:
                loss = loss + sigreg.loss * (loss_scales[3] if loss_scales is not None else 1.)
        self.step_timer.stop("forward", timer_start)
        timer_start = self.step_timer.start("backward")
        self.optimization_runtime.backward(loss, grad_accum=divisor)
        self.step_timer.stop("backward", timer_start)
        metrics = self.algorithm.merge_training_metrics(primary.metrics, sigreg)
        return lambda_wm, metrics, primary.sample_count

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
        self._run_fixed_diagnostic()

    def _run_fixed_diagnostic(self) -> None:
        if self.state.global_step not in self.diagnostic_steps:
            return
        from nimloth.training.sft.stage3.fixed_diagnostics import run_fixed_diagnostic
        run_fixed_diagnostic(self)

    def _validate_and_checkpoint(self, epoch: int) -> None:
        """验证当前模型，并根据 WM MSE 更新 epoch/best checkpoint。"""

        val_metrics = self._evaluate_export(self.val_loader, epoch=epoch, split="eval")
        if self.outcome_eval_dir is not None:
            self._evaluate_export(self.train_loader, epoch=epoch, split="train")
        val_wm_mse = val_metrics.get("wm_mse", float("inf"))
        improved = val_wm_mse < self.state.best_val_wm_mse
        if improved:
            self.state.best_val_wm_mse = val_wm_mse
        if self.state.early_stop_state is not None:
            self.state.converged = update_early_stop(self.state.early_stop_state, val_metrics, epoch)
            val_metrics["early_stop_bad_epochs"] = self.state.early_stop_state["bad_epochs"]
            val_metrics["early_stop_converged"] = float(self.state.converged)
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
