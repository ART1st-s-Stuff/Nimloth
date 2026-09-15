"""SFT2 validation loop and distributed metric aggregation."""

from __future__ import annotations

import torch
import torch.distributed as dist

from nimloth.training.sft.stage3.algorithm import SFT2Algorithm
from nimloth.training.sft.stage3.batch import SFT2BatchBuilder
from nimloth.training.sft.stage3.runtime import SFT2ModelRuntime
from nimloth.training.sft.stage3.utils import preserve_module_modes
from nimloth.util.metrics import MetricAccumulator


def merge_metric_accumulators(
    accumulators: list[tuple[dict[str, float], dict[str, int]]],
) -> dict[str, float]:
    """Merge rank-local metric sums/counts into global averages."""

    merged = MetricAccumulator()
    for sums, counts in accumulators:
        for key, value in sums.items():
            merged.sums[key] = merged.sums.get(key, 0.0) + float(value)
        for key, value in counts.items():
            merged.counts[key] = merged.counts.get(key, 0) + int(value)
    return merged.averages()


def distributed_metric_averages(accumulator: MetricAccumulator) -> dict[str, float]:
    """Return global metric averages when distributed validation is active."""

    if not (dist.is_available() and dist.is_initialized()):
        return accumulator.averages()
    gathered: list[tuple[dict[str, float], dict[str, int]] | None] = [
        None
    ] * dist.get_world_size()
    dist.all_gather_object(gathered, (accumulator.sums, accumulator.counts))
    if any(rank_accumulator is None for rank_accumulator in gathered):
        raise RuntimeError("distributed validation failed to gather a rank accumulator")
    return merge_metric_accumulators(
        [rank_accumulator for rank_accumulator in gathered if rank_accumulator is not None]
    )


@torch.no_grad()
def evaluate(
    algorithm: SFT2Algorithm,
    runtime: SFT2ModelRuntime,
    loader,
    *,
    batch_builder: SFT2BatchBuilder,
    max_batches: int = -1,
    on_batch=None,
) -> dict[str, float]:
    """Evaluate with the same forward implementation used during training."""

    validation_runtime = runtime.unwrapped()
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(getattr(getattr(validation_runtime.agent, "backbone", None), "model", None), FSDP):
        local_batches = len(loader)
        effective_batches = min(local_batches, max_batches) if max_batches > 0 else local_batches
        counts = [None] * dist.get_world_size()
        dist.all_gather_object(counts, effective_batches)
        if len(set(counts)) != 1:
            raise ValueError("FSDP evaluation requires equal padded forward counts on every rank")
    accumulator = MetricAccumulator()
    with (
        preserve_module_modes(
            validation_runtime.agent.trainable_modules,
            training=False,
        ),
    ):
        for index, batch in enumerate(loader):
            if max_batches > 0 and index >= max_batches:
                break
            agent_batch = batch_builder.prepare(batch)
            output = algorithm.evaluation_step(validation_runtime, agent_batch)
            if output.sample_count > 0:
                if on_batch is not None:
                    on_batch(agent_batch, output)
                metrics = dict(output.metrics)
                dino = metrics.pop("dino_grid_mse", None)
                if dino is not None:
                    accumulator.update({"dino_grid_mse": dino},
                                       count=int(agent_batch.observed_state_weights.sum().item()))
                lm = metrics.pop("lm_ce", None)
                if lm is not None:
                    lm_count = int(agent_batch.lm_weights.sum().item())
                    if lm_count > 0:
                        accumulator.update({"lm_ce": lm}, count=lm_count)
                outcome = metrics.pop("outcome_bce", None)
                if outcome is not None:
                    accumulator.update({"outcome_bce": outcome}, count=int(agent_batch.outcome_mask.sum().item()))
                accumulator.update(metrics, count=output.sample_count)
    averages = distributed_metric_averages(accumulator)
    if "total_loss" in averages:
        # Components have different populations: successful windows for LM,
        # labeled transitions for outcome, unique observed states for DINO,
        # and all windows for WM/value and predictive DINO diagnostics.
        averages["total_loss"] = (
            averages.get("wm_mse", 0.0)
            + algorithm.value_weight * averages.get("value_total", 0.0)
            + algorithm.dino_grid_weight * averages.get("dino_grid_mse", 0.0)
            + algorithm.ce_weight * averages.get("lm_ce", 0.0)
            + algorithm.outcome_weight * averages.get("outcome_bce", 0.0)
        )
    return averages
