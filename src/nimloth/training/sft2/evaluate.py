"""SFT2 validation loop and distributed metric aggregation."""

from __future__ import annotations

import torch
import torch.distributed as dist

from nimloth.training.sft2.algorithm import SFT2Algorithm, SFT2Mode
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
    loader,
    *,
    max_batches: int = -1,
) -> dict[str, float]:
    """Evaluate with the same forward implementation used during training."""

    validation_algorithm = algorithm.unwrapped()
    accumulator = MetricAccumulator()
    with validation_algorithm.validation_context():
        for index, batch in enumerate(loader):
            if max_batches > 0 and index >= max_batches:
                break
            output = validation_algorithm.compute(batch, mode=SFT2Mode.VALIDATE)
            accumulator.update(output.metrics)
    return distributed_metric_averages(accumulator)
