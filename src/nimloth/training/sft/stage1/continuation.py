"""Explicit epoch continuation with a new schedule, preserving training state."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from .convergence import ConvergenceState


def _without_query_token_lr(identity):
    """Return an identity with only the reviewed Query-LR field removed."""
    normalized = deepcopy(identity)
    token_rows = normalized.get("token_row_training")
    if not isinstance(token_rows, dict) or "query_token_lr" not in token_rows:
        raise ValueError(
            "query LR continuation requires a saved query-token optimizer identity"
        )
    query_lr = token_rows.pop("query_token_lr")
    return normalized, query_lr


def validate_epoch_continuation(
    path,
    state,
    identity,
    *,
    world,
    allow_dino_weight_change=False,
    allow_projector_lr_change=False,
    allow_query_token_lr_change=False,
):
    if sum(
        bool(value)
        for value in (
            allow_dino_weight_change,
            allow_projector_lr_change,
            allow_query_token_lr_change,
        )
    ) > 1:
        raise ValueError("change only one continuation identity field at a time")
    path = Path(path)
    marker = json.loads((path / "COMMITTED").read_text())
    epoch = state.get("epoch")
    if not isinstance(epoch, int) or epoch < 1 or path.name != f"epoch_{epoch:03d}":
        raise ValueError("continuation requires a completed epoch checkpoint")
    if marker != {"epoch": epoch, "step": state.get("step")} or "resume_schema" in state:
        raise ValueError("continuation COMMITTED boundary mismatch")
    allowed = {"epochs", "convergence", "warmup_ratio"}
    if allow_dino_weight_change:
        allowed = {"weight_dino"}
    elif allow_projector_lr_change:
        allowed.add("projector_lr")
    previous = state.get("identity")
    if allow_query_token_lr_change:
        if not isinstance(previous, dict):
            raise ValueError("continuation stage/dataset/optimizer identity mismatch")
        previous, previous_query_lr = _without_query_token_lr(previous)
        identity, query_lr = _without_query_token_lr(identity)
        if previous_query_lr == query_lr:
            raise ValueError("query LR continuation requires query_token_lr to change")
        # Unlike ordinary schedule continuation, this explicit gate permits no
        # top-level policy or schedule changes alongside the nested Query LR.
        allowed = set()
    if not isinstance(previous, dict) or (
        {k: v for k, v in previous.items() if k not in allowed}
        != {k: v for k, v in identity.items() if k not in allowed}
    ):
        raise ValueError("continuation stage/dataset/optimizer identity mismatch")
    rng = state.get("rank_rng_states")
    if state.get("world_size") != world or not isinstance(rng, list) or len(rng) != world:
        raise ValueError("continuation requires matching world size and per-rank RNG")
    if any(not isinstance(item, dict) or not {"python", "numpy", "torch_cpu"} <= item.keys() for item in rng):
        raise ValueError("continuation has incomplete per-rank RNG")
    if world > 1 and any("torch_cuda" not in item for item in rng):
        raise ValueError("distributed continuation requires per-rank CUDA RNG")
    if not state.get("optimizer") or not state.get("scheduler"):
        raise ValueError("continuation requires complete optimizer and scheduler state")
    return epoch


def restart_schedule(optimizer, learning_rates, *, steps_per_epoch, remaining_epochs,
                     warmup_ratio, until_converged):
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
    if len(optimizer.param_groups) != len(learning_rates):
        raise ValueError("continuation optimizer group count mismatch")
    for group, lr in zip(optimizer.param_groups, learning_rates):
        group["lr"] = group["initial_lr"] = lr
    if until_converged:
        import math
        return get_constant_schedule_with_warmup(optimizer, math.ceil(steps_per_epoch * warmup_ratio))
    if remaining_epochs <= 0:
        raise ValueError("--epochs must exceed the completed source epoch")
    steps = steps_per_epoch * remaining_epochs
    return get_cosine_schedule_with_warmup(optimizer, int(steps * warmup_ratio), steps)


def replay_convergence(path, epoch, policy, monitor):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["epoch"] <= epoch]
    if [row["epoch"] for row in rows] != list(range(1, epoch + 1)):
        raise ValueError("continuation validation history is incomplete or duplicated")
    result = ConvergenceState()
    for row in rows:
        if row.get("monitor") != monitor:
            raise ValueError("continuation validation monitor mismatch")
        result.observe(epoch=row["epoch"], loss=row[monitor], policy=policy)
    return result, rows


def continuation_provenance(path, identity):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with (path / "training_state.pt").open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"parent_checkpoint": str(path), "training_state_sha256": digest.hexdigest(),
            "new_schedule_identity": identity}

def changed_objective_baseline(path, epoch, weight_lm, weight_dino):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["epoch"] == epoch]
    if len(rows) != 1:
        raise ValueError("objective continuation requires exactly one source epoch validation")
    row = dict(rows[0])
    row["source_validation_total_loss"] = row["validation_total_loss"]
    loss = weight_lm * row["validation_lm_loss"] + weight_dino * row["validation_dino_loss"]
    if not __import__("math").isfinite(loss):
        raise ValueError("nonfinite objective continuation baseline")
    row.update(validation_total_loss=loss, weight_lm=weight_lm, weight_dino=weight_dino,
               continuation_baseline=True)
    return ConvergenceState(best_loss=loss, previous_loss=loss, last_epoch=epoch), [row]
