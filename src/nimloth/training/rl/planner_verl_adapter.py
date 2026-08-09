"""Strict PlannerPolicyHead decision batches transported through VERL DataProto."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nimloth.training.rl.episodes import ExecutedTransition


PLANNER_VERL_SCHEMA_VERSION = 2
PINNED_VERL_COMMIT = "65316156d1011d71d62e0542e4b954f9499e872e"
PLANNER_POLICY_OBJECTIVE = "receding_horizon_planner_policy_ppo_v1"


@dataclass(frozen=True)
class VerlSource:
    root: Path
    commit: str


@dataclass(frozen=True)
class PlannerVERLUpdateInputs:
    transitions: tuple[ExecutedTransition, ...]
    return_targets: tuple[torch.Tensor, ...]
    old_action_values: tuple[torch.Tensor, ...]
    old_policy_log_probs: tuple[torch.Tensor, ...]
    policy_advantages: tuple[torch.Tensor, ...]
    loss_weights: tuple[float, ...]
    token_counts: tuple[int, ...]
    dino_grid_targets: tuple[torch.Tensor | None, ...]
    total_transitions: int


@lru_cache(maxsize=4)
def assert_pinned_verl_source(expected_root: Path) -> VerlSource:
    """Fail closed when Python imports VERL from any source except the gitlink."""

    import verl

    expected = expected_root.resolve()
    module_path = Path(verl.__file__).resolve()
    actual = module_path.parents[1]
    if actual != expected:
        raise RuntimeError(
            "VERL import does not resolve to the pinned runtime submodule: "
            f"expected={expected}, actual={actual}"
        )
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(expected), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot verify VERL source at {expected}") from error
    if commit != PINNED_VERL_COMMIT:
        raise RuntimeError(
            "VERL runtime commit differs from the pinned VAGEN gitlink: "
            f"expected={PINNED_VERL_COMMIT}, actual={commit}"
        )
    return VerlSource(root=actual, commit=commit)


def _runtime_verl_root() -> Path:
    return Path(__file__).resolve().parents[4] / "external/VAGEN/verl"


def _finite_scalar_rows(
    values: tuple[torch.Tensor, ...],
    *,
    name: str,
    batch_size: int,
) -> torch.Tensor:
    if len(values) != batch_size:
        raise ValueError(f"{name} must have {batch_size} rows, got {len(values)}")
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise ValueError(f"{name} must contain tensor rows")
    rows = torch.stack(tuple(value.detach().reshape(()) for value in values)).cpu()
    if not torch.isfinite(rows).all():
        raise ValueError(f"{name} must be finite")
    return rows


def build_planner_update_dataproto(
    *,
    transitions: tuple[ExecutedTransition, ...],
    return_targets: tuple[torch.Tensor, ...],
    old_action_values: tuple[torch.Tensor, ...],
    old_policy_log_probs: tuple[torch.Tensor, ...],
    policy_advantages: tuple[torch.Tensor, ...],
    loss_weights: tuple[float, ...],
    token_counts: tuple[int, ...],
    total_transitions: int,
    dino_grid_targets: tuple[torch.Tensor | None, ...] | None = None,
) -> Any:
    """Build one strict action-level update batch without inventing CoT/state."""

    assert_pinned_verl_source(_runtime_verl_root())
    from verl import DataProto

    if not transitions:
        raise ValueError("planner VERL batch must not be empty")
    batch_size = len(transitions)
    if total_transitions < 1:
        raise ValueError("total_transitions must be positive")
    tensors = {
        "return_targets": _finite_scalar_rows(
            return_targets,
            name="return_targets",
            batch_size=batch_size,
        ),
        "old_action_values": _finite_scalar_rows(
            old_action_values,
            name="old_action_values",
            batch_size=batch_size,
        ),
        "old_policy_log_probs": _finite_scalar_rows(
            old_policy_log_probs,
            name="old_policy_log_probs",
            batch_size=batch_size,
        ),
        "policy_advantages": _finite_scalar_rows(
            policy_advantages,
            name="policy_advantages",
            batch_size=batch_size,
        ),
    }
    if len(loss_weights) != batch_size:
        raise ValueError(
            f"loss_weights must have {batch_size} rows, got {len(loss_weights)}"
        )
    loss_weight_tensor = torch.tensor(loss_weights, dtype=torch.float32)
    if not torch.isfinite(loss_weight_tensor).all() or (loss_weight_tensor < 0).any():
        raise ValueError("loss_weights must be finite and non-negative")
    tensors["loss_weights"] = loss_weight_tensor
    if len(token_counts) != batch_size:
        raise ValueError(
            f"token_counts must have {batch_size} rows, got {len(token_counts)}"
        )
    if any(isinstance(value, bool) or int(value) < 1 for value in token_counts):
        raise ValueError("token_counts must be positive integers")
    tensors["token_counts"] = torch.tensor(token_counts, dtype=torch.long)

    dino_rows = (
        dino_grid_targets
        if dino_grid_targets is not None
        else (None,) * batch_size
    )
    if len(dino_rows) != batch_size:
        raise ValueError(
            f"dino_grid_targets must have {batch_size} rows, got {len(dino_rows)}"
        )
    has_dino = tuple(value is not None for value in dino_rows)
    if any(has_dino) and not all(has_dino):
        raise ValueError("DINO-grid targets must cover all rows or no rows")
    if all(has_dino):
        tensors["dino_grid_targets"] = torch.stack(
            tuple(value.detach().cpu() for value in dino_rows if value is not None)
        )
        if not torch.isfinite(tensors["dino_grid_targets"]).all():
            raise ValueError("dino_grid_targets must be finite")

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "transitions": np.asarray(transitions, dtype=object),
        },
        meta_info={
            "schema_version": PLANNER_VERL_SCHEMA_VERSION,
            "objective": PLANNER_POLICY_OBJECTIVE,
            "total_transitions": int(total_transitions),
            "has_dino_grid_targets": bool(all(has_dino)),
        },
    )


def planner_verl_micro_batches(
    data: Any,
    *,
    max_padded_tokens: int,
    max_rows: int,
) -> tuple[Any, ...]:
    """First-fit-decreasing packs multimodal rows by real padded cost.

    Pinned VERL's dynamic batch path is skipped for ``multi_modal_inputs``.
    Planner Qwen therefore budgets ``max(sequence_length) * rows`` directly
    instead of assuming remove-padding is active.
    """

    if max_padded_tokens < 1:
        raise ValueError("max_padded_tokens must be positive")
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    inputs = planner_update_inputs(data)
    over_budget = [
        (index, count)
        for index, count in enumerate(inputs.token_counts)
        if count > max_padded_tokens
    ]
    if over_budget:
        raise ValueError(
            "planner prefix exceeds max_padded_tokens: "
            + ", ".join(f"row{index}={count}" for index, count in over_budget)
        )
    groups: list[list[int]] = []
    for index in sorted(
        range(len(inputs.token_counts)),
        key=lambda item: (-inputs.token_counts[item], item),
    ):
        token_count = inputs.token_counts[index]
        for group in groups:
            candidate = (*group, index)
            padded_tokens = max(
                inputs.token_counts[item] for item in candidate
            ) * len(candidate)
            if len(candidate) <= max_rows and padded_tokens <= max_padded_tokens:
                group.append(index)
                break
        else:
            groups.append([index])

    from verl import DataProto

    batches: list[Any] = []
    for group in groups:
        indices = torch.tensor(group, dtype=torch.long)
        batches.append(
            DataProto(
                batch=data.batch[indices],
                non_tensor_batch={
                    name: values[np.asarray(group, dtype=np.int64)]
                    for name, values in data.non_tensor_batch.items()
                },
                meta_info=dict(data.meta_info),
            )
        )
    return tuple(batches)


def planner_update_inputs(data: Any) -> PlannerVERLUpdateInputs:
    """Validate and unpack a worker-local DataProto chunk."""

    expected_meta = {
        "schema_version": PLANNER_VERL_SCHEMA_VERSION,
        "objective": PLANNER_POLICY_OBJECTIVE,
    }
    for name, expected in expected_meta.items():
        if data.meta_info.get(name) != expected:
            raise ValueError(
                f"planner VERL {name} mismatch: "
                f"{data.meta_info.get(name)!r} != {expected!r}"
            )
    transitions = tuple(data.non_tensor_batch.get("transitions", ()))
    batch_size = len(data)
    if len(transitions) != batch_size or batch_size < 1:
        raise ValueError("planner VERL transitions do not align with tensor rows")
    required = {
        "return_targets",
        "old_action_values",
        "old_policy_log_probs",
        "policy_advantages",
        "loss_weights",
        "token_counts",
    }
    missing = sorted(required - set(data.batch.keys()))
    if missing:
        raise ValueError("planner VERL batch is missing tensors: " + ", ".join(missing))
    has_dino = bool(data.meta_info.get("has_dino_grid_targets", False))
    if has_dino != ("dino_grid_targets" in data.batch):
        raise ValueError("planner VERL DINO metadata does not match tensors")
    total_transitions = int(data.meta_info.get("total_transitions", 0))
    if total_transitions < 1:
        raise ValueError("planner VERL total_transitions must be positive")

    def rows(name: str) -> tuple[torch.Tensor, ...]:
        tensor = data.batch[name]
        if tensor.shape[0] != batch_size or not torch.isfinite(tensor).all():
            raise ValueError(f"planner VERL {name} rows are invalid")
        return tuple(tensor.unbind(0))

    token_counts = tuple(int(value) for value in data.batch["token_counts"].tolist())
    if any(value < 1 for value in token_counts):
        raise ValueError("planner VERL token counts must be positive")
    loss_weights = tuple(
        float(value) for value in data.batch["loss_weights"].tolist()
    )
    if any(value < 0.0 for value in loss_weights):
        raise ValueError("planner VERL loss weights must be non-negative")
    dino_rows: tuple[torch.Tensor | None, ...] = (
        tuple(data.batch["dino_grid_targets"].unbind(0))
        if has_dino
        else (None,) * batch_size
    )
    return PlannerVERLUpdateInputs(
        transitions=transitions,
        return_targets=rows("return_targets"),
        old_action_values=rows("old_action_values"),
        old_policy_log_probs=rows("old_policy_log_probs"),
        policy_advantages=rows("policy_advantages"),
        loss_weights=loss_weights,
        token_counts=token_counts,
        dino_grid_targets=dino_rows,
        total_transitions=total_transitions,
    )


__all__ = [
    "PINNED_VERL_COMMIT",
    "PLANNER_VERL_SCHEMA_VERSION",
    "PlannerVERLUpdateInputs",
    "VerlSource",
    "assert_pinned_verl_source",
    "build_planner_update_dataproto",
    "planner_update_inputs",
    "planner_verl_micro_batches",
]
