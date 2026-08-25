from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from nimloth.training.rl.joint_critic import (
    JointActionValueCritic,
    JointCriticSpec,
    create_frozen_critic_snapshot,
)
from nimloth.wm.grid import SharedSlotProjector
from nimloth.wm.value_head import ValueHead


def _critic() -> JointActionValueCritic:
    return JointActionValueCritic(
        state_projector=SharedSlotProjector(
            input_dim=3,
            output_dim=2,
            hidden_dim=5,
            grid_tokens=2,
        ),
        value_head=ValueHead(emb_dim=2, num_actions=3, hidden_dim=4),
    )


def _snapshot(*, source_step: int = 776):
    return create_frozen_critic_snapshot(
        _critic(),
        source_step=source_step,
        contract_id="sha256:joint-contract",
        score_dtype="float32",
    )


def test_snapshot_state_export_restore_is_cpu_exact_and_independent() -> None:
    from nimloth.training.rl.joint_critic import (
        FROZEN_CRITIC_SNAPSHOT_STATE_SCHEMA,
        FrozenJointCriticSnapshotState,
        export_frozen_critic_snapshot,
        restore_frozen_critic_snapshot,
    )

    torch.manual_seed(17)
    snapshot = _snapshot()
    hidden = torch.randn(4, 2, 3)
    expected = snapshot(hidden)

    state = export_frozen_critic_snapshot(snapshot)
    assert state.schema == FROZEN_CRITIC_SNAPSHOT_STATE_SCHEMA
    assert state.snapshot_id == snapshot.snapshot_id
    assert state.source_step == 776
    assert all(tensor.device.type == "cpu" for _, tensor in state.critic_state)
    assert all(not tensor.requires_grad for _, tensor in state.critic_state)
    assert FrozenJointCriticSnapshotState.from_mapping(state.to_mapping()) == state

    with torch.no_grad():
        next(snapshot.critic.parameters()).add_(1.0)
    restored = restore_frozen_critic_snapshot(state)
    assert restored.snapshot_id == state.snapshot_id
    assert restored.source_step == 776
    assert restored.training is False
    assert all(not parameter.requires_grad for parameter in restored.parameters())
    torch.testing.assert_close(restored(hidden), expected, rtol=0, atol=0)


def test_snapshot_state_mapping_clones_tensors_and_revalidates_fingerprint() -> None:
    from nimloth.training.rl.joint_critic import (
        FrozenJointCriticSnapshotState,
        export_frozen_critic_snapshot,
        restore_frozen_critic_snapshot,
    )

    state = export_frozen_critic_snapshot(_snapshot())
    raw = state.to_mapping()
    round_trip = FrozenJointCriticSnapshotState.from_mapping(raw)
    first_name = next(iter(raw["critic_state"]))
    raw["critic_state"][first_name].add_(1.0)
    assert not torch.equal(
        raw["critic_state"][first_name],
        dict(round_trip.critic_state)[first_name],
    )

    forged = state.to_mapping()
    next(iter(forged["critic_state"].values())).add_(1.0)
    with pytest.raises(ValueError, match="fingerprint|snapshot_id"):
        FrozenJointCriticSnapshotState.from_mapping(forged)
    with pytest.raises(ValueError, match="fingerprint|snapshot_id"):
        restore_frozen_critic_snapshot(forged)


def test_snapshot_state_rejects_non_cpu_nonfinite_and_schema_forgery() -> None:
    from nimloth.training.rl.joint_critic import (
        FrozenJointCriticSnapshotState,
        export_frozen_critic_snapshot,
    )

    state = export_frozen_critic_snapshot(_snapshot())
    with pytest.raises(ValueError, match="schema"):
        replace(state, schema="future")
    with pytest.raises(ValueError, match="source_step"):
        replace(state, source_step=-1)
    with pytest.raises(ValueError, match="score_dtype"):
        replace(state, score_dtype="float16")

    nonfinite = state.to_mapping()
    next(iter(nonfinite["critic_state"].values())).flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        FrozenJointCriticSnapshotState.from_mapping(nonfinite)

    if torch.cuda.is_available():
        cuda_state = state.to_mapping()
        cuda_state["critic_state"] = {
            name: tensor.cuda() for name, tensor in cuda_state["critic_state"].items()
        }
        with pytest.raises(ValueError, match="CPU"):
            FrozenJointCriticSnapshotState.from_mapping(cuda_state)


def test_snapshot_state_rejects_untrusted_oversized_spec_before_allocation() -> None:
    from nimloth.training.rl.joint_critic import FrozenJointCriticSnapshotState

    state = _snapshot()
    from nimloth.training.rl.joint_critic import export_frozen_critic_snapshot

    raw = export_frozen_critic_snapshot(state).to_mapping()
    raw["critic_spec"]["qwen_hidden_dim"] = 1_000_001
    with pytest.raises(ValueError, match="safety bound"):
        FrozenJointCriticSnapshotState.from_mapping(raw)

    with pytest.raises(ValueError, match="safety bound"):
        JointCriticSpec(
            qwen_hidden_dim=1_000_000,
            projector_hidden_dim=1_000_000,
            state_dim=1_000_000,
            grid_tokens=16,
            value_hidden_dim=1_000_000,
            action_count=8,
        )


def test_malformed_state_is_rejected_before_any_tensor_clone(monkeypatch) -> None:
    from nimloth.training.rl.joint_critic import (
        FrozenJointCriticSnapshotState,
        export_frozen_critic_snapshot,
    )

    state = export_frozen_critic_snapshot(_snapshot())
    clone_calls = 0
    original_clone = torch.Tensor.clone

    def tracked_clone(tensor, *args, **kwargs):
        nonlocal clone_calls
        clone_calls += 1
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", tracked_clone)
    missing = state.to_mapping()
    clone_calls = 0
    missing["critic_state"].pop(next(iter(missing["critic_state"])))
    with pytest.raises(ValueError, match="state keys"):
        FrozenJointCriticSnapshotState.from_mapping(missing)
    assert clone_calls == 0

    wrong_shape = state.to_mapping()
    matrix_name = next(
        name
        for name, tensor in wrong_shape["critic_state"].items()
        if tensor.ndim == 2
    )
    wrong_shape["critic_state"][matrix_name] = wrong_shape["critic_state"][
        matrix_name
    ].reshape(-1)
    clone_calls = 0
    with pytest.raises(ValueError, match="tensor shape"):
        FrozenJointCriticSnapshotState.from_mapping(wrong_shape)
    assert clone_calls == 0


def test_snapshot_state_strict_fields_spec_and_module_keys() -> None:
    from nimloth.training.rl.joint_critic import (
        FrozenJointCriticSnapshotState,
        export_frozen_critic_snapshot,
        restore_frozen_critic_snapshot,
    )

    state = export_frozen_critic_snapshot(_snapshot())
    missing = state.to_mapping()
    missing.pop("score_dtype")
    with pytest.raises(ValueError, match="missing fields"):
        FrozenJointCriticSnapshotState.from_mapping(missing)
    unexpected = state.to_mapping()
    unexpected["checkpoint_path"] = "/forbidden"
    with pytest.raises(ValueError, match="unexpected fields"):
        FrozenJointCriticSnapshotState.from_mapping(unexpected)

    bad_spec = state.to_mapping()
    bad_spec["critic_spec"]["action_count"] = 4
    with pytest.raises(ValueError, match="tensor shape|fingerprint|snapshot_id"):
        FrozenJointCriticSnapshotState.from_mapping(bad_spec)

    missing_key = deepcopy(state.to_mapping())
    missing_key["critic_state"].pop(next(iter(missing_key["critic_state"])))
    with pytest.raises(ValueError, match="fingerprint|snapshot_id|state keys"):
        restore_frozen_critic_snapshot(missing_key)

    wrong_shape = state.to_mapping()
    first_name = next(
        name
        for name, tensor in wrong_shape["critic_state"].items()
        if tensor.ndim == 2
    )
    wrong_shape["critic_state"][first_name] = wrong_shape["critic_state"][
        first_name
    ].reshape(-1)
    with pytest.raises(ValueError, match="tensor shape"):
        FrozenJointCriticSnapshotState.from_mapping(wrong_shape)
