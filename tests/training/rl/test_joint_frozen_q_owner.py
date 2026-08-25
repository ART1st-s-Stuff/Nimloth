from __future__ import annotations

import math

import pytest
import torch

from nimloth.training.rl.joint_critic import (
    JointActionValueCritic,
    create_frozen_critic_snapshot,
    export_frozen_critic_snapshot,
)
from nimloth.wm.grid import SharedSlotProjector
from nimloth.wm.value_head import ValueHead


_CONTRACT_ID = "sha256:joint-contract"


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


def _state(
    source_step: int,
    *,
    mutate: bool = False,
    dtype: torch.dtype = torch.float32,
):
    critic = _critic().to(dtype=dtype)
    if mutate:
        with torch.no_grad():
            next(critic.parameters()).add_(0.25)
    return export_frozen_critic_snapshot(
        create_frozen_critic_snapshot(
            critic,
            source_step=source_step,
            contract_id=_CONTRACT_ID,
            score_dtype="float32",
        )
    )


def _policy_state() -> dict[str, object]:
    return {
        "schema": "nimloth_policy_state_v2",
        "request_id": "episode-1",
        "generation_id": "generation-1",
        "latent_token_ids": [10, 11],
        "action_start_token_id": 12,
        "action_token_ids": [20, 21, 22],
        "latent_hidden": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "action_logits": [0.0, math.log(2.0), math.log(3.0)],
    }


def _score_request(pin: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "nimloth_frozen_q_owner_score_request_v1",
        "batch_pin": pin,
        "policy_state": _policy_state(),
        "expected_request_id": "episode-1",
        "expected_generation_id": "generation-1",
        "expected_latent_token_ids": [10, 11],
        "expected_action_start_token_id": 12,
        "expected_action_token_ids": [20, 21, 22],
        "expected_contract_id": _CONTRACT_ID,
    }


def _owner():
    from nimloth.training.rl.joint_frozen_q_owner import FrozenQSnapshotOwner

    return FrozenQSnapshotOwner(
        initial_snapshot_state=_state(776).to_mapping(),
        activation_version=4,
    )


def test_owner_pins_batch_and_scores_only_active_snapshot() -> None:
    from nimloth.training.rl.joint_frozen_q_owner import (
        FROZEN_Q_BATCH_PIN_SCHEMA,
        FROZEN_Q_OWNER_SCORE_RESULT_SCHEMA,
        FrozenQBatchPin,
        FrozenQOwnerScoringResult,
    )

    owner = _owner()
    status = owner.status()
    assert status["activation_version"] == 4
    assert status["active_source_step"] == 776
    assert status["open_batch_count"] == 0
    assert status["staged_snapshot_id"] is None

    pin_raw = owner.pin_batch(
        batch_id="batch-step-1",
        policy_step=1,
        expected_snapshot_id=status["active_snapshot_id"],
        expected_activation_version=4,
    )
    pin = FrozenQBatchPin.from_mapping(pin_raw)
    assert pin.schema == FROZEN_Q_BATCH_PIN_SCHEMA
    assert pin.batch_id == "batch-step-1"
    assert pin.policy_step == 1
    assert pin.snapshot_id == status["active_snapshot_id"]
    assert pin.activation_version == 4

    result = FrozenQOwnerScoringResult.from_mapping(
        owner.score(_score_request(pin.to_mapping()))
    )
    assert result.schema == FROZEN_Q_OWNER_SCORE_RESULT_SCHEMA
    assert result.batch_pin == pin
    assert result.scoring_record.snapshot_id == pin.snapshot_id
    assert result.scoring_record.snapshot_source_step == 776
    assert len(result.scoring_record.frozen_all_action_q) == 3

    forged = pin.to_mapping()
    forged["policy_step"] = 2
    with pytest.raises(ValueError, match="batch pin"):
        owner.score(_score_request(forged))
    with pytest.raises(ValueError, match="not pinned"):
        owner.score(_score_request({**pin.to_mapping(), "batch_id": "other"}))

    owner.unpin_batch(pin.to_mapping())
    with pytest.raises(ValueError, match="not pinned"):
        owner.score(_score_request(pin.to_mapping()))


def test_stage_and_activate_use_cas_and_never_change_open_batch() -> None:
    owner = _owner()
    initial = owner.status()
    pin = owner.pin_batch(
        batch_id="batch-step-1",
        policy_step=1,
        expected_snapshot_id=initial["active_snapshot_id"],
        expected_activation_version=4,
    )
    next_state = _state(777, mutate=True)
    staged = owner.stage_snapshot(
        new_snapshot_state=next_state.to_mapping(),
        expected_active_snapshot_id=initial["active_snapshot_id"],
        expected_activation_version=4,
    )
    assert staged["active_snapshot_id"] == initial["active_snapshot_id"]
    assert staged["staged_snapshot_id"] == next_state.snapshot_id
    with pytest.raises(ValueError, match="open batch"):
        owner.activate_staged(
            staged_snapshot_id=next_state.snapshot_id,
            expected_active_snapshot_id=initial["active_snapshot_id"],
            expected_activation_version=4,
        )

    owner.unpin_batch(pin)
    activated = owner.activate_staged(
        staged_snapshot_id=next_state.snapshot_id,
        expected_active_snapshot_id=initial["active_snapshot_id"],
        expected_activation_version=4,
    )
    assert activated["active_snapshot_id"] == next_state.snapshot_id
    assert activated["active_source_step"] == 777
    assert activated["activation_version"] == 5
    assert activated["staged_snapshot_id"] is None

    with pytest.raises(ValueError, match="activation version|active snapshot"):
        owner.stage_snapshot(
            new_snapshot_state=_state(778).to_mapping(),
            expected_active_snapshot_id=initial["active_snapshot_id"],
            expected_activation_version=4,
        )


def test_stage_rejects_stale_or_incompatible_snapshot() -> None:
    from nimloth.training.rl.joint_critic import create_frozen_critic_snapshot

    owner = _owner()
    status = owner.status()
    with pytest.raises(ValueError, match="newer"):
        owner.stage_snapshot(
            new_snapshot_state=_state(776, mutate=True).to_mapping(),
            expected_active_snapshot_id=status["active_snapshot_id"],
            expected_activation_version=4,
        )

    incompatible = export_frozen_critic_snapshot(
        create_frozen_critic_snapshot(
            _critic(),
            source_step=777,
            contract_id="sha256:other-contract",
            score_dtype="float32",
        )
    )
    with pytest.raises(ValueError, match="contract"):
        owner.stage_snapshot(
            new_snapshot_state=incompatible.to_mapping(),
            expected_active_snapshot_id=status["active_snapshot_id"],
            expected_activation_version=4,
        )

    wrong_parameter_dtype = _state(777, dtype=torch.float64)
    with pytest.raises(ValueError, match="parameter dtype"):
        owner.stage_snapshot(
            new_snapshot_state=wrong_parameter_dtype.to_mapping(),
            expected_active_snapshot_id=status["active_snapshot_id"],
            expected_activation_version=4,
        )


def test_checkpoint_state_requires_no_pin_or_staged_candidate() -> None:
    from nimloth.training.rl.joint_frozen_q_owner import (
        FROZEN_Q_OWNER_CHECKPOINT_SCHEMA,
        FrozenQSnapshotOwner,
    )

    owner = _owner()
    checkpoint = owner.checkpoint_state()
    assert checkpoint["schema"] == FROZEN_Q_OWNER_CHECKPOINT_SCHEMA
    restored = FrozenQSnapshotOwner.from_checkpoint_state(checkpoint)
    assert restored.status() == owner.status()

    status = owner.status()
    pin = owner.pin_batch(
        batch_id="batch-step-1",
        policy_step=1,
        expected_snapshot_id=status["active_snapshot_id"],
        expected_activation_version=4,
    )
    with pytest.raises(ValueError, match="open batch"):
        owner.checkpoint_state()
    owner.unpin_batch(pin)
    owner.stage_snapshot(
        new_snapshot_state=_state(777, mutate=True).to_mapping(),
        expected_active_snapshot_id=status["active_snapshot_id"],
        expected_activation_version=4,
    )
    with pytest.raises(ValueError, match="staged"):
        owner.checkpoint_state()


def test_owner_requests_and_pins_reject_extra_missing_and_coercion() -> None:
    from nimloth.training.rl.joint_frozen_q_owner import FrozenQBatchPin

    owner = _owner()
    status = owner.status()
    with pytest.raises(ValueError, match="policy_step"):
        owner.pin_batch(
            batch_id="batch",
            policy_step=True,
            expected_snapshot_id=status["active_snapshot_id"],
            expected_activation_version=4,
        )
    pin = owner.pin_batch(
        batch_id="batch",
        policy_step=1,
        expected_snapshot_id=status["active_snapshot_id"],
        expected_activation_version=4,
    )
    missing = dict(pin)
    missing.pop("snapshot_id")
    with pytest.raises(ValueError, match="missing fields"):
        FrozenQBatchPin.from_mapping(missing)
    request = _score_request(pin)
    request["current_q"] = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="unexpected fields"):
        owner.score(request)
