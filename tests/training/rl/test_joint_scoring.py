from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from nimloth.training.rl.joint_critic import create_frozen_critic_snapshot
from nimloth.training.rl.joint_scoring import (
    FrozenQScoringRecord,
    score_captured_policy_state,
)
from nimloth.wm.grid import SharedSlotProjector
from nimloth.wm.value_head import ValueHead


def _critic():
    from nimloth.training.rl.joint_critic import JointActionValueCritic

    return JointActionValueCritic(
        state_projector=SharedSlotProjector(
            input_dim=3,
            output_dim=2,
            hidden_dim=5,
            grid_tokens=2,
        ),
        value_head=ValueHead(emb_dim=2, num_actions=3, hidden_dim=4),
    )


def _payload() -> dict[str, object]:
    return {
        "schema": "nimloth_policy_state_v2",
        "request_id": "request-17",
        "generation_id": "generation-23",
        "latent_token_ids": [90, 91],
        "action_start_token_id": 92,
        "action_token_ids": [100, 101, 102],
        "latent_hidden": [[0.25, -0.5, 1.0], [1.5, 0.0, -1.0]],
        "action_logits": [0.2, -0.3, 0.7],
    }


def _snapshot(*, score_dtype: str = "float32"):
    return create_frozen_critic_snapshot(
        _critic(),
        source_step=11,
        contract_id="sha256:joint-contract",
        score_dtype=score_dtype,
    )


def _score(
    payload: dict[str, object] | None = None,
    *,
    snapshot=None,
) -> FrozenQScoringRecord:
    return score_captured_policy_state(
        _payload() if payload is None else payload,
        snapshot=_snapshot() if snapshot is None else snapshot,
        expected_request_id="request-17",
        expected_generation_id="generation-23",
        expected_latent_token_ids=(90, 91),
        expected_action_start_token_id=92,
        expected_action_token_ids=(100, 101, 102),
        expected_contract_id="sha256:joint-contract",
    )


def test_scores_identity_bound_capture_and_preserves_raw_prior_logits() -> None:
    snapshot = _snapshot()
    record = _score(snapshot=snapshot)
    assert isinstance(record, FrozenQScoringRecord)
    assert record.schema == "nimloth_frozen_q_scoring_v1"
    assert record.request_id == "request-17"
    assert record.generation_id == "generation-23"
    assert record.snapshot_id == snapshot.snapshot_id
    assert record.snapshot_source_step == 11
    assert record.contract_id == "sha256:joint-contract"
    assert record.latent_token_ids == (90, 91)
    assert record.action_start_token_id == 92
    assert record.action_token_ids == (100, 101, 102)
    assert record.score_dtype == "float32"
    assert record.prior_logits == tuple(
        torch.tensor([0.2, -0.3, 0.7], dtype=torch.float32).tolist()
    )
    assert len(record.frozen_all_action_q) == 3
    assert all(torch.isfinite(torch.tensor(record.frozen_all_action_q)))
    assert record.record_id().startswith("sha256:")
    assert FrozenQScoringRecord.from_mapping(record.to_mapping()) == record


@pytest.mark.parametrize(
    ("parameter_dtype", "score_dtype", "expected_score_dtype"),
    [
        (torch.float64, "float64", torch.float64),
        (torch.float32, "bfloat16", torch.bfloat16),
    ],
)
def test_scoring_quantizes_prior_and_q_to_contract_bound_dtype(
    parameter_dtype: torch.dtype,
    score_dtype: str,
    expected_score_dtype: torch.dtype,
) -> None:
    critic = _critic().to(dtype=parameter_dtype)
    snapshot = create_frozen_critic_snapshot(
        critic,
        source_step=11,
        contract_id="sha256:joint-contract",
        score_dtype=score_dtype,
    )
    expected_q = snapshot(
        torch.tensor(
            _payload()["latent_hidden"],
            dtype=parameter_dtype,
        ).unsqueeze(0)
    )[0].to(dtype=expected_score_dtype)
    record = _score(snapshot=snapshot)
    assert record.score_dtype == score_dtype
    assert record.prior_logits == tuple(
        torch.tensor(
            _payload()["action_logits"],
            dtype=expected_score_dtype,
        ).tolist()
    )
    assert record.frozen_all_action_q == tuple(expected_q.tolist())
    assert all(parameter.grad is None for parameter in snapshot.parameters())


def test_rejects_schema_identity_and_token_table_mismatch() -> None:
    cases = (
        ("schema", "wrong", "schema"),
        ("request_id", "other-request", "request"),
        ("generation_id", "other-generation", "generation"),
        ("latent_token_ids", [91, 90], "latent token"),
        ("action_start_token_id", 93, "action-start"),
        ("action_token_ids", [101, 100, 102], "action token"),
    )
    for field, value, message in cases:
        payload = _payload()
        payload[field] = value
        with pytest.raises(ValueError, match=message):
            _score(payload)


def test_scorer_rejects_collapsed_expected_identity() -> None:
    with pytest.raises(ValueError, match="differ"):
        score_captured_policy_state(
            {**_payload(), "generation_id": "request-17"},
            snapshot=_snapshot(),
            expected_request_id="request-17",
            expected_generation_id="request-17",
            expected_latent_token_ids=(90, 91),
            expected_action_start_token_id=92,
            expected_action_token_ids=(100, 101, 102),
            expected_contract_id="sha256:joint-contract",
        )


def test_rejects_hidden_and_logit_shape_type_and_finite_errors() -> None:
    cases = (
        ("latent_hidden", [[0.0, 1.0, 2.0]], "latent hidden shape"),
        ("latent_hidden", [[0.0, 1.0], [2.0, 3.0]], "latent hidden shape"),
        (
            "latent_hidden",
            [[0.0, float("nan"), 2.0], [2.0, 3.0, 4.0]],
            "latent hidden",
        ),
        ("latent_hidden", [[0.0, True, 2.0], [2.0, 3.0, 4.0]], "latent hidden"),
        ("action_logits", [0.0, 1.0], "action logits shape"),
        ("action_logits", [0.0, float("inf"), 2.0], "action logits"),
        ("action_logits", [0.0, True, 2.0], "action logits"),
    )
    for field, value, message in cases:
        payload = _payload()
        payload[field] = value
        with pytest.raises(ValueError, match=message):
            _score(payload)


def test_rejects_contract_snapshot_action_count_and_dtype_mismatch() -> None:
    snapshot = _snapshot()
    with pytest.raises(ValueError, match="contract"):
        score_captured_policy_state(
            _payload(),
            snapshot=snapshot,
            expected_request_id="request-17",
            expected_generation_id="generation-23",
            expected_latent_token_ids=(90, 91),
            expected_action_start_token_id=92,
            expected_action_token_ids=(100, 101, 102),
            expected_contract_id="sha256:other-contract",
        )
    with pytest.raises(ValueError, match="action count"):
        score_captured_policy_state(
            {**_payload(), "action_token_ids": [100, 101]},
            snapshot=snapshot,
            expected_request_id="request-17",
            expected_generation_id="generation-23",
            expected_latent_token_ids=(90, 91),
            expected_action_start_token_id=92,
            expected_action_token_ids=(100, 101),
            expected_contract_id="sha256:joint-contract",
        )


def test_record_construction_and_mapping_always_revalidate() -> None:
    record = _score()
    with pytest.raises(ValueError, match="snapshot"):
        replace(record, snapshot_id="")
    with pytest.raises(ValueError, match="generation"):
        replace(record, generation_id="")
    with pytest.raises(ValueError, match="differ"):
        replace(record, generation_id=record.request_id)
    with pytest.raises(ValueError, match="align"):
        replace(record, frozen_all_action_q=(0.0,))
    with pytest.raises(ValueError, match="finite"):
        replace(record, prior_logits=(0.0, float("nan"), 1.0))

    bad = record.to_mapping()
    bad["frozen_all_action_q"][0] = float("nan")
    with pytest.raises(ValueError, match="frozen Q"):
        FrozenQScoringRecord.from_mapping(bad)


def test_all_record_construction_paths_quantize_declared_bfloat16_dtype() -> None:
    record = _score(snapshot=_snapshot(score_dtype="bfloat16"))
    unquantized = replace(
        record,
        prior_logits=(0.2, -0.3, 0.7),
        frozen_all_action_q=(0.1, -0.2, 0.3),
    )
    expected_prior = tuple(
        torch.tensor([0.2, -0.3, 0.7], dtype=torch.bfloat16).tolist()
    )
    expected_q = tuple(
        torch.tensor([0.1, -0.2, 0.3], dtype=torch.bfloat16).tolist()
    )
    assert unquantized.prior_logits == expected_prior
    assert unquantized.frozen_all_action_q == expected_q

    built = FrozenQScoringRecord.build(
        request_id=record.request_id,
        generation_id=record.generation_id,
        contract_id=record.contract_id,
        snapshot_id=record.snapshot_id,
        snapshot_source_step=record.snapshot_source_step,
        latent_token_ids=record.latent_token_ids,
        action_start_token_id=record.action_start_token_id,
        action_token_ids=record.action_token_ids,
        score_dtype="bfloat16",
        prior_logits=(0.2, -0.3, 0.7),
        frozen_all_action_q=(0.1, -0.2, 0.3),
    )
    assert built.prior_logits == expected_prior
    assert built.frozen_all_action_q == expected_q

    mapping = record.to_mapping()
    mapping["prior_logits"] = [0.2, -0.3, 0.7]
    mapping["frozen_all_action_q"] = [0.1, -0.2, 0.3]
    restored = FrozenQScoringRecord.from_mapping(mapping)
    assert restored.prior_logits == expected_prior
    assert restored.frozen_all_action_q == expected_q
