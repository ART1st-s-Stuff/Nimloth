from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nimloth.training.rl.joint_critic import (
    JointActionValueCritic,
    create_frozen_critic_snapshot,
    load_joint_action_value_critic,
)
from nimloth.training.sft2.algorithm import SFT2_VALUE_OBJECTIVE
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


def _write_checkpoint(root: Path, *, value_objective: str = SFT2_VALUE_OBJECTIVE) -> None:
    critic = _critic()
    torch.save(critic.state_projector.state_dict(), root / "state_proj.pt")
    critic.value_head.save_checkpoint(root / "value_head")
    predictor = root / "wm_predictor"
    predictor.mkdir()
    (predictor / "config.json").write_text(
        json.dumps({"grid_tokens": 2, "emb_dim": 2}),
        encoding="utf-8",
    )
    torch.save(
        {"training_invariants": {"value_objective": value_objective}},
        root / "training_state.pt",
    )


def test_joint_critic_matches_project_mean_pool_head_and_backpropagates() -> None:
    critic = _critic()
    hidden = torch.randn(4, 2, 3)
    expected = critic.value_head(critic.state_projector(hidden).mean(dim=1))
    actual = critic(hidden)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (4, 3)
    assert actual.dtype == torch.float32

    actual.sum().backward()
    assert critic.state_projector.net[0].weight.grad is not None
    assert critic.value_head.net[0].weight.grad is not None


def test_joint_critic_preserves_parameter_dtype() -> None:
    critic = _critic().double()
    output = critic(torch.randn(2, 2, 3, dtype=torch.float64))
    assert output.dtype == torch.float64


def test_joint_critic_rejects_wrong_hidden_shape_and_mismatched_components() -> None:
    critic = _critic()
    with pytest.raises(ValueError, match="hidden shape"):
        critic(torch.randn(4, 3))
    with pytest.raises(ValueError, match="hidden shape"):
        critic(torch.randn(4, 3, 3))

    with pytest.raises(ValueError, match="embedding dimension"):
        JointActionValueCritic(
            state_projector=SharedSlotProjector(
                input_dim=3,
                output_dim=2,
                hidden_dim=5,
                grid_tokens=2,
            ),
            value_head=ValueHead(emb_dim=4, num_actions=3),
        )


def test_frozen_snapshot_is_deterministic_detached_and_independent() -> None:
    torch.manual_seed(7)
    critic = _critic()
    hidden = torch.randn(2, 2, 3, requires_grad=True)
    snapshot = create_frozen_critic_snapshot(
        critic,
        source_step=11,
        contract_id="sha256:joint-contract",
        score_dtype="float32",
    )
    same = create_frozen_critic_snapshot(
        critic,
        source_step=11,
        contract_id="sha256:joint-contract",
        score_dtype="float32",
    )
    assert snapshot.snapshot_id == same.snapshot_id
    assert snapshot.source_step == 11
    assert snapshot.contract_id == "sha256:joint-contract"
    assert snapshot.score_dtype == "float32"
    assert snapshot.training is False
    assert all(not parameter.requires_grad for parameter in snapshot.parameters())

    before = snapshot(hidden)
    assert before.shape == (2, 3)
    assert before.requires_grad is False
    assert hidden.grad is None

    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.add_(1.0)
    after = snapshot(hidden)
    torch.testing.assert_close(after, before)
    assert create_frozen_critic_snapshot(
        critic,
        source_step=11,
        contract_id="sha256:joint-contract",
        score_dtype="float32",
    ).snapshot_id != snapshot.snapshot_id


def test_snapshot_rejects_bad_identity_nonfinite_weights_and_train_mode() -> None:
    critic = _critic()
    with pytest.raises(ValueError, match="source_step"):
        create_frozen_critic_snapshot(
            critic,
            source_step=-1,
            contract_id="sha256:joint-contract",
            score_dtype="float32",
        )
    with pytest.raises(ValueError, match="contract_id"):
        create_frozen_critic_snapshot(
            critic,
            source_step=0,
            contract_id="",
            score_dtype="float32",
        )
    with pytest.raises(ValueError, match="score_dtype"):
        create_frozen_critic_snapshot(
            critic,
            source_step=0,
            contract_id="sha256:joint-contract",
            score_dtype="float16",
        )

    with torch.no_grad():
        next(critic.parameters()).flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        create_frozen_critic_snapshot(
            critic,
            source_step=0,
            contract_id="sha256:joint-contract",
            score_dtype="float32",
        )


def test_snapshot_rejects_attempt_to_enable_training() -> None:
    snapshot = create_frozen_critic_snapshot(
        _critic(),
        source_step=0,
        contract_id="sha256:joint-contract",
        score_dtype="float32",
    )
    with pytest.raises(RuntimeError, match="frozen"):
        snapshot.train(True)
    with pytest.raises(RuntimeError, match="frozen"):
        snapshot.requires_grad_(True)
    with pytest.raises(RuntimeError, match="frozen"):
        snapshot.half()
    with pytest.raises(RuntimeError, match="frozen"):
        snapshot.load_state_dict(snapshot.state_dict())
    assert snapshot.train(False) is snapshot
    assert snapshot.requires_grad_(False) is snapshot


@pytest.mark.parametrize(
    "mutation",
    [
        "parameter",
        "critic_mode",
        "nested_mode",
        "child_grad",
        "wrapper_metadata",
        "child_spec",
        "projector_metadata",
    ],
)
def test_snapshot_detects_child_or_metadata_mutation(mutation: str) -> None:
    snapshot = create_frozen_critic_snapshot(
        _critic(),
        source_step=0,
        contract_id="sha256:joint-contract",
        score_dtype="float32",
    )
    if mutation == "parameter":
        with torch.no_grad():
            next(snapshot.critic.parameters()).add_(1.0)
    elif mutation == "critic_mode":
        snapshot.critic.train(True)
    elif mutation == "nested_mode":
        snapshot.critic.state_projector.train(True)
    elif mutation == "child_grad":
        next(snapshot.critic.parameters()).requires_grad_(True)
    elif mutation == "wrapper_metadata":
        snapshot.source_step = 1
    elif mutation == "child_spec":
        snapshot.critic.spec = type(snapshot.critic.spec)(
            **{**snapshot.critic.spec.__dict__, "grid_tokens": 1}
        )
    else:
        snapshot.critic.state_projector.grid_tokens = 1
    with pytest.raises(RuntimeError, match="snapshot|frozen"):
        snapshot(torch.randn(1, 2, 3))


def test_snapshot_identity_binds_source_step_contract_and_dtype() -> None:
    critic = _critic()
    baseline = create_frozen_critic_snapshot(
        critic,
        source_step=0,
        contract_id="sha256:joint-contract",
        score_dtype="float32",
    )
    assert create_frozen_critic_snapshot(
        critic,
        source_step=1,
        contract_id="sha256:joint-contract",
        score_dtype="float32",
    ).snapshot_id != baseline.snapshot_id
    assert create_frozen_critic_snapshot(
        critic,
        source_step=0,
        contract_id="sha256:other-contract",
        score_dtype="float32",
    ).snapshot_id != baseline.snapshot_id
    assert create_frozen_critic_snapshot(
        critic,
        source_step=0,
        contract_id="sha256:joint-contract",
        score_dtype="float64",
    ).snapshot_id != baseline.snapshot_id
    critic.double()
    assert create_frozen_critic_snapshot(
        critic,
        source_step=0,
        contract_id="sha256:joint-contract",
        score_dtype="float32",
    ).snapshot_id != baseline.snapshot_id


def test_strict_checkpoint_loader_reuses_projector_and_value_head(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    loaded = load_joint_action_value_critic(
        checkpoint_root=tmp_path,
        expected_qwen_hidden_dim=3,
        expected_grid_tokens=2,
        expected_state_dim=2,
        expected_action_count=3,
        device=torch.device("cpu"),
        trainable=True,
    )
    assert loaded.spec.qwen_hidden_dim == 3
    assert loaded.spec.projector_hidden_dim == 5
    assert loaded.spec.state_dim == 2
    assert loaded.spec.grid_tokens == 2
    assert loaded.spec.action_count == 3
    assert loaded.spec.value_hidden_dim == 4
    assert loaded.training is True
    assert all(parameter.requires_grad for parameter in loaded.parameters())
    assert loaded(torch.randn(1, 2, 3)).shape == (1, 3)

    frozen = load_joint_action_value_critic(
        checkpoint_root=tmp_path,
        expected_qwen_hidden_dim=3,
        expected_grid_tokens=2,
        expected_state_dim=2,
        expected_action_count=3,
        device=torch.device("cpu"),
        trainable=False,
    )
    assert frozen.training is False
    assert all(not parameter.requires_grad for parameter in frozen.parameters())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"expected_qwen_hidden_dim": 4}, "Qwen hidden"),
        ({"expected_grid_tokens": 3}, "grid token"),
        ({"expected_state_dim": 4}, "state dimension"),
        ({"expected_action_count": 8}, "action count"),
    ],
)
def test_loader_rejects_dimension_mismatch(
    tmp_path: Path,
    override: dict[str, int],
    message: str,
) -> None:
    _write_checkpoint(tmp_path)
    kwargs = {
        "expected_qwen_hidden_dim": 3,
        "expected_grid_tokens": 2,
        "expected_state_dim": 2,
        "expected_action_count": 3,
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        load_joint_action_value_critic(
            checkpoint_root=tmp_path,
            device=torch.device("cpu"),
            trainable=True,
            **kwargs,
        )


def test_loader_rejects_mixed_component_dtypes(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    state = torch.load(tmp_path / "state_proj.pt", weights_only=True)
    state["net.3.weight"] = state["net.3.weight"].double()
    torch.save(state, tmp_path / "state_proj.pt")
    with pytest.raises(ValueError, match="consistent floating dtype"):
        load_joint_action_value_critic(
            checkpoint_root=tmp_path,
            expected_qwen_hidden_dim=3,
            expected_grid_tokens=2,
            expected_state_dim=2,
            expected_action_count=3,
            device=torch.device("cpu"),
            trainable=True,
        )


def test_loader_rejects_projector_value_head_dtype_mismatch(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    state = torch.load(tmp_path / "value_head" / "value_head.pt", weights_only=True)
    state = {name: value.double() for name, value in state.items()}
    torch.save(state, tmp_path / "value_head" / "value_head.pt")
    with pytest.raises(ValueError, match="one consistent dtype"):
        load_joint_action_value_critic(
            checkpoint_root=tmp_path,
            expected_qwen_hidden_dim=3,
            expected_grid_tokens=2,
            expected_state_dim=2,
            expected_action_count=3,
            device=torch.device("cpu"),
            trainable=True,
        )


def test_loader_accepts_compatible_rl_semantics_without_sft2_sidecar(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path)
    (tmp_path / "training_state.pt").unlink()
    torch.save(
        {"planner_training_objective": "receding_horizon_decision_state_ppo_value_v1"},
        tmp_path / "rl_state.pt",
    )
    loaded = load_joint_action_value_critic(
        checkpoint_root=tmp_path,
        expected_qwen_hidden_dim=3,
        expected_grid_tokens=2,
        expected_state_dim=2,
        expected_action_count=3,
        device=torch.device("cpu"),
        trainable=True,
    )
    assert loaded.spec.action_count == 3


def test_loader_rejects_incompatible_rl_semantics(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    (tmp_path / "training_state.pt").unlink()
    torch.save(
        {"planner_training_objective": "successor_state_value"},
        tmp_path / "rl_state.pt",
    )
    with pytest.raises(ValueError, match="incompatible RL value objective"):
        load_joint_action_value_critic(
            checkpoint_root=tmp_path,
            expected_qwen_hidden_dim=3,
            expected_grid_tokens=2,
            expected_state_dim=2,
            expected_action_count=3,
            device=torch.device("cpu"),
            trainable=True,
        )


def test_loader_rejects_ambiguous_semantic_sidecars(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    torch.save(
        {"planner_training_objective": "receding_horizon_decision_state_ppo_value_v1"},
        tmp_path / "rl_state.pt",
    )
    with pytest.raises(ValueError, match="ambiguous value semantics"):
        load_joint_action_value_critic(
            checkpoint_root=tmp_path,
            expected_qwen_hidden_dim=3,
            expected_grid_tokens=2,
            expected_state_dim=2,
            expected_action_count=3,
            device=torch.device("cpu"),
            trainable=True,
        )


def test_loader_rejects_incompatible_value_semantics_and_incomplete_root(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path, value_objective="incoming_action_value")
    with pytest.raises(ValueError, match="incompatible SFT2 value objective"):
        load_joint_action_value_critic(
            checkpoint_root=tmp_path,
            expected_qwen_hidden_dim=3,
            expected_grid_tokens=2,
            expected_state_dim=2,
            expected_action_count=3,
            device=torch.device("cpu"),
            trainable=True,
        )

    (tmp_path / "training_state.pt").unlink()
    with pytest.raises(FileNotFoundError, match="outgoing Q"):
        load_joint_action_value_critic(
            checkpoint_root=tmp_path,
            expected_qwen_hidden_dim=3,
            expected_grid_tokens=2,
            expected_state_dim=2,
            expected_action_count=3,
            device=torch.device("cpu"),
            trainable=True,
        )
