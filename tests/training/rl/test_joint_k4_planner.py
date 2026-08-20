from __future__ import annotations

import copy

import pytest
import torch

from nimloth.training.rl.joint_planner import (
    FrozenMCTSPlanningConfig,
    JointWorldModelCritic,
    create_frozen_planning_snapshot,
    export_frozen_planning_snapshot,
    restore_frozen_planning_snapshot,
)
from nimloth.wm.grid import (
    GridPredictorConfig,
    SharedSlotProjector,
    TemporalSpatialGridPredictor,
)
from nimloth.wm.value_head import ValueHead


def _tiny_world_model() -> JointWorldModelCritic:
    torch.manual_seed(19)
    return JointWorldModelCritic(
        state_projector=SharedSlotProjector(
            input_dim=3,
            hidden_dim=5,
            output_dim=4,
            grid_tokens=2,
        ),
        wm_predictor=TemporalSpatialGridPredictor(
            GridPredictorConfig(
                grid_tokens=2,
                emb_dim=4,
                action_dim=2,
                history_size=1,
                depth=1,
                heads=1,
                dim_head=4,
                mlp_dim=8,
                dropout=0.0,
            )
        ),
        value_head=ValueHead(emb_dim=4, num_actions=2, hidden_dim=4),
    )


def _planning_config() -> FrozenMCTSPlanningConfig:
    return FrozenMCTSPlanningConfig(
        horizon=4,
        num_simulations=8,
        exploration_constant=1.0,
    )


def test_frozen_k4_snapshot_scores_direct_q_and_mcts_root_means() -> None:
    snapshot = create_frozen_planning_snapshot(
        _tiny_world_model(),
        source_step=776,
        contract_id="sha256:" + "a" * 64,
        score_dtype="float32",
        planning_config=_planning_config(),
    )
    hidden = torch.tensor(
        [[[0.1, 0.2, 0.3], [0.4, -0.2, 0.5]]],
        dtype=torch.float32,
    )

    scored = snapshot.score(hidden)

    assert scored.direct_all_action_q.shape == (1, 2)
    assert scored.planner_root_mean_values.shape == (1, 2)
    assert scored.root_visit_counts.shape == (1, 2)
    assert scored.root_visit_counts.sum().item() == 8
    assert torch.all(scored.root_visit_counts > 0)
    assert scored.candidate_sequences.ndim == 3
    assert scored.candidate_sequences.shape[0] == 1
    assert scored.candidate_sequences.shape[-1] == 4
    assert torch.isfinite(scored.direct_all_action_q).all()
    assert torch.isfinite(scored.planner_root_mean_values).all()
    assert torch.isfinite(scored.candidate_mean_values).all()
    assert torch.equal(scored.direct_all_action_q, snapshot.score(hidden).direct_all_action_q)
    assert torch.equal(
        scored.planner_root_mean_values,
        snapshot.score(hidden).planner_root_mean_values,
    )

    captured = snapshot.score(hidden, capture_mcts_trace=True)
    assert captured.current_state.shape == (1, 2, 4)
    assert captured.mcts_trace is not None
    assert len(captured.mcts_trace["simulations"]) == 8
    assert len(captured.mcts_trace["tree_nodes"]) > 4
    assert all(
        node["predicted_state"].shape == (2, 4)
        for node in captured.mcts_trace["tree_nodes"][1:]
    )


def test_frozen_k4_snapshot_export_restore_is_exact_and_device_explicit() -> None:
    original = create_frozen_planning_snapshot(
        _tiny_world_model(),
        source_step=9,
        contract_id="sha256:" + "b" * 64,
        score_dtype="float32",
        planning_config=_planning_config(),
    )
    state = export_frozen_planning_snapshot(original)
    restored = restore_frozen_planning_snapshot(
        state.to_mapping(),
        device=torch.device("cpu"),
    )
    hidden = torch.randn(1, 2, 3)

    assert restored.snapshot_id == original.snapshot_id
    assert restored.source_step == 9
    assert restored.contract_id == original.contract_id
    assert restored.planning_config == _planning_config()
    assert torch.equal(
        restored.score(hidden).direct_all_action_q,
        original.score(hidden).direct_all_action_q,
    )
    assert torch.equal(
        restored.score(hidden).planner_root_mean_values,
        original.score(hidden).planner_root_mean_values,
    )

    tampered = copy.deepcopy(state.to_mapping())
    first_name = next(iter(tampered["model_state"]))
    tampered["model_state"][first_name].reshape(-1)[0] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        restore_frozen_planning_snapshot(tampered, device=torch.device("cpu"))


def test_frozen_k4_snapshot_rejects_live_parameter_mutation() -> None:
    snapshot = create_frozen_planning_snapshot(
        _tiny_world_model(),
        source_step=1,
        contract_id="sha256:" + "c" * 64,
        score_dtype="float32",
        planning_config=_planning_config(),
    )
    with torch.no_grad():
        next(snapshot.parameters()).reshape(-1)[0] += 1

    with pytest.raises(RuntimeError, match="changed after publication"):
        snapshot.score(torch.randn(1, 2, 3))


def test_k4_planning_config_is_strict() -> None:
    with pytest.raises(ValueError, match="horizon must be exactly 4"):
        FrozenMCTSPlanningConfig(
            horizon=3,
            num_simulations=8,
            exploration_constant=1.0,
        )
    with pytest.raises(ValueError, match="visit every root action"):
        create_frozen_planning_snapshot(
            _tiny_world_model(),
            source_step=1,
            contract_id="sha256:" + "d" * 64,
            score_dtype="float32",
            planning_config=FrozenMCTSPlanningConfig(
                horizon=4,
                num_simulations=1,
                exploration_constant=1.0,
            ),
        )
