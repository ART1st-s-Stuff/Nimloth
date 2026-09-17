"""DINO-grid SFT2 checkpoint 到 RL world-model 的严格装载测试。"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from nimloth.config.rl import parse_rl_config
from nimloth.agent.planning import WorldModelPlanner
from nimloth.training.rl.planning_loader import load_planning_world_model
from nimloth.training.sft.stage3.algorithm import SFT2_VALUE_OBJECTIVE
from nimloth.training.rl.trainer import _build_world_model
from nimloth.wm.grid import (
    GridPredictorConfig,
    GridWorldModel,
    SharedSlotProjector,
    TemporalSpatialGridPredictor,
    ResidualTemporalSpatialGridPredictor,
)
from nimloth.wm.value_head import ValueHead
from nimloth.wm.outcome import ActionOutcomeHead


@pytest.mark.parametrize(
    "predictor_type",
    [TemporalSpatialGridPredictor, ResidualTemporalSpatialGridPredictor],
)
def test_rl_loads_self_contained_grid_state_without_sft1_sidecars(
    tmp_path,
    predictor_type,
) -> None:
    state_proj = SharedSlotProjector(
        input_dim=3,
        output_dim=2,
        hidden_dim=5,
        grid_tokens=2,
    )
    predictor = predictor_type(
        GridPredictorConfig(
            grid_tokens=2,
            emb_dim=2,
            history_size=2,
            depth=1,
            heads=1,
            dim_head=2,
            mlp_dim=4,
            dropout=0.0,
        )
    )
    source = GridWorldModel(
        state_proj=state_proj,
        wm_predictor=predictor,
        value_head=ValueHead(emb_dim=2),
    )
    torch.save(state_proj.state_dict(), tmp_path / "state_proj.pt")
    predictor.save_checkpoint(tmp_path / "wm_predictor")
    source.value_head.save_checkpoint(tmp_path / "value_head")

    llm = torch.nn.Linear(1, 1, bias=False)
    llm.config = SimpleNamespace(hidden_size=3)
    config = parse_rl_config(
        {
            "freeze": {"state_proj": True},
            "gradient": {
                "state_source": "recompute",
                "representation_to_backbone": True,
            },
            "predictor": {"emb_dim": 2, "history_size": 2},
            "validation": {"enabled": False, "envs": 0},
        }
    )
    args = Namespace(
        model=tmp_path,
        wm_checkpoint=tmp_path / "wm_predictor",
        state_proj_checkpoint=tmp_path / "state_proj.pt",
        value_head_checkpoint=tmp_path / "value_head",
    )

    loaded = _build_world_model(
        args,
        config,
        llm=llm,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded, GridWorldModel)
    assert isinstance(loaded.wm_predictor, predictor_type)
    assert loaded.wm_predictor.config.history_size == 2
    assert loaded.state_proj.hidden_dim == 5
    assert all(not parameter.requires_grad for parameter in loaded.state_proj.parameters())


@pytest.mark.parametrize("predictor_type", [TemporalSpatialGridPredictor, ResidualTemporalSpatialGridPredictor])
def test_planning_loader_preserves_grid_rollout_and_value_contract(tmp_path, predictor_type) -> None:
    state_proj = SharedSlotProjector(
        input_dim=3,
        output_dim=2,
        hidden_dim=5,
        grid_tokens=2,
    )
    predictor = predictor_type(
        GridPredictorConfig(
            grid_tokens=2,
            emb_dim=2,
            history_size=2,
            depth=1,
            heads=1,
            dim_head=2,
            mlp_dim=4,
            dropout=0.0,
        )
    )
    value_head = ValueHead(emb_dim=2)
    torch.save(state_proj.state_dict(), tmp_path / "state_proj.pt")
    predictor.save_checkpoint(tmp_path / "wm_predictor")
    value_head.save_checkpoint(tmp_path / "value_head")
    torch.save(
        {
            "training_invariants": {
                "value_objective": SFT2_VALUE_OBJECTIVE,
            }
        },
        tmp_path / "training_state.pt",
    )

    planning_model = load_planning_world_model(
        qwen_config=SimpleNamespace(hidden_size=3),
        wm_checkpoint=tmp_path / "wm_predictor",
        state_proj_checkpoint=tmp_path / "state_proj.pt",
        value_head_checkpoint=tmp_path / "value_head",
        device=torch.device("cpu"),
    )
    state = planning_model.project_state(torch.randn(1, 2, 3))
    assert state.shape == (1, 2, 2)
    assert planning_model.predict_action_values(state).shape == (1, 8)

    plan = WorldModelPlanner(
        planning_model,
        horizon=2,
        search_mode="greedy",
    ).plan(
        state.unsqueeze(1),
        torch.empty((1, 0), dtype=torch.long),
    )
    assert plan.candidate_sequences.shape == (1, 2)
    assert plan.candidate_scores.shape == (1,)
    assert plan.root_action_scores.shape == (8,)


def test_planning_loader_rejects_incoming_action_value_semantics(tmp_path) -> None:
    (tmp_path / "wm_predictor").mkdir()
    (tmp_path / "state_proj.pt").touch()
    (tmp_path / "value_head").mkdir()
    torch.save(
        {
            "training_invariants": {
                "value_objective": "predicted_rollout_executed_action_mc_v2",
            }
        },
        tmp_path / "training_state.pt",
    )

    with pytest.raises(ValueError, match="incompatible SFT2 value objective"):
        load_planning_world_model(
            qwen_config=SimpleNamespace(hidden_size=3),
            wm_checkpoint=tmp_path / "wm_predictor",
            state_proj_checkpoint=tmp_path / "state_proj.pt",
            value_head_checkpoint=tmp_path / "value_head",
            device=torch.device("cpu"),
        )


def test_rl_loads_outcome_head_from_same_stage3_checkpoint(tmp_path) -> None:
    state_proj = SharedSlotProjector(
        input_dim=3, output_dim=2, hidden_dim=5, grid_tokens=2
    )
    predictor = ResidualTemporalSpatialGridPredictor(
        GridPredictorConfig(
            grid_tokens=2,
            emb_dim=2,
            history_size=2,
            depth=1,
            heads=1,
            dim_head=2,
            mlp_dim=4,
            dropout=0.0,
        )
    )
    value_head = ValueHead(emb_dim=2)
    outcome_head = ActionOutcomeHead(2)
    torch.save(state_proj.state_dict(), tmp_path / "state_proj.pt")
    predictor.save_checkpoint(tmp_path / "wm_predictor")
    value_head.save_checkpoint(tmp_path / "value_head")
    torch.save(
        {
            "schema": outcome_head.schema,
            "emb_dim": outcome_head.emb_dim,
            "state_dict": outcome_head.state_dict(),
        },
        tmp_path / "outcome_head.pt",
    )
    llm = torch.nn.Linear(1, 1, bias=False)
    llm.config = SimpleNamespace(hidden_size=3)
    config = parse_rl_config(
        {
            "freeze": {"state_proj": False},
            "gradient": {
                "state_source": "recompute",
                "representation_to_backbone": False,
            },
            "predictor": {
                "emb_dim": 2,
                "history_size": 2,
                "train_wm": True,
            },
            "outcome_head": {"enabled": True, "lr": 1e-4, "lambda_bce": 1.0},
            "validation": {"enabled": False, "envs": 0},
        }
    )
    args = Namespace(
        model=tmp_path,
        wm_checkpoint=tmp_path / "wm_predictor",
        state_proj_checkpoint=tmp_path / "state_proj.pt",
        value_head_checkpoint=tmp_path / "value_head",
        outcome_head_checkpoint=tmp_path / "outcome_head.pt",
    )

    loaded = _build_world_model(
        args, config, llm=llm, device=torch.device("cpu")
    )

    assert isinstance(loaded.outcome_head, ActionOutcomeHead)
    for expected, actual in zip(
        outcome_head.parameters(), loaded.outcome_head.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)
