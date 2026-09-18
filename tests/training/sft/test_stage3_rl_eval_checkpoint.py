from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nimloth.training.sft.stage3.cli import parse_sft2_args
from nimloth.training.sft.stage3.trainer import (
    _load_rl_eval_grid_world_model,
    _validate_rl_eval_checkpoint_contract,
)
from nimloth.wm.grid import (
    GridPredictorConfig,
    ResidualTemporalSpatialGridPredictor,
    SharedSlotProjector,
)
from nimloth.wm.outcome import ActionOutcomeHead
from nimloth.wm.value_head import ValueHead


def _required_cli(root: Path) -> list[str]:
    return [
        "--model", str(root),
        "--train-jsonl", str(root / "train.jsonl"),
        "--val-jsonl", str(root / "eval.jsonl"),
        "--output-dir", str(root / "output"),
        "--objective", "dino_grid",
        "--dino-grid-cache", str(root / "dino"),
    ]


def _write_rl_checkpoint(
    root: Path,
) -> tuple[
    SharedSlotProjector,
    ResidualTemporalSpatialGridPredictor,
    ValueHead,
    ActionOutcomeHead,
]:
    root.mkdir()
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    torch.save(
        {"outcome_config": {"enabled": True}, "global_step": 10},
        root / "rl_state.pt",
    )
    projector = SharedSlotProjector(
        input_dim=3, output_dim=2, hidden_dim=5, grid_tokens=4
    )
    torch.save(projector.state_dict(), root / "state_proj.pt")
    predictor = ResidualTemporalSpatialGridPredictor(
        GridPredictorConfig(
            grid_tokens=4,
            emb_dim=2,
            history_size=1,
            depth=1,
            heads=1,
            dim_head=2,
            mlp_dim=4,
            dropout=0.0,
        )
    )
    predictor.save_checkpoint(root / "wm_predictor")
    value_head = ValueHead(2)
    value_head.save_checkpoint(root / "value_head")
    outcome_head = ActionOutcomeHead(2)
    torch.save(
        {
            "schema": outcome_head.schema,
            "emb_dim": outcome_head.emb_dim,
            "state_dict": outcome_head.state_dict(),
        },
        root / "outcome_head.pt",
    )
    return projector, predictor, value_head, outcome_head


def test_rl_eval_cli_requires_eval_only_and_same_qwen_root(tmp_path: Path) -> None:
    root = tmp_path / "rl"
    other = tmp_path / "other"
    with pytest.raises(SystemExit):
        parse_sft2_args([*_required_cli(root), "--rl-eval-checkpoint", str(root)])
    with pytest.raises(SystemExit):
        parse_sft2_args(
            [
                *_required_cli(other),
                "--eval-only",
                "--rl-eval-checkpoint",
                str(root),
            ]
        )


def test_rl_eval_cli_rejects_component_and_resume_mixing(tmp_path: Path) -> None:
    root = tmp_path / "rl"
    common = [
        *_required_cli(root),
        "--eval-only",
        "--rl-eval-checkpoint", str(root),
    ]
    with pytest.raises(SystemExit):
        parse_sft2_args([*common, "--resume"])
    with pytest.raises(SystemExit):
        parse_sft2_args(
            [*common, "--wm-predictor-checkpoint", str(tmp_path / "wm")]
        )


def test_rl_eval_loads_every_stage3_component_from_one_root(tmp_path: Path) -> None:
    root = tmp_path / "rl"
    expected_modules = _write_rl_checkpoint(root)
    args = Namespace(
        model=root,
        rl_eval_checkpoint=root,
        outcome_head=True,
        grid_predictor_kind="residual",
        latent_token_count=4,
        grid_size=2,
        emb_dim=2,
        history_size=1,
        grid_wm_depth=1,
        grid_wm_heads=1,
        grid_wm_dim_head=2,
        grid_wm_mlp_dim=4,
        grid_wm_dropout=0.0,
    )
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(hidden_size=3)

    loaded, loaded_device = _load_rl_eval_grid_world_model(
        args, model=model, device=torch.device("cpu"), pair_parallel=False
    )

    assert loaded_device == torch.device("cpu")
    for expected, actual in zip(expected_modules, loaded.trainable_modules, strict=True):
        expected_state = expected.state_dict()
        actual_state = actual.state_dict()
        assert expected_state.keys() == actual_state.keys()
        for name in expected_state:
            torch.testing.assert_close(actual_state[name], expected_state[name])


def test_rl_eval_fails_closed_for_missing_or_disabled_outcome(tmp_path: Path) -> None:
    root = tmp_path / "rl"
    _write_rl_checkpoint(root)
    args = Namespace(model=root, rl_eval_checkpoint=root, outcome_head=True)
    (root / "outcome_head.pt").unlink()
    with pytest.raises(FileNotFoundError, match="outcome_head.pt"):
        _validate_rl_eval_checkpoint_contract(args)

    outcome_head = ActionOutcomeHead(2)
    torch.save(
        {
            "schema": outcome_head.schema,
            "emb_dim": outcome_head.emb_dim,
            "state_dict": outcome_head.state_dict(),
        },
        root / "outcome_head.pt",
    )
    torch.save({"outcome_config": {"enabled": False}}, root / "rl_state.pt")
    with pytest.raises(ValueError, match="enabled OutcomeHead"):
        _validate_rl_eval_checkpoint_contract(args)


def test_rl_eval_rejects_predictor_config_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "rl"
    _write_rl_checkpoint(root)
    args = Namespace(
        model=root,
        rl_eval_checkpoint=root,
        outcome_head=True,
        grid_predictor_kind="residual",
        latent_token_count=4,
        grid_size=2,
        emb_dim=2,
        history_size=1,
        grid_wm_depth=1,
        grid_wm_heads=1,
        grid_wm_dim_head=2,
        grid_wm_mlp_dim=4,
        grid_wm_dropout=0.25,
    )
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(hidden_size=3)

    with pytest.raises(ValueError, match="grid_wm_dropout"):
        _load_rl_eval_grid_world_model(
            args, model=model, device=torch.device("cpu"), pair_parallel=False
        )


def test_rl_eval_rejects_malformed_outcome_payload(tmp_path: Path) -> None:
    root = tmp_path / "rl"
    _write_rl_checkpoint(root)
    torch.save(["not", "a", "checkpoint"], root / "outcome_head.pt")
    args = Namespace(
        model=root,
        rl_eval_checkpoint=root,
        outcome_head=True,
        grid_predictor_kind="residual",
        latent_token_count=4,
        grid_size=2,
        emb_dim=2,
        history_size=1,
        grid_wm_depth=1,
        grid_wm_heads=1,
        grid_wm_dim_head=2,
        grid_wm_mlp_dim=4,
        grid_wm_dropout=0.0,
    )
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(hidden_size=3)

    with pytest.raises(ValueError, match="must contain a dictionary"):
        _load_rl_eval_grid_world_model(
            args, model=model, device=torch.device("cpu"), pair_parallel=False
        )
