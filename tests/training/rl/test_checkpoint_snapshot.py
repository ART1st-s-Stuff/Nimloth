from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from nimloth.training.rl.algorithm import PLANNER_TRAINING_OBJECTIVE
from nimloth.training.rl.checkpoint import link_checkpoint_snapshot
from nimloth.training.rl.trainer import _load_resume_state


def test_checkpoint_snapshot_hardlinks_complete_tree(tmp_path: Path) -> None:
    source = tmp_path / "latest"
    (source / "wm_predictor").mkdir(parents=True)
    (source / "rl_state.pt").write_bytes(b"optimizer")
    (source / "wm_predictor" / "predictor.pt").write_bytes(b"predictor")

    snapshot = tmp_path / "iter_0001"
    link_checkpoint_snapshot(source, snapshot)

    for relative in (Path("rl_state.pt"), Path("wm_predictor/predictor.pt")):
        source_file = source / relative
        snapshot_file = snapshot / relative
        assert snapshot_file.read_bytes() == source_file.read_bytes()
        assert os.stat(snapshot_file).st_ino == os.stat(source_file).st_ino


def test_resume_rejects_old_incoming_action_planner_objective(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter])
    state = {
        "optimizer": optimizer.state_dict(),
        "planner_config": {"enabled": True},
        "planner_training_objective": "receding_horizon_transition_mc_v1",
        "reference_kl_config": {"weight": 0.0, "type": None},
        "train_world_model": True,
    }
    monkeypatch.setattr(
        "nimloth.training.rl.trainer.load_rl_wm_checkpoint",
        lambda *_args, **_kwargs: state,
    )

    with pytest.raises(ValueError, match="planner training objective mismatch"):
        _load_resume_state(
            checkpoint_dir=tmp_path,
            world_model=object(),  # type: ignore[arg-type]
            optimizer=optimizer,
            device=torch.device("cpu"),
            rank=0,
            world_size=1,
            optimizer_state_sharded=False,
            expected_checkpoint_metric="success_rate",
            expected_credit_assignment="none",
            expected_token_credit_config={},
            expected_truncated_bootstrap=None,
            expected_planner_config={"enabled": True},
            expected_planner_training_objective=PLANNER_TRAINING_OBJECTIVE,
            expected_reference_kl_config={"weight": 0.0, "type": None},
            expected_train_world_model=True,
        )
