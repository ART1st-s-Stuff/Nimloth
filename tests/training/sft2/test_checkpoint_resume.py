"""Tests for SFT2 resume checkpoint selection."""

from __future__ import annotations

from pathlib import Path
import random

import pytest
import torch

from nimloth.training.sft2.checkpoint import (
    find_resume_checkpoint,
    is_trainable_checkpoint_dir,
    load_aux_checkpoint,
    resolve_resume_checkpoint_dir,
    resume_epoch_and_micro_step,
)
from nimloth.training.sft2.components import require_sft2_wm_history
from nimloth.training.sft2.utils import seed_training_micro_step, training_micro_seed
from nimloth.wm.lewm import LeWMConfig
from nimloth.wm.predictor import LatentWMPredictor


def _write_aux_markers(ckpt_dir: Path) -> None:
    torch.save({}, ckpt_dir / "state_proj.pt")
    (ckpt_dir / "wm_predictor").mkdir()
    (ckpt_dir / "wm_predictor" / "config.json").write_text("{}", encoding="utf-8")
    torch.save({}, ckpt_dir / "wm_predictor" / "predictor.pt")
    (ckpt_dir / "value_head").mkdir()
    torch.save({}, ckpt_dir / "value_head" / "value_head.pt")


def _write_ckpt(ckpt_dir: Path, *, step: int, epoch: int) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text("{}", encoding="utf-8")
    torch.save({"step": step, "epoch": epoch}, ckpt_dir / "training_state.pt")
    _write_aux_markers(ckpt_dir)


def test_counter_based_micro_seed_replays_stochastic_operations() -> None:
    seed = seed_training_micro_step(42, epoch=3, micro_step=7, rank=1)
    first = (random.random(), torch.rand(4))
    random.random()
    torch.rand(11)
    assert seed_training_micro_step(42, epoch=3, micro_step=7, rank=1) == seed
    second = (random.random(), torch.rand(4))

    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])
    assert training_micro_seed(42, 3, 7, 0) != training_micro_seed(42, 3, 7, 1)


def test_resume_position_for_epoch_complete_and_legacy_checkpoints() -> None:
    assert resume_epoch_and_micro_step({"epoch": 3, "epoch_complete": True}) == (4, 0)
    assert resume_epoch_and_micro_step({"epoch": 3}) == (4, 0)


def test_resume_position_for_partial_epoch_checkpoint() -> None:
    assert resume_epoch_and_micro_step(
        {"epoch": 3, "epoch_complete": False, "micro_step_in_epoch": 17}
    ) == (3, 17)


def test_find_resume_checkpoint_prefers_latest_epoch(tmp_path: Path) -> None:
    out = tmp_path / "run"
    _write_ckpt(out / "best", step=855, epoch=1)
    _write_ckpt(out / "epoch_001", step=855, epoch=1)
    _write_ckpt(out / "epoch_002", step=1710, epoch=2)

    assert find_resume_checkpoint(out) == out / "epoch_002"


def test_resolve_resume_checkpoint_dir_explicit_path(tmp_path: Path) -> None:
    out = tmp_path / "run"
    _write_ckpt(out / "epoch_002", step=1710, epoch=2)

    resolved = resolve_resume_checkpoint_dir(out, Path("epoch_002"))
    assert resolved == out / "epoch_002"


def test_resume_rejects_query_tune_mismatch(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    proj = torch.nn.Linear(3, 2)
    proj.latent_token_count = 1
    proj.qwen_hidden_dim = 3
    proj.input_dim = 3
    torch.save(proj.state_dict(), ckpt / "state_proj.pt")
    torch.save(
        {
            "latent_token_count": 1,
            "latent_query_mode": "inject",
            "query_tune": "freeze",
            "qwen_hidden_dim": 3,
            "state_proj_input_dim": 3,
        },
        ckpt / "training_state.pt",
    )
    (ckpt / "wm_predictor").mkdir()
    (ckpt / "wm_predictor" / "config.json").write_text("{}", encoding="utf-8")
    torch.save({}, ckpt / "wm_predictor" / "predictor.pt")
    (ckpt / "value_head").mkdir()
    torch.save({}, ckpt / "value_head" / "value_head.pt")

    with pytest.raises(ValueError, match="checkpoint query_tune mismatch"):
        load_aux_checkpoint(
            ckpt,
            proj,
            object(),
            object(),
            torch.device("cpu"),
            latent_query_mode="inject",
            query_tune="adapter",
        )


def test_resume_checkpoint_requires_all_auxiliary_weights(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint"
    _write_ckpt(ckpt, step=1, epoch=1)
    (ckpt / "value_head" / "value_head.pt").unlink()

    assert not is_trainable_checkpoint_dir(ckpt)
    with pytest.raises(FileNotFoundError, match="incomplete SFT2 auxiliary checkpoint"):
        load_aux_checkpoint(
            ckpt,
            object(),
            object(),
            object(),
            torch.device("cpu"),
        )


def test_sft2_rejects_multi_step_predictor_checkpoint() -> None:
    predictor = LatentWMPredictor.create(LeWMConfig(emb_dim=16, history_size=4))

    with pytest.raises(ValueError, match="history_size=1"):
        require_sft2_wm_history(predictor, Path("wm"))
