"""Tests for SFT2 resume checkpoint selection."""

from __future__ import annotations

from pathlib import Path

import torch

from nimloth.training.sft2.checkpoint import (
    find_resume_checkpoint,
    resolve_resume_checkpoint_dir,
    resume_epoch_and_micro_step,
)


def _write_ckpt(ckpt_dir: Path, *, step: int, epoch: int) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text("{}", encoding="utf-8")
    torch.save({"step": step, "epoch": epoch}, ckpt_dir / "training_state.pt")


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
