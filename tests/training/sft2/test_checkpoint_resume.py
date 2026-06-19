"""Tests for SFT2 resume checkpoint selection."""

from __future__ import annotations

from pathlib import Path

import torch

from nimloth.training.sft2.checkpoint import find_resume_checkpoint, resolve_resume_checkpoint_dir


def _write_ckpt(ckpt_dir: Path, *, step: int, epoch: int) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text("{}", encoding="utf-8")
    torch.save({"step": step, "epoch": epoch}, ckpt_dir / "training_state.pt")


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
