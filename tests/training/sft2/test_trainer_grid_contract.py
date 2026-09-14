from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from nimloth.backbone import DINOV2_LARGE_IDENTITY
from nimloth.training.sft.stage3.trainer import _validate_dino_grid_contract


def _args(model: Path, **overrides):
    values = {
        "model": model,
        "emb_dim": 1024,
        "latent_query_mode": "inject",
        "lambda_sigreg": 0.1,
        "latent_token_count": 64,
        "grid_size": 8,
        "dino_grid_cache": Path("/tmp/dino-cache"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_state_config(model: Path, *, grid_tokens: int = 64, grid_size: int = 8):
    model.mkdir()
    (model / "grid_state_config.json").write_text(
        json.dumps(
            {
                "objective": {"grid_size": grid_size},
                "dino_identity": asdict(DINOV2_LARGE_IDENTITY),
                "grid_tokens": grid_tokens,
                "state_dim": 1024,
                "shared_slot_projector": True,
                "ordering": "row_major",
            }
        ),
        encoding="utf-8",
    )


def test_dino_grid_contract_accepts_checkpoint_matched_k64_grid8(tmp_path: Path):
    model = tmp_path / "model"
    _write_state_config(model)

    config = _validate_dino_grid_contract(_args(model))

    assert config["grid_tokens"] == 64
    assert config["objective"]["grid_size"] == 8


def test_dino_grid_contract_rejects_token_grid_mismatch(tmp_path: Path):
    model = tmp_path / "model"
    _write_state_config(model)

    with pytest.raises(ValueError, match="token/grid mismatch"):
        _validate_dino_grid_contract(_args(model, latent_token_count=16))


def test_dino_grid_contract_rejects_checkpoint_grid_mismatch(tmp_path: Path):
    model = tmp_path / "model"
    _write_state_config(model, grid_tokens=16, grid_size=4)

    with pytest.raises(ValueError, match="state interface mismatch"):
        _validate_dino_grid_contract(_args(model))
