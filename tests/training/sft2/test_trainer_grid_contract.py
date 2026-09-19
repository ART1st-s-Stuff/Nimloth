from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from nimloth.backbone import DINOV2_LARGE_IDENTITY
from nimloth.backbone.dino_grid import (
    STANDALONE_DINO_STATE_CACHE_FORMAT,
    _json_fingerprint,
)
from nimloth.training.sft.stage3.trainer import (
    _validate_dino_grid_contract,
    _validate_stage2_stage3_dino_cache_contract,
)
from nimloth.training.sft.stage3.cli import parse_sft2_args


def _args(model: Path, **overrides):
    values = {
        "model": model,
        "emb_dim": 1024,
        "latent_query_mode": "inject",
        "lambda_sigreg": 0.1,
        "latent_token_count": 64,
        "grid_size": 8,
        "dino_grid_cache": Path("/tmp/dino-cache"),
        "grid_global_tokens": 0,
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


def _write_spatial_cls_cache(
    root: Path,
    *,
    corpus: str,
    teacher_file_digest: str = "same-weights",
) -> str:
    root.mkdir()
    manifest = {
        "format": STANDALONE_DINO_STATE_CACHE_FORMAT,
        "identity": asdict(DINOV2_LARGE_IDENTITY),
        "processor_fingerprint": DINOV2_LARGE_IDENTITY.processor_fingerprint,
        "teacher_provenance": {
            "source": DINOV2_LARGE_IDENTITY.source,
            "revision": DINOV2_LARGE_IDENTITY.revision,
            "files": {"model.safetensors": teacher_file_digest},
        },
        "build_commit": "a" * 40,
        "parent_data_fingerprint": corpus,
        "grid_size": 8,
        "spatial_tokens": 64,
        "global_tokens": 1,
        "state_tokens": 65,
        "global_role": "dino_cls",
        "ordering": "row_major_spatial_then_global",
        "feature_dtype": "float32",
        "images": [{"path": f"/{corpus}.png", "sha256": corpus}],
        "splits": {"train": {"corpus": corpus}, "val": {"corpus": corpus}},
        "shards": [{"file": f"{corpus}.pt", "count": 1, "sha256": corpus}],
        "reuse": {"source": None, "fingerprint": None, "images": 0},
    }
    manifest["fingerprint"] = _json_fingerprint(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "COMPLETED").write_text(manifest["fingerprint"], encoding="utf-8")
    return manifest["fingerprint"]


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


def test_stage3_accepts_compatible_feature_space_across_different_corpora(tmp_path):
    aligned_root = tmp_path / "aligned"
    stage3_root = tmp_path / "stage3"
    aligned_fingerprint = _write_spatial_cls_cache(aligned_root, corpus="aligned")
    stage3_fingerprint = _write_spatial_cls_cache(stage3_root, corpus="future")
    args = _args(
        tmp_path / "model",
        dino_grid_cache=stage3_root,
        stage2_aligned_dino_cache=aligned_root,
        grid_global_tokens=1,
    )

    audit = _validate_stage2_stage3_dino_cache_contract(
        args,
        {"dino_cache_fingerprint": aligned_fingerprint},
    )

    assert aligned_fingerprint != stage3_fingerprint
    assert audit["stage2_aligned_cache_fingerprint"] == aligned_fingerprint
    assert audit["stage3_supervision_cache_fingerprint"] == stage3_fingerprint
    assert audit["feature_space"]["feature_dim"] == 1024


def test_stage3_rejects_different_corpus_without_aligned_cache_anchor(tmp_path):
    stage3_root = tmp_path / "stage3"
    _write_spatial_cls_cache(stage3_root, corpus="future")
    args = _args(
        tmp_path / "model",
        dino_grid_cache=stage3_root,
        stage2_aligned_dino_cache=None,
        grid_global_tokens=1,
    )

    with pytest.raises(ValueError, match="stage2-aligned-dino-cache"):
        _validate_stage2_stage3_dino_cache_contract(
            args,
            {"dino_cache_fingerprint": "different-stage2-corpus"},
        )


def test_stage3_rejects_true_feature_space_mismatch(tmp_path):
    aligned_root = tmp_path / "aligned"
    stage3_root = tmp_path / "stage3"
    aligned_fingerprint = _write_spatial_cls_cache(
        aligned_root,
        corpus="aligned",
        teacher_file_digest="stage2-weights",
    )
    _write_spatial_cls_cache(
        stage3_root,
        corpus="future",
        teacher_file_digest="different-weights",
    )
    args = _args(
        tmp_path / "model",
        dino_grid_cache=stage3_root,
        stage2_aligned_dino_cache=aligned_root,
        grid_global_tokens=1,
    )

    with pytest.raises(ValueError, match="feature-space identity mismatch"):
        _validate_stage2_stage3_dino_cache_contract(
            args,
            {"dino_cache_fingerprint": aligned_fingerprint},
        )


def _cli_args(model: Path, coefficient: str) -> list[str]:
    return [
        "--model", str(model), "--train-jsonl", "train.jsonl",
        "--val-jsonl", "val.jsonl", "--output-dir", "out",
        "--objective", "dino_grid", "--grid-size", "8",
        "--latent-token-count", "64", "--dino-grid-cache", "cache",
        f"--lambda-sigreg={coefficient}",
    ]


@pytest.mark.parametrize("coefficient", [0.0, 0.1, 0.25])
def test_sigreg_coefficient_passes_cli_and_grid_validation(tmp_path, coefficient):
    model = tmp_path / "model"
    _write_state_config(model)
    args = parse_sft2_args(_cli_args(model, str(coefficient)))
    assert args.lambda_sigreg == coefficient
    assert _validate_dino_grid_contract(args)["grid_tokens"] == 64


@pytest.mark.parametrize("coefficient", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_sigreg_coefficient_rejects_invalid_cli_and_direct_calls(tmp_path, coefficient):
    model = tmp_path / "model"
    with pytest.raises(SystemExit):
        parse_sft2_args(_cli_args(model, str(coefficient)))
    with pytest.raises(ValueError, match="lambda_sigreg must be finite and nonnegative"):
        _validate_dino_grid_contract(_args(model, lambda_sigreg=coefficient))


@pytest.mark.parametrize("mutation", ["grid", "teacher", "dimension", "query_mode"])
def test_disabling_sigreg_preserves_state_interface_checks(tmp_path, mutation):
    model = tmp_path / "model"
    _write_state_config(model)
    args = parse_sft2_args(_cli_args(model, "0"))
    if mutation == "teacher":
        path = model / "grid_state_config.json"
        config = json.loads(path.read_text())
        config["dino_identity"] = {}
        path.write_text(json.dumps(config))
    elif mutation == "grid":
        args.latent_token_count = 16
    elif mutation == "dimension":
        args.emb_dim = 512
    else:
        args.latent_query_mode = "generate"
    with pytest.raises(ValueError):
        _validate_dino_grid_contract(args)
