from __future__ import annotations

from pathlib import Path

import pytest

from nimloth.rollout.fresh import (
    FreshJSONLRolloutCollector,
    FreshRolloutManifest,
    policy_artifact_fingerprint,
)


def _policy_artifact(root: Path, payload: bytes = b"weights") -> Path:
    root.mkdir()
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    (root / "model.safetensors").write_bytes(payload)
    return root


def _trajectory_jsonl(path: Path) -> None:
    # Collector construction is intentionally lazy; freshness tests do not need
    # to duplicate the complete rollout schema fixture.
    path.write_text("{}\n", encoding="utf-8")


def test_policy_fingerprint_tracks_weight_contents(tmp_path: Path) -> None:
    model = _policy_artifact(tmp_path / "model")
    before = policy_artifact_fingerprint(model)
    (model / "model.safetensors").write_bytes(b"changed")
    assert policy_artifact_fingerprint(model) != before


def test_fresh_manifest_validates_exact_policy(tmp_path: Path) -> None:
    model = _policy_artifact(tmp_path / "model")
    trajectories = tmp_path / "trajectories.jsonl"
    _trajectory_jsonl(trajectories)
    manifest_path = tmp_path / "fresh.json"
    FreshRolloutManifest.create(
        policy_path=model,
        trajectory_path=trajectories,
        num_trajectories=1,
    ).write(manifest_path)

    collector = FreshJSONLRolloutCollector(manifest_path, model_path=model)
    collector.validate_policy()
    (model / "model.safetensors").write_bytes(b"stale")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        collector.validate_policy()


def test_fresh_manifest_rejects_second_claim(tmp_path: Path) -> None:
    model = _policy_artifact(tmp_path / "model")
    trajectories = tmp_path / "trajectories.jsonl"
    _trajectory_jsonl(trajectories)
    manifest_path = tmp_path / "fresh.json"
    FreshRolloutManifest.create(
        policy_path=model,
        trajectory_path=trajectories,
        num_trajectories=1,
    ).write(manifest_path)
    first = FreshJSONLRolloutCollector(manifest_path, model_path=model)
    first._claim_once()
    second = FreshJSONLRolloutCollector(manifest_path, model_path=model)
    with pytest.raises(RuntimeError, match="already consumed"):
        second._claim_once()


def test_fresh_manifest_binds_planner_modules(tmp_path: Path) -> None:
    model = _policy_artifact(tmp_path / "model")
    planner = tmp_path / "wm_predictor"
    planner.mkdir()
    (planner / "predictor.pt").write_bytes(b"wm")
    trajectories = tmp_path / "trajectories.jsonl"
    _trajectory_jsonl(trajectories)
    manifest_path = tmp_path / "fresh.json"
    FreshRolloutManifest.create(
        policy_path=model,
        trajectory_path=trajectories,
        num_trajectories=1,
        planner_artifacts={"wm_predictor": planner},
    ).write(manifest_path)

    collector = FreshJSONLRolloutCollector(
        manifest_path,
        model_path=model,
        planner_artifacts={"wm_predictor": planner},
    )
    collector.validate_policy()
    (planner / "predictor.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="planner fingerprint mismatch"):
        collector.validate_policy()
