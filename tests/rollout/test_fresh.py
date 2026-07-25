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


def test_fresh_manifest_consumption_is_transactional(tmp_path: Path) -> None:
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
    consumption_id = first.begin_consumption(
        output_dir=tmp_path / "output",
        global_step=0,
    )
    second = FreshJSONLRolloutCollector(manifest_path, model_path=model)
    with pytest.raises(RuntimeError, match="already exists"):
        second.begin_consumption(output_dir=tmp_path / "output", global_step=0)

    first.abort_consumption(consumption_id)
    retry_id = second.begin_consumption(
        output_dir=tmp_path / "output",
        global_step=0,
    )
    checkpoint = tmp_path / "output" / "latest"
    with pytest.raises(RuntimeError, match="complete checkpoint"):
        second.commit_consumption(
            retry_id,
            checkpoint_path=checkpoint,
            global_step=1,
        )
    checkpoint.mkdir(parents=True)
    (checkpoint / "rl_state.pt").write_bytes(b"state")
    second.commit_consumption(
        retry_id,
        checkpoint_path=checkpoint,
        global_step=1,
    )
    with pytest.raises(RuntimeError, match="state=committed"):
        first.begin_consumption(output_dir=tmp_path / "output", global_step=1)


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


def test_fresh_manifest_binds_frozen_reference_and_enriched_jsonl(
    tmp_path: Path,
) -> None:
    model = _policy_artifact(tmp_path / "model")
    reference = _policy_artifact(tmp_path / "reference", payload=b"reference")
    behavior = tmp_path / "behavior.jsonl"
    enriched = tmp_path / "enriched.jsonl"
    _trajectory_jsonl(behavior)
    _trajectory_jsonl(enriched)
    manifest_path = tmp_path / "fresh.json"
    manifest = FreshRolloutManifest.create(
        policy_path=model,
        trajectory_path=behavior,
        num_trajectories=1,
    ).with_reference(
        reference_policy_path=reference,
        trajectory_path=enriched,
    )
    manifest.write(manifest_path)

    restored = FreshRolloutManifest.read(manifest_path)
    assert restored.format_version == 4
    assert restored.trajectory_path == str(enriched.resolve())
    assert restored.behavior_trajectory_path == str(behavior.resolve())
    collector = FreshJSONLRolloutCollector(
        manifest_path,
        model_path=model,
        reference_model_path=reference,
    )
    collector.validate_policy()
    (reference / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="reference fingerprint mismatch"):
        collector.validate_policy()


def test_fresh_manifest_rejects_changed_trajectory_bytes(tmp_path: Path) -> None:
    model = _policy_artifact(tmp_path / "model")
    trajectories = tmp_path / "trajectories.jsonl"
    _trajectory_jsonl(trajectories)
    manifest_path = tmp_path / "fresh.json"
    FreshRolloutManifest.create(
        policy_path=model,
        trajectory_path=trajectories,
        num_trajectories=1,
    ).write(manifest_path)

    trajectories.write_text('{"changed": true}\n', encoding="utf-8")
    collector = FreshJSONLRolloutCollector(manifest_path, model_path=model)
    with pytest.raises(ValueError, match="trajectory fingerprint mismatch"):
        collector.validate_policy()


def test_fresh_collector_requires_the_complete_manifest_batch(tmp_path: Path) -> None:
    model = _policy_artifact(tmp_path / "model")
    trajectories = tmp_path / "trajectories.jsonl"
    _trajectory_jsonl(trajectories)
    manifest_path = tmp_path / "fresh.json"
    FreshRolloutManifest.create(
        policy_path=model,
        trajectory_path=trajectories,
        num_trajectories=2,
    ).write(manifest_path)
    collector = FreshJSONLRolloutCollector(manifest_path, model_path=model)
    collector._all_trajectories = [object()]  # type: ignore[list-item]

    with pytest.raises(ValueError, match="complete manifest batch"):
        collector.collect(num_episodes=1)

    collector = FreshJSONLRolloutCollector(manifest_path, model_path=model)
    collector._all_trajectories = [object()]  # type: ignore[list-item]
    with pytest.raises(ValueError, match="count does not match manifest"):
        collector.collect(num_episodes=2)


def test_reference_manifest_keeps_behavior_bytes_bound(tmp_path: Path) -> None:
    model = _policy_artifact(tmp_path / "model")
    reference = _policy_artifact(tmp_path / "reference", payload=b"reference")
    behavior = tmp_path / "behavior.jsonl"
    enriched = tmp_path / "enriched.jsonl"
    _trajectory_jsonl(behavior)
    _trajectory_jsonl(enriched)
    manifest_path = tmp_path / "fresh.json"
    FreshRolloutManifest.create(
        policy_path=model,
        trajectory_path=behavior,
        num_trajectories=1,
    ).with_reference(
        reference_policy_path=reference,
        trajectory_path=enriched,
    ).write(manifest_path)

    behavior.write_text('{"changed": true}\n', encoding="utf-8")
    collector = FreshJSONLRolloutCollector(
        manifest_path,
        model_path=model,
        reference_model_path=reference,
    )
    with pytest.raises(ValueError, match="behavior trajectory fingerprint mismatch"):
        collector.validate_policy()
