"""Strictly merge independent fresh-policy rollout shards."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from nimloth.rollout.fresh import (
    FreshRolloutManifest,
    file_artifact_fingerprint,
)
from nimloth.rollout.schema import RolloutTrajectory
from nimloth.rollout.storage import load_trajectories, save_trajectories
from nimloth.rollout.validation import validate_rollout_trajectory


def _rollout_identity(manifest: FreshRolloutManifest) -> tuple[object, ...]:
    """Fields that must be identical for shards to describe one policy batch."""

    return (
        manifest.format_version,
        manifest.policy_fingerprint,
        manifest.policy_path,
        manifest.processor_min_pixels,
        manifest.processor_max_pixels,
        manifest.planner_fingerprints,
        manifest.planner_paths,
    )


def merge_fresh_rollout_shards(
    manifest_paths: Sequence[Path],
    *,
    output_dir: Path,
    expected_record_ids: Sequence[str],
) -> tuple[list[RolloutTrajectory], FreshRolloutManifest]:
    """Merge ordered shards and bind a new manifest to the combined JSONL.

    The order of ``manifest_paths`` is the global rollout order.  Every shard
    must come from the exact same policy, processor and planner artifacts, must
    still be unconsumed, and must contribute the exact expected record IDs.
    """

    if not manifest_paths:
        raise ValueError("fresh rollout merge requires at least one shard")
    target_dir = Path(output_dir).resolve()
    trajectory_path = target_dir / "trajectories.jsonl"
    manifest_path = target_dir / "fresh_policy_manifest.json"
    if trajectory_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite merged rollout artifacts in {target_dir}"
        )

    manifests: list[FreshRolloutManifest] = []
    trajectories: list[RolloutTrajectory] = []
    shard_counts: list[int] = []
    for raw_path in manifest_paths:
        current_path = Path(raw_path).resolve()
        manifest = FreshRolloutManifest.read(current_path)
        manifest.validate_trajectory_artifacts()
        if manifest.reference_policy_fingerprint is not None:
            raise ValueError("cannot merge reference-enriched rollout shards")
        consumption_path = current_path.with_suffix(
            current_path.suffix + ".consumption.json"
        )
        if consumption_path.exists():
            raise ValueError(f"cannot merge consumed rollout shard: {current_path}")
        current = load_trajectories(Path(manifest.trajectory_path))
        if len(current) != manifest.num_trajectories:
            raise ValueError(
                "rollout shard trajectory count does not match its manifest: "
                f"path={current_path}, loaded={len(current)}, "
                f"manifest={manifest.num_trajectories}"
            )
        for trajectory in current:
            validate_rollout_trajectory(trajectory)
        manifests.append(manifest)
        shard_counts.append(len(current))
        trajectories.extend(current)

    reference_identity = _rollout_identity(manifests[0])
    for index, manifest in enumerate(manifests[1:], start=1):
        if _rollout_identity(manifest) != reference_identity:
            raise ValueError(
                "fresh rollout shards do not share one policy/planner/processor "
                f"identity: shard={index}"
            )

    actual_record_ids = tuple(item.record_id for item in trajectories)
    expected = tuple(str(record_id) for record_id in expected_record_ids)
    if actual_record_ids != expected:
        raise ValueError(
            "merged rollout record IDs do not match the requested global order: "
            f"actual={actual_record_ids}, expected={expected}"
        )
    if len(set(actual_record_ids)) != len(actual_record_ids):
        raise ValueError("merged rollout record IDs are not unique")
    if sum(shard_counts) != len(expected):
        raise ValueError(
            "merged rollout count does not match the requested batch: "
            f"shards={sum(shard_counts)}, expected={len(expected)}"
        )

    combined_path = save_trajectories(trajectories, target_dir)
    combined_fingerprint = file_artifact_fingerprint(combined_path)
    combined_manifest = replace(
        manifests[0],
        trajectory_path=str(combined_path.resolve()),
        trajectory_fingerprint=combined_fingerprint,
        num_trajectories=len(trajectories),
        created_at=datetime.now(timezone.utc).isoformat(),
        behavior_trajectory_path=str(combined_path.resolve()),
        behavior_trajectory_fingerprint=combined_fingerprint,
    )
    combined_manifest.write(manifest_path)
    combined_manifest.validate_trajectory_artifacts()
    return trajectories, combined_manifest


__all__ = ["merge_fresh_rollout_shards"]
