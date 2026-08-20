"""Atomic, immutable storage for complete evaluation rollout browsers."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .render import render_evaluation_index, render_rollout_html
from .schema import EVALUATION_MANIFEST_SCHEMA, validate_rollout_audit
from .sft_adapter import RolloutBrowserArtifact

BATCH_MARKER_SCHEMA = "nimloth_evaluation_rollout_browser_batch_v1"
COMPLETE_MARKER_SCHEMA = "nimloth_evaluation_rollout_browser_complete_v1"


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _identity_key(audit: Mapping[str, Any]) -> tuple[str, int]:
    identity = audit["identity"]
    primary = identity.get("rollout_sample_id") or identity.get("record_id")
    return str(primary), int(identity["rollout_repeat_index"])


def _safe_identity(identity: tuple[str, int]) -> str:
    payload = json.dumps(identity, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _install_images(
    rollout_dir: Path,
    audit: dict[str, Any],
    image_sources: Mapping[str, Any],
) -> None:
    expected_names = {
        turn["observation"]["image"] for turn in audit["turns"]
    }
    expected_names.update(
        turn["terminal"]["observation"]["image"]
        for turn in audit["turns"]
        if "terminal" in turn
    )
    if expected_names != set(image_sources):
        raise ValueError("rollout image source set does not match audit")
    hashes: dict[str, str] = {}
    for name in sorted(expected_names):
        if Path(name).name != name or not name.endswith(".png"):
            raise ValueError(f"rollout image name is unsafe: {name}")
        raw_source = image_sources[name]
        destination = rollout_dir / name
        if hasattr(raw_source, "save") and not isinstance(raw_source, (str, Path)):
            raw_source.save(destination, format="PNG")
            with destination.open("rb") as handle:
                os.fsync(handle.fileno())
        else:
            source = Path(raw_source)
            if not source.is_file():
                raise FileNotFoundError(f"missing rollout image: {source}")
            with source.open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst)
                dst.flush()
                os.fsync(dst.fileno())
        hashes[name] = _sha256(destination)
    for turn in audit["turns"]:
        turn["observation"]["sha256"] = hashes[turn["observation"]["image"]]
        if "terminal" in turn:
            observation = turn["terminal"]["observation"]
            observation["sha256"] = hashes[observation["image"]]


def write_evaluation_browser_batch(
    root: Path,
    artifacts: Sequence[RolloutBrowserArtifact],
    *,
    batch_index: int,
) -> dict[str, Any]:
    """Atomically commit one complete validation batch below an incomplete root."""

    root = Path(root)
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
        raise ValueError("browser batch_index must be non-negative")
    if not artifacts:
        raise ValueError("browser batch must contain at least one rollout")
    if (root / "complete.json").exists():
        raise FileExistsError("evaluation browser is already complete")
    identities = [_identity_key(artifact.audit) for artifact in artifacts]
    if len(set(identities)) != len(identities):
        raise ValueError("browser batch contains duplicate rollout identities")
    batches_root = root / "batches"
    batches_root.mkdir(parents=True, exist_ok=True)
    destination = batches_root / f"batch_{batch_index:04d}"
    if destination.exists():
        raise FileExistsError(f"evaluation browser batch already exists: {destination}")
    temporary = batches_root / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        rollouts_root = temporary / "rollouts"
        rollouts_root.mkdir()
        rows: list[dict[str, Any]] = []
        for artifact, identity in zip(artifacts, identities, strict=True):
            audit = copy.deepcopy(artifact.audit)
            safe_id = _safe_identity(identity)
            rollout_dir = rollouts_root / safe_id
            rollout_dir.mkdir()
            _install_images(rollout_dir, audit, artifact.image_sources)
            validate_rollout_audit(audit)
            audit_payload = _json_bytes(audit)
            _write_fsynced(rollout_dir / "rollout.json", audit_payload)
            _write_fsynced(
                rollout_dir / "index.html",
                render_rollout_html(audit).encode("utf-8"),
            )
            rows.append(
                {
                    "identity": audit["identity"],
                    "audit": f"rollouts/{safe_id}/rollout.json",
                    "artifact": f"rollouts/{safe_id}/index.html",
                    "audit_sha256": "sha256:"
                    + hashlib.sha256(audit_payload).hexdigest(),
                }
            )
        marker = {
            "schema": BATCH_MARKER_SCHEMA,
            "batch_index": batch_index,
            "rollout_count": len(rows),
            "rollouts": rows,
        }
        _write_fsynced(temporary / "complete.json", _json_bytes(marker))
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temporary, destination)
        parent_fd = os.open(batches_root, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return marker
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _manifest_row(audit: Mapping[str, Any], artifact: str, digest: str) -> dict[str, Any]:
    return {
        "identity": audit["identity"],
        "data_source": audit.get("data_source"),
        "seed": audit.get("seed"),
        "task": audit["task"],
        "success": audit["success"],
        "reward": audit["reward"],
        "turn_count": audit["turn_count"],
        "stop_reason": audit["stop_reason"],
        "capabilities": audit["capabilities"],
        "artifact": artifact,
        "audit_sha256": digest,
    }


def finalize_evaluation_browser(
    root: Path,
    *,
    evaluation: Mapping[str, Any],
    expected_rollouts: int,
    expected_batches: int,
) -> dict[str, Any]:
    """Verify every immutable batch before publishing the searchable index."""

    root = Path(root)
    if (root / "complete.json").exists():
        raise FileExistsError("evaluation browser is already complete")
    if expected_rollouts < 1 or expected_batches < 1:
        raise ValueError("expected browser counts must be positive")
    required_evaluation = {
        "evaluation_id",
        "policy_family",
        "global_step",
        "source_step",
        "checkpoint_identity",
        "snapshot_identity",
    }
    missing = required_evaluation - set(evaluation)
    if missing:
        raise ValueError(f"evaluation metadata is missing fields: {sorted(missing)}")
    batch_dirs = sorted((root / "batches").glob("batch_*"))
    if len(batch_dirs) != expected_batches:
        raise ValueError(
            f"evaluation browser batch count mismatch: {len(batch_dirs)} != {expected_batches}"
        )
    identities: list[tuple[str, int]] = []
    manifest_rows: list[dict[str, Any]] = []
    success_count = 0
    reward_sum = 0.0
    source_counts: dict[str, int] = {}
    snapshot_ids: set[str] = set()
    source_steps: set[int] = set()
    for expected_index, batch_dir in enumerate(batch_dirs):
        marker = json.loads((batch_dir / "complete.json").read_text())
        if marker.get("schema") != BATCH_MARKER_SCHEMA or marker.get("batch_index") != expected_index:
            raise ValueError("evaluation browser batch marker mismatch")
        if marker.get("rollout_count") != len(marker.get("rollouts", [])):
            raise ValueError("evaluation browser batch rollout count mismatch")
        for row in marker["rollouts"]:
            audit_path = batch_dir / row["audit"]
            audit_payload = audit_path.read_bytes()
            digest = "sha256:" + hashlib.sha256(audit_payload).hexdigest()
            if digest != row["audit_sha256"]:
                raise ValueError("evaluation browser audit hash mismatch")
            audit = json.loads(audit_payload)
            validate_rollout_audit(audit)
            identity = _identity_key(audit)
            if audit["identity"] != row["identity"]:
                raise ValueError("evaluation browser audit identity mismatch")
            identities.append(identity)
            provenance = audit.get("provenance", {})
            if provenance.get("snapshot_id") is not None:
                snapshot_ids.add(str(provenance["snapshot_id"]))
            if provenance.get("source_step") is not None:
                source_steps.add(int(provenance["source_step"]))
            artifact = f"batches/{batch_dir.name}/{row['artifact']}"
            manifest_rows.append(_manifest_row(audit, artifact, digest))
            source_key = str(audit.get("data_source") or "unknown")
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
            success_count += int(audit["success"])
            reward_sum += float(audit["reward"])
    if len(manifest_rows) != expected_rollouts:
        raise ValueError(
            f"evaluation rollout count mismatch: {len(manifest_rows)} != {expected_rollouts}"
        )
    if len(set(identities)) != len(identities):
        raise ValueError("evaluation browser contains duplicate rollout identities")
    if len(snapshot_ids) > 1 or len(source_steps) > 1:
        raise ValueError("evaluation browser contains mixed snapshot provenance")
    resolved_evaluation = dict(evaluation)
    derived_snapshot = next(iter(snapshot_ids)) if snapshot_ids else None
    derived_source_step = next(iter(source_steps)) if source_steps else None
    if resolved_evaluation["snapshot_identity"] is None:
        resolved_evaluation["snapshot_identity"] = derived_snapshot
    elif derived_snapshot is not None and str(resolved_evaluation["snapshot_identity"]) != derived_snapshot:
        raise ValueError("evaluation browser snapshot identity mismatch")
    if resolved_evaluation["source_step"] is None:
        resolved_evaluation["source_step"] = derived_source_step
    elif derived_source_step is not None and int(resolved_evaluation["source_step"]) != derived_source_step:
        raise ValueError("evaluation browser source step mismatch")
    manifest = {
        "schema": EVALUATION_MANIFEST_SCHEMA,
        "status": "complete",
        **resolved_evaluation,
        "expected_rollouts": expected_rollouts,
        "rollout_count": len(manifest_rows),
        "summary": {
            "success_count": success_count,
            "reward_mean": reward_sum / len(manifest_rows),
            "data_source_counts": dict(sorted(source_counts.items())),
        },
        "rollouts": manifest_rows,
    }
    manifest_payload = _json_bytes(manifest)
    _write_fsynced(root / "manifest.json", manifest_payload)
    _write_fsynced(root / "index.html", render_evaluation_index(manifest).encode("utf-8"))
    complete = {
        "schema": COMPLETE_MARKER_SCHEMA,
        "evaluation_id": evaluation["evaluation_id"],
        "batch_count": expected_batches,
        "rollout_count": len(manifest_rows),
        "manifest_sha256": "sha256:" + hashlib.sha256(manifest_payload).hexdigest(),
    }
    _write_fsynced(root / "complete.json", _json_bytes(complete))
    return manifest


def write_evaluation_browser(
    destination: Path,
    artifacts: Sequence[RolloutBrowserArtifact],
    *,
    evaluation: Mapping[str, Any],
    expected_rollouts: int,
) -> dict[str, Any]:
    """Publish a complete single-batch browser with one atomic root rename."""

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"evaluation browser already exists: {destination}")
    if len(artifacts) != expected_rollouts:
        raise ValueError(
            f"evaluation rollout count mismatch: {len(artifacts)} != {expected_rollouts}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        write_evaluation_browser_batch(temporary, artifacts, batch_index=0)
        manifest = finalize_evaluation_browser(
            temporary,
            evaluation=evaluation,
            expected_rollouts=expected_rollouts,
            expected_batches=1,
        )
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temporary, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


__all__ = [
    "finalize_evaluation_browser",
    "write_evaluation_browser",
    "write_evaluation_browser_batch",
]
