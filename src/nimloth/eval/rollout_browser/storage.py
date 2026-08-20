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
    image_sources: Mapping[str, Path],
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
        source = Path(image_sources[name])
        if not source.is_file():
            raise FileNotFoundError(f"missing rollout image: {source}")
        destination = rollout_dir / name
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


def write_evaluation_browser(
    destination: Path,
    artifacts: Sequence[RolloutBrowserArtifact],
    *,
    evaluation: Mapping[str, Any],
    expected_rollouts: int,
) -> dict[str, Any]:
    """Publish one complete browser root with a single atomic directory rename."""

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"evaluation browser already exists: {destination}")
    if expected_rollouts < 1:
        raise ValueError("expected_rollouts must be positive")
    if len(artifacts) != expected_rollouts:
        raise ValueError(
            f"evaluation rollout count mismatch: {len(artifacts)} != {expected_rollouts}"
        )
    identities = [_identity_key(artifact.audit) for artifact in artifacts]
    if len(set(identities)) != len(identities):
        raise ValueError("evaluation browser contains duplicate rollout identities")
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        rollouts_root = temporary / "rollouts"
        rollouts_root.mkdir()
        manifest_rows: list[dict[str, Any]] = []
        success_count = 0
        reward_sum = 0.0
        source_counts: dict[str, int] = {}
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
            audit_sha256 = "sha256:" + hashlib.sha256(audit_payload).hexdigest()
            data_source = audit.get("data_source")
            source_key = str(data_source) if data_source is not None else "unknown"
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
            success_count += int(audit["success"])
            reward_sum += float(audit["reward"])
            manifest_rows.append(
                {
                    "identity": audit["identity"],
                    "data_source": data_source,
                    "seed": audit.get("seed"),
                    "task": audit["task"],
                    "success": audit["success"],
                    "reward": audit["reward"],
                    "turn_count": audit["turn_count"],
                    "stop_reason": audit["stop_reason"],
                    "capabilities": audit["capabilities"],
                    "artifact": f"rollouts/{safe_id}/index.html",
                    "audit_sha256": audit_sha256,
                }
            )
        manifest = {
            "schema": EVALUATION_MANIFEST_SCHEMA,
            "status": "complete",
            **dict(evaluation),
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
        _write_fsynced(temporary / "manifest.json", manifest_payload)
        _write_fsynced(
            temporary / "index.html",
            render_evaluation_index(manifest).encode("utf-8"),
        )
        complete = {
            "schema": "nimloth_evaluation_rollout_browser_complete_v1",
            "evaluation_id": evaluation["evaluation_id"],
            "rollout_count": len(manifest_rows),
            "manifest_sha256": "sha256:"
            + hashlib.sha256(manifest_payload).hexdigest(),
        }
        _write_fsynced(temporary / "complete.json", _json_bytes(complete))
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


__all__ = ["write_evaluation_browser"]
