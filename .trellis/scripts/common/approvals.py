"""Hash-bound review packages and typed approval identities."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACT_NAMES = ("prd.md", "design.md", "implement.md")
APPROVAL_KINDS = {"planning", "implementation", "experiment_launch", "commit", "push_merge"}
DECISIONS = {"approve", "decline", "comment"}


def _iso(value: datetime | None = None) -> str:
    value = value or datetime.now(timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def root_fingerprint(root: Path) -> str:
    return hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()


def artifact_review_package(task_dir: Path, bound_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    issues: list[dict[str, Any]] = []
    for name in ARTIFACT_NAMES:
        path = task_dir / name
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            artifacts[name] = {"present": False, "raw": None, "sha256": None, "sections": []}
            continue
        except OSError as exc:
            artifacts[name] = {
                "present": True, "raw": None, "sha256": None, "sections": [],
                "error": "read_error",
            }
            issues.append({
                "code": "unreadable_planning_artifact", "severity": "error",
                "artifact": name, "message": f"Cannot read {name}: {exc}",
            })
            continue
        digest = hashlib.sha256(payload).hexdigest()
        hashes[name] = digest
        try:
            raw = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            artifacts[name] = {
                "present": True, "raw": None, "sha256": digest, "sections": [],
                "error": "invalid_utf8",
            }
            issues.append({
                "code": "unreadable_planning_artifact", "severity": "error",
                "artifact": name, "message": f"{name} is not valid UTF-8: {exc}",
            })
            continue
        sections = [
            {"level": len(match.group(1)), "title": match.group(2).strip(), "line": index}
            for index, line in enumerate(raw.splitlines(), start=1)
            if (match := re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line))
        ]
        artifacts[name] = {"present": True, "raw": raw, "sha256": digest, "sections": sections}
    review_set_hash = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    all_names = sorted(set(hashes) | set(bound_hashes or {}))
    changes = {
        name: (
            "added" if name in hashes and name not in (bound_hashes or {}) else
            "removed" if name not in hashes and name in (bound_hashes or {}) else
            "unchanged" if hashes.get(name) == (bound_hashes or {}).get(name) else
            "modified"
        )
        for name in all_names
    }
    result = {
        "artifacts": artifacts,
        "artifactHashes": hashes,
        "reviewSetHash": review_set_hash,
        "changesSinceRequest": {
            "bound": bound_hashes is not None,
            "changed": any(value != "unchanged" for value in changes.values()),
            "artifacts": changes,
        },
    }
    if issues:
        result["issues"] = issues
    return result


def _bounded_strings(name: str, values: list[str], *, maximum: int = 1000) -> list[str]:
    if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() and len(v) <= maximum for v in values):
        raise ValueError(f"{name} must be a list of non-empty strings up to {maximum} characters")
    return [v.strip() for v in values]


def _review_set_hash(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def create_approval_request(
    *, root: Path, context_key: str, session_id: str, request_id: str,
    task_ref: str, kind: str, artifact_hashes: dict[str, str], scope: list[str],
    exclusions: list[str], validation_commands: list[str], created_at: datetime | None = None,
) -> dict[str, Any]:
    if kind not in APPROVAL_KINDS:
        raise ValueError(f"unknown approval kind: {kind}")
    for name, value in (("context key", context_key), ("session id", session_id), ("request id", request_id), ("task ref", task_ref)):
        if not isinstance(value, str) or not value.strip() or len(value) > 512:
            raise ValueError(f"invalid {name}")
    if not isinstance(artifact_hashes, dict) or any(name not in ARTIFACT_NAMES or not re.fullmatch(r"[0-9a-f]{64}", digest) for name, digest in artifact_hashes.items()):
        raise ValueError("invalid artifact hashes")
    return {
        "requestId": request_id,
        "rootFingerprint": root_fingerprint(root),
        "contextKey": context_key,
        "sessionId": session_id,
        "taskRef": task_ref,
        "kind": kind,
        "artifactHashes": dict(sorted(artifact_hashes.items())),
        "reviewSetHash": _review_set_hash(artifact_hashes),
        "scope": _bounded_strings("scope", scope),
        "exclusions": _bounded_strings("exclusions", exclusions),
        "validationCommands": _bounded_strings("validation commands", validation_commands, maximum=2000),
        "createdAt": _iso(created_at),
        "status": "pending",
    }


def create_approval_receipt(
    request: dict[str, Any], *, decision: str, receipt_id: str,
    responded_at: datetime | None = None, comment: str | None = None,
) -> dict[str, Any]:
    if decision not in DECISIONS:
        raise ValueError(f"unknown approval decision: {decision}")
    if not receipt_id or len(receipt_id) > 512:
        raise ValueError("invalid receipt id")
    if comment is not None and len(comment) > 2000:
        raise ValueError("approval comment exceeds 2000 characters")
    identity = {key: request[key] for key in (
        "requestId", "rootFingerprint", "contextKey", "sessionId", "taskRef", "kind",
        "artifactHashes", "reviewSetHash", "scope", "exclusions", "validationCommands",
    )}
    return {
        "receiptId": receipt_id,
        **identity,
        "decision": decision,
        "comment": comment,
        "respondedAt": _iso(responded_at),
    }


def validate_approval_receipt(
    *, request: dict[str, Any], receipt: dict[str, Any], current_artifact_hashes: dict[str, str],
    root: Path, context_key: str, task_ref: str, kind: str,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    expected = {
        "rootFingerprint": root_fingerprint(root), "contextKey": context_key,
        "taskRef": task_ref, "kind": kind,
    }
    for key, value in expected.items():
        if request.get(key) != value or receipt.get(key) != value:
            issues.append({"code": "approval_identity_mismatch", "severity": "error", "field": key, "message": f"Approval {key} does not match"})
    for key in (
        "requestId", "sessionId", "artifactHashes", "reviewSetHash", "scope",
        "exclusions", "validationCommands",
    ):
        if receipt.get(key) != request.get(key):
            issues.append({"code": "approval_receipt_mismatch", "severity": "error", "field": key, "message": f"Receipt {key} does not match request"})
    current = dict(sorted(current_artifact_hashes.items()))
    if request.get("artifactHashes") != current or request.get("reviewSetHash") != _review_set_hash(current):
        issues.append({"code": "artifact_hash_mismatch", "severity": "error", "message": "Planning artifacts changed after the approval request"})
    if receipt.get("decision") != "approve":
        issues.append({"code": "approval_not_granted", "severity": "error", "message": "Receipt is not an approve decision"})
    return {"authorized": not issues, "issues": issues}
