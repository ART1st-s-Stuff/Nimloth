#!/usr/bin/env python3
"""Capture and validate a metadata-only Git worktree migration manifest.

This task-local tool intentionally uses only the Python standard library.  It
never reads file contents into the output: filesystem payloads are represented
by path, kind, size, and SHA-256 only.  Git config values and remote endpoints
are fingerprinted rather than copied, so credentials cannot enter task
artifacts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "nimloth-worktree-migration-manifest/v1"
DEFAULT_CANONICAL_ROOT = "/workspace/remote2/nimloth"
DEFAULT_DEV_WORKTREE = "/workspace/remote2/nimloth-dev"
HASH_LIMIT_BYTES = 64 * 1024 * 1024
HEX_OBJECT_RE = re.compile(r"^[0-9a-f]{40,64}$")


class CaptureError(RuntimeError):
    """Raised when required Git metadata cannot be captured."""


def _run(
    repo: str,
    *args: str,
    timeout: int = 30,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    command = ["git", "-C", repo, *args]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except subprocess.TimeoutExpired as exc:
        raise CaptureError(f"timeout after {timeout}s: {' '.join(command)}") from exc
    if check and completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace").strip()
        raise CaptureError(f"failed ({completed.returncode}): {' '.join(command)}: {stderr}")
    return completed


def _text(repo: str, *args: str, timeout: int = 30, check: bool = True) -> str:
    return _run(repo, *args, timeout=timeout, check=check).stdout.decode(
        "utf-8", "surrogateescape"
    ).strip()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[str | None, str]:
    size = path.stat().st_size
    if size > HASH_LIMIT_BYTES:
        return None, f"skipped-size-limit:{HASH_LIMIT_BYTES}"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), "complete"


def _path_metadata(root: str, relative: str) -> dict[str, Any]:
    path = Path(root) / relative
    result: dict[str, Any] = {
        "path": relative,
        "exists": os.path.lexists(path),
        "kind": "missing",
        "size_bytes": None,
        "sha256": None,
        "hash_status": "not-applicable",
    }
    if not result["exists"]:
        return result

    metadata = path.lstat()
    result["size_bytes"] = metadata.st_size
    mode = metadata.st_mode
    if stat.S_ISLNK(mode):
        target = os.readlink(path)
        result.update(
            kind="symlink",
            sha256=_sha256_bytes(os.fsencode(target)),
            hash_status="symlink-target",
        )
    elif stat.S_ISREG(mode):
        digest, hash_status = _sha256_file(path)
        result.update(kind="file", sha256=digest, hash_status=hash_status)
    elif stat.S_ISDIR(mode):
        result.update(kind="directory", hash_status="not-recursively-scanned")
    else:
        result.update(kind="other", hash_status="unsupported-file-kind")
    return result


def _parse_worktree_porcelain(repo: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in _text(repo, "worktree", "list", "--porcelain").splitlines() + [""]:
        if not line:
            if current:
                records.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key in {"bare", "detached", "locked", "prunable"}:
            current[key] = value or True
        else:
            current[key] = value
    return records


def _parse_status(
    repo: str, *, bind_work_tree: bool = False
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    prefix = (f"--work-tree={repo}",) if bind_work_tree else ()
    payload = _run(
        repo,
        *prefix,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        timeout=45,
    ).stdout
    fields = payload.split(b"\0")
    entries: list[dict[str, Any]] = []
    index = 0
    while index < len(fields):
        raw = fields[index]
        index += 1
        if not raw:
            continue
        decoded = raw.decode("utf-8", "surrogateescape")
        if len(decoded) < 4 or decoded[2] != " ":
            raise CaptureError(f"unrecognized porcelain status record in {repo!r}: {decoded!r}")
        code = decoded[:2]
        relative = decoded[3:]
        entry = {"status": code, **_path_metadata(repo, relative)}
        if "R" in code or "C" in code:
            if index >= len(fields) or not fields[index]:
                raise CaptureError(f"missing rename source in status for {repo!r}")
            entry["source_path"] = fields[index].decode("utf-8", "surrogateescape")
            index += 1
        entries.append(entry)

    counts = {
        "total": len(entries),
        "tracked": sum(entry["status"] != "??" for entry in entries),
        "untracked": sum(entry["status"] == "??" for entry in entries),
        "hash_incomplete": sum(entry["hash_status"].startswith("skipped-") for entry in entries),
    }
    return entries, counts


def _ignored_summary(repo: str, *, bind_work_tree: bool = False) -> dict[str, Any]:
    # --directory collapses wholly ignored trees.  This is deliberately a
    # bounded cleanup-risk overview, not an expensive recursive payload scan.
    prefix = (f"--work-tree={repo}",) if bind_work_tree else ()
    try:
        payload = _run(
            repo,
            *prefix,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--directory",
            "--no-empty-directory",
            "-z",
            timeout=20,
        ).stdout
    except CaptureError as exc:
        return {
            "scan_status": "blocked",
            "scan_mode": "top-level-collapsed",
            "entries": [],
            "entry_count": None,
            "blocking_reason": str(exc),
        }
    paths = [
        field.decode("utf-8", "surrogateescape").rstrip("/")
        for field in payload.split(b"\0")
        if field
    ]
    entries = [_path_metadata(repo, path) for path in paths]
    return {
        "scan_status": "complete",
        "scan_mode": "top-level-collapsed",
        "entries": entries,
        "entry_count": len(entries),
        "blocking_reason": None,
        "cleanup_gate": "recursively inspect and approve every listed tree before removal",
    }


def _submodule_worktree(repo: str, relative: str, state: str) -> dict[str, Any]:
    path = str(Path(repo) / relative)
    git_marker = Path(path) / ".git"
    if not os.path.lexists(git_marker):
        return {
            "scan_status": "not-initialized" if state == "-" else "blocked",
            "actual_head": None,
            "branch": None,
            "status": None,
            "ignored": None,
            "blocking_reason": "submodule worktree has no .git marker at the recorded path",
        }
    # Submodule Git dirs are shared across superproject worktrees and may carry
    # a core.worktree for another checkout.  Bind --work-tree explicitly so a
    # read-only status cannot silently inspect that sibling instead.
    prefix = (f"--work-tree={path}",)
    probe = _run(path, *prefix, "rev-parse", "--show-toplevel", check=False)
    probe_top = probe.stdout.decode("utf-8", "surrogateescape").strip()
    if probe.returncode != 0 or str(Path(probe_top).resolve()) != str(Path(path).resolve()):
        return {
            "scan_status": "blocked",
            "actual_head": None,
            "branch": None,
            "status": None,
            "ignored": None,
            "blocking_reason": probe.stderr.decode("utf-8", "replace").strip()
            or "submodule worktree could not be bound to the recorded path",
        }
    entries, summary = _parse_status(path, bind_work_tree=True)
    return {
        "scan_status": "complete",
        "actual_head": _text(path, *prefix, "rev-parse", "HEAD"),
        "branch": _text(path, *prefix, "branch", "--show-current") or None,
        "status": {"summary": summary, "entries": entries},
        "ignored": _ignored_summary(path, bind_work_tree=True),
        "blocking_reason": None,
    }


def _submodules(repo: str) -> dict[str, Any]:
    completed = _run(repo, "submodule", "status", "--recursive", timeout=45, check=False)
    if completed.returncode != 0:
        return {
            "scan_status": "blocked",
            "entries": [],
            "entry_count": 0,
            "working_tree_blocked": [],
            "blocking_reason": completed.stderr.decode("utf-8", "replace").strip(),
        }
    entries: list[dict[str, Any]] = []
    for line in completed.stdout.decode("utf-8", "surrogateescape").splitlines():
        if not line:
            continue
        state = line[0]
        body = line[1:]
        object_name, _, tail = body.partition(" ")
        path, _, description = tail.partition(" ")
        rendered_description = description.encode("utf-8", "surrogateescape")
        entries.append(
            {
                "path": path,
                "state": state,
                "head": object_name,
                "description_sha256": _sha256_bytes(rendered_description),
                "description_size_bytes": len(rendered_description),
                "working_tree": _submodule_worktree(repo, path, state),
            }
        )
    blocked = [
        entry["path"]
        for entry in entries
        if entry["working_tree"]["scan_status"] == "blocked"
        or (
            entry["working_tree"]["scan_status"] == "complete"
            and entry["working_tree"]["ignored"]["scan_status"] != "complete"
        )
    ]
    return {
        "scan_status": "complete",
        "entries": entries,
        "entry_count": len(entries),
        "working_tree_blocked": blocked,
        "blocking_reason": None,
    }


def _local_link(repo: str) -> dict[str, Any]:
    local = Path(repo) / ".local"
    result = _path_metadata(repo, ".local")
    if local.is_symlink():
        # Machine paths are allowed evidence; do not dereference or read state.
        result["target_path"] = os.readlink(local)
        result["resolved_path"] = str(local.resolve(strict=False))
    elif local.is_dir():
        result["resolved_path"] = str(local.resolve(strict=False))
    result["recovery_method"] = "restore directory or recreate symlink to canonical root .local after exact-path approval"
    return result


def _upstream(repo: str) -> dict[str, Any]:
    completed = _run(
        repo,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        check=False,
    )
    if completed.returncode != 0:
        return {"name": None, "head": None, "ahead": None, "behind": None}
    symbolic = completed.stdout.decode("utf-8", "surrogateescape").strip()
    if not symbolic:
        return {"name": None, "head": None, "ahead": None, "behind": None}
    head = _text(repo, "rev-parse", "@{upstream}")
    counts = _text(repo, "rev-list", "--left-right", "--count", "HEAD...@{upstream}").split()
    ahead = int(counts[0]) if len(counts) == 2 else None
    behind = int(counts[1]) if len(counts) == 2 else None
    return {"name": symbolic, "head": head, "ahead": ahead, "behind": behind}


def _capture_worktree(registered: dict[str, Any]) -> dict[str, Any]:
    path = registered["worktree"]
    status_entries, status_summary = _parse_status(path)
    branch_ref = registered.get("branch")
    return {
        "path": path,
        "registration": {
            "head": registered.get("HEAD"),
            "branch_ref": branch_ref,
            "detached": bool(registered.get("detached")),
            "locked": registered.get("locked", False),
            "prunable": registered.get("prunable", False),
        },
        "actual": {
            "top_level": _text(path, "rev-parse", "--show-toplevel"),
            "git_dir": _text(path, "rev-parse", "--path-format=absolute", "--git-dir"),
            "common_dir": _text(path, "rev-parse", "--path-format=absolute", "--git-common-dir"),
            "branch": _text(path, "branch", "--show-current") or None,
            "head": _text(path, "rev-parse", "HEAD"),
            "upstream": _upstream(path),
        },
        "status": {"summary": status_summary, "entries": status_entries},
        "submodules": _submodules(path),
        "ignored": _ignored_summary(path),
        "local_state": _local_link(path),
        "recovery_method": "recreate from recorded branch/HEAD, then restore separately approved metadata-only indexed payload",
    }


def _capture_refs(repo: str) -> list[dict[str, Any]]:
    separator = "%00"
    output = _text(
        repo,
        "for-each-ref",
        f"--format=%(refname){separator}%(objectname){separator}%(objecttype){separator}%(upstream)",
    )
    entries: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = line.split("\0")
        if len(fields) != 4:
            raise CaptureError(f"unexpected for-each-ref record: {line!r}")
        entries.append(
            {"name": fields[0], "head": fields[1], "object_type": fields[2], "upstream": fields[3] or None}
        )
    return entries


def _fingerprints(values: Iterable[bytes]) -> list[dict[str, Any]]:
    return [
        {"sha256": _sha256_bytes(value), "size_bytes": len(value), "recovery_method": "read from live Git config"}
        for value in values
    ]


def _capture_remotes(repo: str) -> list[dict[str, Any]]:
    remotes: list[dict[str, Any]] = []
    for name in filter(None, _text(repo, "remote").splitlines()):
        fetch = _run(repo, "remote", "get-url", "--all", name, check=False).stdout.splitlines()
        push = _run(repo, "remote", "get-url", "--push", "--all", name, check=False).stdout.splitlines()
        remotes.append(
            {
                "name": name,
                "fetch_endpoint_fingerprints": _fingerprints(fetch),
                "push_endpoint_fingerprints": _fingerprints(push),
                "recovery_method": f"git remote get-url --all {name} and --push --all from the live common repository",
            }
        )
    return remotes


def _capture_config(repo: str) -> list[dict[str, Any]]:
    payload = _run(repo, "config", "--local", "--null", "--list").stdout
    entries: list[dict[str, Any]] = []
    for record in filter(None, payload.split(b"\0")):
        key, separator, value = record.partition(b"\n")
        if not separator:
            key, separator, value = record.partition(b"=")
        decoded_key = key.decode("utf-8", "surrogateescape")
        entries.append(
            {
                "key": decoded_key,
                "value_sha256": _sha256_bytes(value),
                "value_size_bytes": len(value),
                "recovery_method": f"git config --local --get-all {decoded_key}",
            }
        )
    return entries


def _tree_object(repo: str, revision: str, path: str) -> dict[str, Any]:
    completed = _run(repo, "ls-tree", revision, "--", path, check=False)
    if completed.returncode != 0 or not completed.stdout:
        return {"object": None, "kind": None}
    first = completed.stdout.decode("utf-8", "surrogateescape").splitlines()[0]
    metadata, _, _name = first.partition("\t")
    mode, kind, object_name = metadata.split()
    return {"object": object_name, "kind": kind, "mode": mode}


def _canonical_conflict(
    worktrees: list[dict[str, Any]], canonical_root: str, dev_worktree: str
) -> dict[str, Any]:
    by_path = {entry["path"]: entry for entry in worktrees}
    canonical = by_path.get(canonical_root)
    dev = by_path.get(dev_worktree)
    if canonical is None or dev is None:
        return {
            "status": "blocked",
            "reason": "canonical root or linked dev worktree is not registered",
            "canonical_root": canonical_root,
            "dev_worktree": dev_worktree,
        }
    canonical_paths = {entry["path"] for entry in canonical["status"]["entries"]}
    dev_paths = {entry["path"] for entry in dev["status"]["entries"]}
    canonical_comparison = []
    for path in sorted(canonical_paths):
        canonical_comparison.append(
            {
                "path": path,
                "main_head": _tree_object(canonical_root, "HEAD", path),
                "dev_head": _tree_object(canonical_root, "dev", path),
            }
        )
    return {
        "status": "conflict-preservation-required",
        "canonical_root": canonical_root,
        "canonical_branch": canonical["actual"]["branch"],
        "canonical_head": canonical["actual"]["head"],
        "canonical_dirty_count": canonical["status"]["summary"]["total"],
        "dev_worktree": dev_worktree,
        "dev_branch": dev["actual"]["branch"],
        "dev_head": dev["actual"]["head"],
        "dev_dirty_count": dev["status"]["summary"]["total"],
        "dirty_path_overlap": sorted(canonical_paths & dev_paths),
        "canonical_dirty_tree_comparison": canonical_comparison,
        "recovery_method": "preserve and verify canonical-main and linked-dev payloads independently before any branch cutover",
    }


def capture(repo: str, canonical_root: str, dev_worktree: str) -> dict[str, Any]:
    repo = str(Path(repo).resolve())
    registered = _parse_worktree_porcelain(repo)
    worktrees = [_capture_worktree(entry) for entry in registered]
    refs = _capture_refs(repo)
    remotes = _capture_remotes(repo)
    config = _capture_config(repo)
    blocked = [
        f"{entry['path']}: {category} scan blocked"
        for entry in worktrees
        for category in ("submodules", "ignored")
        if entry[category]["scan_status"] != "complete"
    ]
    blocked.extend(
        f"{entry['path']}: submodule working-tree scan blocked: {submodule_path}"
        for entry in worktrees
        for submodule_path in entry["submodules"].get("working_tree_blocked", [])
    )
    incomplete_hashes = [
        f"{entry['path']}: {item['path']} ({item['hash_status']})"
        for entry in worktrees
        for item in entry["status"]["entries"]
        if item["hash_status"].startswith("skipped-")
    ]
    incomplete_hashes.extend(
        f"{entry['path']}::{submodule['path']}: {item['path']} ({item['hash_status']})"
        for entry in worktrees
        for submodule in entry["submodules"]["entries"]
        if submodule["working_tree"]["scan_status"] == "complete"
        for item in submodule["working_tree"]["status"]["entries"]
        if item["hash_status"].startswith("skipped-")
    )
    return {
        "schema": SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "capture_root": repo,
        "policy": {
            "canonical_root": canonical_root,
            "daily_branch": "dev",
            "dev_worktree_before_cutover": dev_worktree,
            "payload_boundary": "metadata-only; no file contents, config values, remote endpoints, or protected memory contents",
            "ignored_scan_mode": "top-level-collapsed in each superproject and initialized submodule; recursive expansion remains an exact-path cleanup gate",
            "no_force_fallback": True,
            "manual_git_worktrees_edit": False,
        },
        "summary": {
            "registered_worktrees": len(worktrees),
            "dirty_worktrees": sum(item["status"]["summary"]["total"] > 0 for item in worktrees),
            "clean_worktrees": sum(item["status"]["summary"]["total"] == 0 for item in worktrees),
            "detached_worktrees": sum(item["registration"]["detached"] for item in worktrees),
            "blocked_scans": blocked,
            "incomplete_payload_hashes": incomplete_hashes,
        },
        "worktrees": worktrees,
        "refs": refs,
        "tags": [entry for entry in refs if entry["name"].startswith("refs/tags/")],
        "remotes": remotes,
        "config": config,
        "canonical_main_dev_conflict": _canonical_conflict(worktrees, canonical_root, dev_worktree),
    }


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _current_registration(repo: str) -> dict[str, tuple[str | None, str | None]]:
    result = {}
    for entry in _parse_worktree_porcelain(repo):
        result[entry["worktree"]] = (entry.get("HEAD"), entry.get("branch"))
    return result


def _validate_metadata(item: Any, context: str, errors: list[str]) -> None:
    if not isinstance(item, dict):
        errors.append(f"{context} metadata must be an object")
        return
    if not isinstance(item.get("path"), str):
        errors.append(f"{context} path must be a string")
    kind = item.get("kind")
    if kind not in {"missing", "symlink", "file", "directory", "other"}:
        errors.append(f"{context} has invalid kind: {kind!r}")
    size = item.get("size_bytes")
    if size is not None and (not isinstance(size, int) or size < 0):
        errors.append(f"{context} has invalid size")
    digest = item.get("sha256")
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        errors.append(f"{context} has invalid SHA-256")
    if kind in {"file", "symlink"} and item.get("hash_status") in {"complete", "symlink-target"} and digest is None:
        errors.append(f"{context} is missing a completed SHA-256")


def _validate_status(status: Any, context: str, errors: list[str]) -> None:
    if not isinstance(status, dict) or not isinstance(status.get("entries"), list):
        errors.append(f"{context} status must contain entries")
        return
    entries = status["entries"]
    for index, item in enumerate(entries):
        if not isinstance(item, dict) or not isinstance(item.get("status"), str) or len(item["status"]) != 2:
            errors.append(f"{context} status entry {index} has invalid porcelain code")
        _validate_metadata(item, f"{context} status entry {index}", errors)
    summary = status.get("summary", {})
    expected = {
        "total": len(entries),
        "tracked": sum(item.get("status") != "??" for item in entries if isinstance(item, dict)),
        "untracked": sum(item.get("status") == "??" for item in entries if isinstance(item, dict)),
        "hash_incomplete": sum(
            str(item.get("hash_status", "")).startswith("skipped-")
            for item in entries
            if isinstance(item, dict)
        ),
    }
    if summary != expected:
        errors.append(f"{context} status summary does not match entries")


def _validate_ignored(ignored: Any, context: str, errors: list[str], warnings: list[str]) -> None:
    if not isinstance(ignored, dict):
        errors.append(f"{context} ignored summary must be an object")
        return
    if ignored.get("scan_status") != "complete":
        warnings.append(f"ignored scan blocked: {context}")
        return
    entries = ignored.get("entries")
    if not isinstance(entries, list) or ignored.get("entry_count") != len(entries):
        errors.append(f"{context} ignored entry_count does not match entries")
        return
    for index, item in enumerate(entries):
        _validate_metadata(item, f"{context} ignored entry {index}", errors)


def validate(manifest: dict[str, Any], repo: str | None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if manifest.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    worktrees = manifest.get("worktrees")
    if not isinstance(worktrees, list) or not worktrees:
        errors.append("worktrees must be a non-empty list")
        worktrees = []
    if any(not isinstance(entry, dict) for entry in worktrees):
        errors.append("every worktree entry must be an object")
        worktrees = [entry for entry in worktrees if isinstance(entry, dict)]
    paths = [entry.get("path") for entry in worktrees]
    if len(paths) != len(set(paths)):
        errors.append("worktree paths are not unique")
    if any(not isinstance(path, str) or not os.path.isabs(path) for path in paths):
        errors.append("every worktree path must be absolute")

    incomplete_count = 0
    blocked_count = 0
    for entry in worktrees:
        path = entry.get("path")
        registration = entry.get("registration", {})
        actual = entry.get("actual", {})
        head = actual.get("head")
        if not isinstance(head, str) or not HEX_OBJECT_RE.fullmatch(head):
            errors.append(f"invalid HEAD for {path!r}: {head!r}")
        if registration.get("head") != head:
            errors.append(f"registration and actual HEAD differ: {path}")
        branch_ref = registration.get("branch_ref")
        expected_branch = branch_ref.removeprefix("refs/heads/") if isinstance(branch_ref, str) else None
        if expected_branch != actual.get("branch") and not registration.get("detached"):
            errors.append(f"registration and actual branch differ: {path}")
        if actual.get("top_level") != path:
            errors.append(f"actual top-level differs from recorded path: {path}")
        _validate_status(entry.get("status"), str(path), errors)
        incomplete_count += entry.get("status", {}).get("summary", {}).get("hash_incomplete", 0)
        _validate_ignored(entry.get("ignored"), str(path), errors, warnings)
        if entry.get("ignored", {}).get("scan_status") != "complete":
            blocked_count += 1

        submodules = entry.get("submodules", {})
        submodule_entries = submodules.get("entries")
        if submodules.get("scan_status") != "complete" or not isinstance(submodule_entries, list):
            warnings.append(f"submodule scan blocked: {path}")
            blocked_count += 1
            continue
        if submodules.get("entry_count") != len(submodule_entries):
            errors.append(f"submodule entry_count does not match entries: {path}")
        submodule_paths = [item.get("path") for item in submodule_entries if isinstance(item, dict)]
        if len(submodule_paths) != len(set(submodule_paths)):
            errors.append(f"submodule paths are not unique: {path}")
        calculated_blocked: list[str] = []
        for submodule in submodule_entries:
            submodule_path = submodule.get("path")
            submodule_head = submodule.get("head")
            context = f"{path}::{submodule_path}"
            if not isinstance(submodule_head, str) or not HEX_OBJECT_RE.fullmatch(submodule_head):
                errors.append(f"invalid submodule HEAD: {context}")
            working_tree = submodule.get("working_tree", {})
            scan_status = working_tree.get("scan_status")
            if scan_status == "complete":
                _validate_status(working_tree.get("status"), context, errors)
                incomplete_count += working_tree.get("status", {}).get("summary", {}).get(
                    "hash_incomplete", 0
                )
                _validate_ignored(working_tree.get("ignored"), context, errors, warnings)
                if working_tree.get("ignored", {}).get("scan_status") != "complete":
                    calculated_blocked.append(submodule_path)
            elif scan_status == "not-initialized" and submodule.get("state") == "-":
                continue
            else:
                calculated_blocked.append(submodule_path)
                warnings.append(f"submodule working-tree scan blocked: {context}")
        if submodules.get("working_tree_blocked") != calculated_blocked:
            errors.append(f"submodule working_tree_blocked does not match entries: {path}")
        blocked_count += len(calculated_blocked)

    summary = manifest.get("summary", {})
    dirty = sum(entry.get("status", {}).get("summary", {}).get("total", 0) > 0 for entry in worktrees)
    expected_summary_counts = {
        "registered_worktrees": len(worktrees),
        "dirty_worktrees": dirty,
        "clean_worktrees": len(worktrees) - dirty,
        "detached_worktrees": sum(bool(entry.get("registration", {}).get("detached")) for entry in worktrees),
    }
    for key, expected in expected_summary_counts.items():
        if summary.get(key) != expected:
            errors.append(f"summary.{key} does not match worktree records")
    if len(summary.get("blocked_scans", [])) != blocked_count:
        errors.append("summary.blocked_scans does not match blocked worktree/submodule scans")
    if len(summary.get("incomplete_payload_hashes", [])) != incomplete_count:
        errors.append("summary.incomplete_payload_hashes does not match status metadata")

    forbidden_exact_keys = {"content", "contents", "value", "values", "url", "urls"}
    leaked_keys = sorted(set(_walk_keys(manifest)) & forbidden_exact_keys)
    if leaked_keys:
        errors.append(f"manifest contains forbidden secret-bearing keys: {leaked_keys}")
    policy = manifest.get("policy", {})
    canonical_root = policy.get("canonical_root")
    dev_worktree = policy.get("dev_worktree_before_cutover")
    if canonical_root not in paths:
        errors.append("canonical root is absent from registered worktrees")
    if dev_worktree not in paths:
        errors.append("pre-cutover dev worktree is absent from registered worktrees")
    if policy.get("daily_branch") != "dev":
        errors.append("manifest daily branch must be dev")
    if not policy.get("no_force_fallback"):
        errors.append("manifest policy must prohibit force fallback")
    if policy.get("manual_git_worktrees_edit"):
        errors.append("manifest policy must prohibit manual .git/worktrees edits")

    refs = manifest.get("refs")
    if not isinstance(refs, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("name"), str)
        or not isinstance(item.get("head"), str)
        or not HEX_OBJECT_RE.fullmatch(item["head"])
        for item in refs or []
    ):
        errors.append("refs must contain valid names and object IDs")
        refs = []
    if len({item["name"] for item in refs}) != len(refs):
        errors.append("ref names are not unique")
    expected_tags = [item for item in refs if item["name"].startswith("refs/tags/")]
    if manifest.get("tags") != expected_tags:
        errors.append("tags must equal the tagged subset of refs")
    remotes = manifest.get("remotes", [])
    if not isinstance(remotes, list):
        errors.append("remotes must be a list")
        remotes = []
    for index, remote in enumerate(remotes):
        if not isinstance(remote, dict) or not isinstance(remote.get("name"), str):
            errors.append(f"remote {index} is malformed")
            continue
        for key in ("fetch_endpoint_fingerprints", "push_endpoint_fingerprints"):
            for fingerprint in remote.get(key, []):
                if not isinstance(fingerprint, dict) or not re.fullmatch(
                    r"[0-9a-f]{64}", str(fingerprint.get("sha256", ""))
                ):
                    errors.append(f"remote {index} contains an invalid endpoint fingerprint")
    config_entries = manifest.get("config", [])
    if not isinstance(config_entries, list):
        errors.append("config must be a list")
        config_entries = []
    for index, config in enumerate(config_entries):
        if not isinstance(config, dict) or not isinstance(config.get("key"), str):
            errors.append(f"config entry {index} is malformed")
        elif not re.fullmatch(r"[0-9a-f]{64}", str(config.get("value_sha256", ""))):
            errors.append(f"config entry {index} contains an invalid value fingerprint")

    by_path = {entry.get("path"): entry for entry in worktrees}
    conflict = manifest.get("canonical_main_dev_conflict", {})
    canonical = by_path.get(canonical_root)
    dev = by_path.get(dev_worktree)
    if canonical and dev:
        expected_conflict_fields = {
            "canonical_root": canonical_root,
            "canonical_branch": canonical["actual"]["branch"],
            "canonical_head": canonical["actual"]["head"],
            "canonical_dirty_count": canonical["status"]["summary"]["total"],
            "dev_worktree": dev_worktree,
            "dev_branch": dev["actual"]["branch"],
            "dev_head": dev["actual"]["head"],
            "dev_dirty_count": dev["status"]["summary"]["total"],
        }
        for key, expected in expected_conflict_fields.items():
            if conflict.get(key) != expected:
                errors.append(f"canonical conflict field differs from worktree records: {key}")
        if conflict.get("status") != "conflict-preservation-required":
            errors.append("canonical main/dev conflict must remain preservation-required")

    if repo:
        resolved_repo = str(Path(repo).resolve())
        current = _current_registration(resolved_repo)
        recorded = {
            entry["path"]: (
                entry.get("registration", {}).get("head"),
                entry.get("registration", {}).get("branch_ref"),
            )
            for entry in worktrees
        }
        if current != recorded:
            errors.append("live worktree registration/HEAD/branch differs from manifest")
        for entry in worktrees:
            path = entry["path"]
            if path not in current:
                continue
            actual = entry["actual"]
            live_actual = {
                "top_level": _text(path, "rev-parse", "--show-toplevel"),
                "git_dir": _text(path, "rev-parse", "--path-format=absolute", "--git-dir"),
                "common_dir": _text(path, "rev-parse", "--path-format=absolute", "--git-common-dir"),
                "branch": _text(path, "branch", "--show-current") or None,
                "head": _text(path, "rev-parse", "HEAD"),
                "upstream": _upstream(path),
            }
            if live_actual != actual:
                errors.append(f"live Git identity/upstream differs from manifest: {path}")
            if _submodules(path) != entry.get("submodules"):
                errors.append(f"live submodule state differs from manifest: {path}")
            if _ignored_summary(path) != entry.get("ignored"):
                errors.append(f"live collapsed ignored state differs from manifest: {path}")
            if _local_link(path) != entry.get("local_state"):
                errors.append(f"live .local state differs from manifest: {path}")
        if _capture_refs(resolved_repo) != refs:
            errors.append("live refs differ from manifest")
        if _capture_remotes(resolved_repo) != manifest.get("remotes"):
            errors.append("live remote endpoint fingerprints differ from manifest")
        if _capture_config(resolved_repo) != manifest.get("config"):
            errors.append("live local config fingerprints differ from manifest")
    return errors, warnings


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise CaptureError("manifest root must be a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture", help="capture a metadata-only manifest")
    capture_parser.add_argument("--repo", required=True)
    capture_parser.add_argument("--output", required=True)
    capture_parser.add_argument("--canonical-root", default=DEFAULT_CANONICAL_ROOT)
    capture_parser.add_argument("--dev-worktree", default=DEFAULT_DEV_WORKTREE)

    validate_parser = subparsers.add_parser("validate", help="validate schema and optional live registration")
    validate_parser.add_argument("--manifest", required=True)
    validate_parser.add_argument("--repo", help="also compare live worktree registration, HEAD, and branch")

    args = parser.parse_args()
    try:
        if args.command == "capture":
            manifest = capture(args.repo, args.canonical_root, args.dev_worktree)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                "PASS: captured "
                f"{manifest['summary']['registered_worktrees']} worktrees; "
                f"dirty={manifest['summary']['dirty_worktrees']} "
                f"clean={manifest['summary']['clean_worktrees']} "
                f"blocked_scans={len(manifest['summary']['blocked_scans'])}"
            )
            return 0

        manifest = _read_json(args.manifest)
        errors, warnings = validate(manifest, args.repo)
        for warning in warnings:
            print(f"WARNING: {warning}")
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print(
            "PASS: manifest schema, metadata-only boundary, policy, counts, "
            + (
                "and live registration/HEAD/branch/refs/remotes/config fingerprints"
                if args.repo
                else "and registration snapshot"
            )
        )
        return 0
    except (CaptureError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
