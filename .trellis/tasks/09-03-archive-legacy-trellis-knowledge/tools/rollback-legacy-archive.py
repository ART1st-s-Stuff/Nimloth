#!/usr/bin/env python3
"""Deterministic partial-execution rollback; dry-run unless explicitly approved."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
from pathlib import Path
from typing import Any


TOOL_PATH = Path(__file__).with_name("legacy_archive_plan.py")
spec = importlib.util.spec_from_file_location("legacy_archive_plan", TOOL_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {TOOL_PATH}")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def _path_errors(root: Path, relative: str, label: str) -> list[str]:
    return [
        f"{label}: {error}"
        for error in tool.validate_destination_path(root, relative, require_absent=False)
    ]


def classify_moves(root: Path, rows: list[dict[str, Any]]):
    """Read-only classification and validation of every rollback move row."""
    moved, untouched, errors = [], [], []
    for row in rows:
        errors.extend(_path_errors(root, row["source"], "restore path is unsafe"))
        errors.extend(_path_errors(root, row["destination"], "archive path is unsafe"))
        source = root / row["source"]
        destination = root / row["destination"]
        source_exists = os.path.lexists(source)
        destination_exists = os.path.lexists(destination)
        if destination_exists and not source_exists:
            if (not destination.is_file() or destination.is_symlink()
                    or tool.sha256_file(destination) != row["sha256"]
                    or tool.worktree_blob_id(root, row["destination"]) != row["blob"]):
                errors.append(f"moved destination bytes/blob changed: {row['destination']}")
            else:
                moved.append(row)
        elif source_exists and not destination_exists:
            if (not source.is_file() or source.is_symlink()
                    or tool.sha256_file(source) != row["sha256"]
                    or tool.worktree_blob_id(root, row["source"]) != row["blob"]):
                errors.append(f"untouched source bytes/blob changed: {row['source']}")
            else:
                untouched.append(row)
        elif source_exists and destination_exists:
            errors.append(f"both source and destination exist: {row['source']}")
        else:
            errors.append(f"neither source nor destination exists: {row['source']}")
    return moved, untouched, errors


def classify_rewrites(root: Path, patch: dict[str, Any]):
    """Validate all rewrite hashes and exact text before returning rollback actions."""
    actions, errors = [], []
    for entry in reversed(patch["files"]):
        relative = entry["path"]
        errors.extend(_path_errors(root, relative, "rewrite path is unsafe"))
        path = root / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f"rewrite target missing, non-file, or symlink: {relative}")
            continue
        actual = tool.sha256_file(path)
        if actual == entry["pre_sha256"]:
            needle_key = "old"
        elif actual == entry["post_sha256"]:
            needle_key = "new"
            actions.append(entry)
        else:
            errors.append(f"rewrite target has neither reviewed hash: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for replacement in entry["replacements"]:
            needle = replacement[needle_key]
            if text.count(needle) != 1:
                errors.append(f"rewrite text is not exact and unique: {relative}: {needle}")
    return actions, errors


def preflight_rollback(root: Path, rows: list[dict[str, Any]], patch: dict[str, Any]):
    """Complete read-only preflight; callers must not mutate unless errors is empty."""
    tool.assert_exact_repo_root(root)
    move_manifest = tool.sha256_file(tool.MOVES) if root == tool.ROOT else None
    if move_manifest is not None and patch.get("move_manifest_sha256") != move_manifest:
        return [], [], [], ["rollback patch/manifest binding mismatch"]
    moved, untouched, move_errors = classify_moves(root, rows)
    rewrite_actions, rewrite_errors = classify_rewrites(root, patch)
    return moved, untouched, rewrite_actions, move_errors + rewrite_errors


def rollback_rewrites(root: Path, entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        path = root / entry["path"]
        errors = _path_errors(root, entry["path"], "rewrite path changed")
        if errors:
            raise RuntimeError("\n".join(errors))
        if (not path.is_file() or path.is_symlink()
                or tool.sha256_file(path) != entry["post_sha256"]):
            raise RuntimeError(f"rewrite target changed after preflight: {entry['path']}")
        text = path.read_text(encoding="utf-8")
        for replacement in reversed(entry["replacements"]):
            if text.count(replacement["new"]) != 1:
                raise RuntimeError(f"post text changed: {entry['path']}: {replacement['new']}")
            text = text.replace(replacement["new"], replacement["old"], 1)
        path.write_text(text, encoding="utf-8")
        if tool.sha256_file(path) != entry["pre_sha256"]:
            raise RuntimeError(f"reverse rewrite hash mismatch: {entry['path']}")


def reverse_moves(root: Path, moved: list[dict[str, Any]]) -> None:
    for row in reversed(moved):
        print(f"git mv -- {row['destination']} {row['source']}")
        tool.assert_exact_repo_root(root)
        errors = (
            _path_errors(root, row["source"], "restore path changed")
            + _path_errors(root, row["destination"], "archive path changed")
        )
        if errors:
            raise SystemExit("\n".join(errors))
        restore = root / row["source"]
        archived = root / row["destination"]
        if os.path.lexists(restore):
            raise SystemExit(f"restore destination appeared: {row['source']}")
        if (not archived.is_file() or archived.is_symlink()
                or tool.sha256_file(archived) != row["sha256"]
                or tool.worktree_blob_id(root, row["destination"]) != row["blob"]):
            raise SystemExit(f"archive source bytes/blob changed: {row['destination']}")
        subprocess.run(
            ["git", "mv", "--", row["destination"], row["source"]],
            cwd=root, check=True,
        )


def execute_rollback(root: Path, rows: list[dict[str, Any]], patch: dict[str, Any]):
    """Execute only after the complete move-and-rewrite preflight succeeds."""
    moved, untouched, rewrite_actions, errors = preflight_rollback(root, rows, patch)
    if errors:
        raise RuntimeError("\n".join(errors))
    rollback_rewrites(root, rewrite_actions)
    reverse_moves(root, moved)
    return moved, untouched, rewrite_actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved-rollback", action="store_true")
    args = parser.parse_args()
    rows = tool.load_jsonl(tool.MOVES)
    patch = tool.load_rewrite_patch()
    moved, untouched, rewrite_actions, errors = preflight_rollback(tool.ROOT, rows, patch)
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    print(f"{'EXECUTE' if args.execute_approved_rollback else 'DRY-RUN'} rollback: "
          f"moved={len(moved)} untouched={len(untouched)} rewrites={len(rewrite_actions)}")
    if not args.execute_approved_rollback:
        for row in reversed(moved):
            print(f"git mv -- {row['destination']} {row['source']}")
        print("No rollback mutation performed; execution requires --execute-approved-rollback.")
        return

    # Every rewrite and every move row passed the global read-only preflight.
    execute_rollback(tool.ROOT, rows, patch)
    final_moved, _, final_errors = classify_moves(tool.ROOT, rows)
    if final_moved or final_errors:
        raise SystemExit(f"rollback verification failed: moved={len(final_moved)} errors={final_errors}")
    tool.check(mode="pre")


if __name__ == "__main__":
    main()
