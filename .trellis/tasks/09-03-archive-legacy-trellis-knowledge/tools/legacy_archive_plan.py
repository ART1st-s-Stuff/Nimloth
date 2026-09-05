#!/usr/bin/env python3
"""Build, validate, and (only behind the separate shell gate) apply the legacy archive plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


TASK = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
AUDIT = TASK / "research/known-error-audit.jsonl"
MOVES = TASK / "research/legacy-move-manifest.jsonl"
REWRITE_PATCH = TASK / "research/approved-rewrite-patch.json"
DESTINATION_PREFIX = "ai_tasks/archive/pre-trellis/"
ALLOWED_DISPOSITIONS = {
    "retain-contract", "obsolete", "archive-only", "needs-human-decision",
}
UNCERTAIN_EVIDENCE = {"insufficient", "not_reproducible", "superseded"}
EXPECTED_HUMAN_DECISIONS = {
    "ai_rules/known_errors/E0066_do_not_drop_dino_from_grid_rl.md",
    "ai_rules/known_errors/E0068_full_cache_must_request_at_least_64_cpus.md",
    "ai_rules/known_errors/E0072_vllm_memory_pool_rejects_expandable_segments.md",
    "ai_rules/known_errors/E0147_archived_task_metadata_may_not_be_bound_to_trajectory_row.md",
}
LEGACY_PATTERN = re.compile(
    r"ai_rules/known_errors/|ai_tasks/(?!archive/pre-trellis/)|AI_branch_progress\.md|AI_issues\.md"
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
FOCUSED_POST_TESTS = (
    "tests/eval/test_id54_id189_cfm_launcher.py",
    "tests/eval/test_id55_id189_cfm_all_launcher.py",
    "tests/eval/test_id56_wm_decoder_diagnostic_launcher.py",
    "tests/eval/test_id57_state_dino_alignment_launcher.py",
)


def run_git(root: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=check, capture_output=True, text=True
    ).stdout


def assert_exact_repo_root(root: Path) -> None:
    declared = Path(run_git(root, "rev-parse", "--show-toplevel").strip())
    if declared.absolute() != root.absolute() or declared.resolve() != root.resolve():
        raise ValueError(f"repository root mismatch: expected {root}, Git reports {declared}")
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"repository root must be a real directory: {root}")


def tracked_paths(root: Path, prefix: str) -> list[str]:
    output = run_git(root, "ls-tree", "-r", "--name-only", "HEAD", "--", prefix)
    return sorted(line for line in output.splitlines() if line)


def indexed_paths(root: Path, prefix: str) -> list[str]:
    output = run_git(root, "ls-files", "--", prefix)
    return sorted(line for line in output.splitlines() if line)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blob_id(root: Path, source: str) -> str:
    return run_git(root, "rev-parse", f"HEAD:{source}").strip()


def worktree_blob_id(root: Path, path: str) -> str:
    return run_git(root, "hash-object", "--", path).strip()


def expected_known_errors(root: Path, *, mode: str) -> list[str]:
    if mode == "pre":
        candidates = tracked_paths(root, "ai_rules/known_errors")
        return [
            path for path in candidates
            if Path(path).name.startswith("E") and Path(path).suffix == ".md"
        ]
    candidates = indexed_paths(root, "ai_rules/archive/known_errors")
    return [
        "ai_rules/known_errors/" + Path(path).name for path in candidates
        if Path(path).name.startswith("E") and Path(path).suffix == ".md"
    ]


def _mapping_error(root: Path, source: str, mapping: Any) -> str | None:
    if not isinstance(mapping, dict):
        return f"contract mapping is not an object: {source}"
    spec = mapping.get("spec_path")
    anchor = mapping.get("anchor")
    excerpt = mapping.get("excerpt")
    if not all(isinstance(value, str) and value for value in (spec, anchor, excerpt)):
        return f"contract mapping lacks spec_path/anchor/excerpt: {source}"
    if not spec.startswith(".trellis/spec/"):
        return f"contract mapping is outside owning specs: {source}: {spec}"
    path = root / spec
    if not path.is_file():
        return f"contract mapping spec does not exist: {source}: {spec}"
    text = path.read_text(encoding="utf-8")
    if text.count(anchor) != 1:
        return f"contract mapping anchor is not exact and unique: {source}: {anchor}"
    anchor_at = text.index(anchor)
    excerpt_at = text.find(excerpt, anchor_at + len(anchor))
    if excerpt_at < 0:
        return f"contract mapping excerpt is absent after anchor: {source}: {excerpt}"
    next_heading = text.find("\n## ", anchor_at + len(anchor))
    if next_heading >= 0 and excerpt_at >= next_heading:
        return f"contract mapping excerpt is outside anchored section: {source}: {anchor}"
    return None


def validate_audit(root: Path, rows: list[dict[str, Any]], *, mode: str = "pre") -> list[str]:
    if mode not in {"pre", "post"}:
        return [f"unknown audit validation mode: {mode}"]
    errors: list[str] = []
    required = {
        "source", "sha256", "id_prefix", "title", "topic", "evidence_paths",
        "evidence_status", "disposition", "target_specs", "contract_mappings",
        "destination", "notes",
    }
    sources = [row.get("source") for row in rows]
    expected = expected_known_errors(root, mode=mode)
    if sources != expected:
        errors.append("audit source order/union differs from the full sorted tracked E-file set")
    if len(set(sources)) != len(sources):
        errors.append("audit contains duplicate full source paths")
    for index, row in enumerate(rows, 1):
        missing = sorted(required - row.keys())
        if missing:
            errors.append(f"audit row {index} missing fields: {missing}")
            continue
        source = row["source"]
        artifact = source if mode == "pre" else row["destination"]
        path = root / artifact
        if not path.is_file():
            errors.append(f"audit {mode} artifact is not a file: {artifact}")
            continue
        if sha256_file(path) != row["sha256"]:
            errors.append(f"audit SHA-256 mismatch: {artifact}")
        if row["destination"] != "ai_rules/archive/known_errors/" + Path(source).name:
            errors.append(f"audit destination mismatch: {source}")
        if row["id_prefix"] != Path(source).name.split("_", 1)[0]:
            errors.append(f"audit id_prefix mismatch: {source}")
        if row["disposition"] not in ALLOWED_DISPOSITIONS:
            errors.append(f"unknown disposition: {source}")
        if not row["evidence_paths"]:
            errors.append(f"audit has no evidence paths: {source}")
        if not row["notes"].strip():
            errors.append(f"audit has no uncertainty/evidence note: {source}")
        mappings = row["contract_mappings"]
        if not isinstance(mappings, list):
            errors.append(f"contract_mappings is not a list: {source}")
            mappings = []
        if row["disposition"] == "retain-contract":
            if row["evidence_status"] in UNCERTAIN_EVIDENCE:
                errors.append(f"uncertain record cannot assert retain-contract: {source}")
            missing_evidence = [p for p in row["evidence_paths"] if not (root / p).is_file()]
            if missing_evidence:
                errors.append(f"retain-contract cites unavailable current evidence: {source}: {missing_evidence}")
            if not mappings:
                errors.append(f"retain-contract has no exact contract mapping: {source}")
            mapped_specs = {m.get("spec_path") for m in mappings if isinstance(m, dict)}
            if mapped_specs != set(row["target_specs"]):
                errors.append(f"target_specs differ from exact contract mappings: {source}")
            for mapping in mappings:
                error = _mapping_error(root, source, mapping)
                if error:
                    errors.append(error)
        elif mappings or row["target_specs"]:
            errors.append(f"non-retained record must not claim an active contract: {source}")
    decisions = {row["source"] for row in rows if row["disposition"] == "needs-human-decision"}
    if decisions != EXPECTED_HUMAN_DECISIONS:
        errors.append("needs-human-decision set differs from the four reviewed unresolved records")
    return errors


def expected_moves(root: Path, *, mode: str = "pre") -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    if mode == "pre":
        for source in tracked_paths(root, "ai_rules/known_errors"):
            rows.append((source, "ai_rules/archive/known_errors/" + Path(source).name, "known-errors"))
        for source in tracked_paths(root, "ai_tasks"):
            if source.startswith(DESTINATION_PREFIX):
                continue
            rows.append((source, DESTINATION_PREFIX + source.removeprefix("ai_tasks/"), "ai-tasks"))
        for source in ("AI_branch_progress.md", "AI_issues.md"):
            rows.append((source, DESTINATION_PREFIX + "root/" + source, "root-history"))
        return sorted(rows)

    for destination in indexed_paths(root, "ai_rules/archive/known_errors"):
        rows.append(("ai_rules/known_errors/" + Path(destination).name, destination, "known-errors"))
    for destination in indexed_paths(root, DESTINATION_PREFIX):
        suffix = destination.removeprefix(DESTINATION_PREFIX)
        if suffix in {"root/AI_branch_progress.md", "root/AI_issues.md"}:
            rows.append((Path(suffix).name, destination, "root-history"))
        else:
            rows.append(("ai_tasks/" + suffix, destination, "ai-tasks"))
    return sorted(rows)


def build_move_manifest(root: Path) -> list[dict[str, str]]:
    return [
        {"blob": blob_id(root, source), "destination": destination, "group": group,
         "sha256": sha256_file(root / source), "source": source}
        for source, destination, group in expected_moves(root)
    ]


def _safe_relative(path: str) -> bool:
    pure = PurePosixPath(path)
    return not pure.is_absolute() and bool(pure.parts) and ".." not in pure.parts and "." not in pure.parts


def validate_destination_path(root: Path, destination: str, *, require_absent: bool) -> list[str]:
    """Reject traversal, symlinks, non-directory ancestors, and escapes using lexists."""
    errors: list[str] = []
    if not _safe_relative(destination):
        return [f"unsafe destination path: {destination}"]
    root_real = root.resolve(strict=True)
    current = root
    for part in PurePosixPath(destination).parent.parts:
        current = current / part
        if os.path.lexists(current):
            if current.is_symlink():
                errors.append(f"destination ancestor is a symlink: {current.relative_to(root)}")
                break
            if not current.is_dir():
                errors.append(f"destination ancestor is not a directory: {current.relative_to(root)}")
                break
            try:
                current.resolve(strict=True).relative_to(root_real)
            except ValueError:
                errors.append(f"destination ancestor resolves outside exact repo root: {current.relative_to(root)}")
                break
    target = root / destination
    if require_absent and os.path.lexists(target):
        errors.append(f"move destination already exists: {destination}")
    return errors


def prepare_destination_directories(root: Path, rows: list[dict[str, Any]]) -> None:
    """Create and then validate every destination directory before the first move."""
    assert_exact_repo_root(root)
    for row in rows:
        errors = validate_destination_path(root, row["destination"], require_absent=True)
        if errors:
            raise ValueError("; ".join(errors))
    parents = sorted({PurePosixPath(row["destination"]).parent for row in rows}, key=lambda p: (len(p.parts), str(p)))
    for parent in parents:
        current = root
        for part in parent.parts:
            current = current / part
            if os.path.lexists(current):
                if current.is_symlink() or not current.is_dir():
                    raise ValueError(f"unsafe destination ancestor appeared: {current}")
            else:
                current.mkdir()
            current.resolve(strict=True).relative_to(root.resolve(strict=True))
    for row in rows:
        errors = validate_destination_path(root, row["destination"], require_absent=True)
        if errors:
            raise ValueError("; ".join(errors))


def validate_move_manifest(root: Path, rows: list[dict[str, Any]], *, mode: str = "pre") -> list[str]:
    errors: list[str] = []
    expected = expected_moves(root, mode=mode)
    triples = [(row.get("source"), row.get("destination"), row.get("group")) for row in rows]
    if triples != expected:
        errors.append("move manifest order/union differs from tracked source inventory")
    sources = [row.get("source") for row in rows]
    destinations = [row.get("destination") for row in rows]
    if len(sources) != len(set(sources)):
        errors.append("move manifest contains duplicate sources")
    if len(destinations) != len(set(destinations)):
        errors.append("move manifest contains destination collisions")
    for row in rows:
        source, destination = row.get("source"), row.get("destination")
        if not isinstance(source, str) or not isinstance(destination, str):
            errors.append("move manifest has non-string source/destination")
            continue
        if not _safe_relative(source) or not _safe_relative(destination):
            errors.append(f"move path is absolute or traversing: {source} -> {destination}")
            continue
        source_path, target = root / source, root / destination
        if mode == "pre":
            if not source_path.is_file() or source_path.is_symlink():
                errors.append(f"move source missing, non-file, or symlink: {source}")
                continue
            errors.extend(validate_destination_path(root, destination, require_absent=True))
            if row.get("sha256") != sha256_file(source_path):
                errors.append(f"move SHA-256 mismatch: {source}")
            if row.get("blob") != blob_id(root, source):
                errors.append(f"move Git blob mismatch: {source}")
        else:
            if os.path.lexists(source_path):
                errors.append(f"post-move source still exists: {source}")
            if not target.is_file() or target.is_symlink():
                errors.append(f"post-move destination missing, non-file, or symlink: {destination}")
                continue
            errors.extend(validate_destination_path(root, destination, require_absent=False))
            if row.get("sha256") != sha256_file(target):
                errors.append(f"post-move SHA-256 mismatch: {destination}")
            if row.get("blob") != worktree_blob_id(root, destination):
                errors.append(f"post-move Git blob mismatch: {destination}")
    if mode == "pre" and not errors and rows != build_move_manifest(root):
        errors.append("move manifest is not deterministic for current source bytes")
    return errors


def load_rewrite_patch() -> dict[str, Any]:
    return json.loads(REWRITE_PATCH.read_text(encoding="utf-8"))


def task_context_files(root: Path, statuses: set[str]) -> list[Path]:
    result: list[Path] = []
    for task_json in sorted((root / ".trellis/tasks").glob("**/task.json")):
        try:
            task = json.loads(task_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if task.get("status") not in statuses:
            continue
        for name in ("implement.jsonl", "check.jsonl"):
            path = task_json.parent / name
            if path.is_file():
                result.append(path)
    return result


def active_task_context_files(root: Path) -> list[Path]:
    return task_context_files(root, {"in_progress"})


def _legacy_occurrences(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if LEGACY_PATTERN.search(line)]


def active_non_task_files(root: Path) -> list[Path]:
    output = run_git(root, "ls-files", "-co", "--exclude-standard")
    result: list[Path] = []
    for relative in sorted(set(output.splitlines())):
        if not relative:
            continue
        if relative.startswith(("ai_rules/", "ai_tasks/", ".trellis/tasks/", ".memory/", ".local/")):
            continue
        if relative in {"AI_branch_progress.md", "AI_issues.md", "CHANGELOG.md"}:
            continue
        path = root / relative
        if path.is_file() and not path.is_symlink():
            result.append(path)
    return result


def validate_rewrite_patch(root: Path, patch: dict[str, Any], *, mode: str) -> list[str]:
    errors: list[str] = []
    if patch.get("schema_version") != 1:
        errors.append("rewrite patch schema_version is not 1")
    if patch.get("move_manifest_sha256") != sha256_file(MOVES):
        errors.append("rewrite patch is not bound to the exact move manifest hash")
    files = patch.get("files")
    if not isinstance(files, list):
        return errors + ["rewrite patch files is not a list"]
    paths = [entry.get("path") for entry in files if isinstance(entry, dict)]
    if len(paths) != len(set(paths)):
        errors.append("rewrite patch contains duplicate file entries")
    kinds = Counter()
    covered_pre: dict[str, list[str]] = {}
    eligible_context_paths = {
        str(path.relative_to(root))
        for path in task_context_files(root, {"in_progress", "completed"})
    }
    for entry in files:
        if not isinstance(entry, dict):
            errors.append("rewrite patch file entry is not an object")
            continue
        relative = entry.get("path")
        if not isinstance(relative, str) or not _safe_relative(relative):
            errors.append(f"unsafe rewrite path: {relative}")
            continue
        path = root / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f"rewrite target missing, non-file, or symlink: {relative}")
            continue
        payload = path.read_bytes()
        expected_hash = entry.get("pre_sha256" if mode == "pre" else "post_sha256")
        if sha256_bytes(payload) != expected_hash:
            errors.append(f"rewrite {mode} hash mismatch: {relative}")
            continue
        text = payload.decode("utf-8")
        replacements = entry.get("replacements")
        if not isinstance(replacements, list) or not replacements:
            errors.append(f"rewrite entry has no replacements: {relative}")
            continue
        for replacement in replacements:
            old, new, kind = replacement.get("old"), replacement.get("new"), replacement.get("kind")
            if not all(isinstance(value, str) and value for value in (old, new, kind)):
                errors.append(f"invalid rewrite replacement: {relative}")
                continue
            kinds[kind] += 1
            if kind == "active-context" and relative not in eligible_context_paths:
                errors.append(f"active-context rewrite is outside an active or completed task: {relative}")
            needle = old if mode == "pre" else new
            if text.count(needle) != 1:
                errors.append(f"rewrite {mode} text is not exact and unique: {relative}: {needle}")
            if mode == "pre":
                covered_pre.setdefault(relative, []).append(old)
    if kinds != Counter({"active-context": 19, "eval-provenance": 4, "slurm-provenance": 5}):
        errors.append(f"rewrite patch category counts changed: {dict(kinds)}")
    context_files = active_task_context_files(root)
    for path in context_files:
        relative = str(path.relative_to(root))
        occurrences = _legacy_occurrences(path)
        if mode == "pre":
            for line in occurrences:
                if not any(old in line for old in covered_pre.get(relative, [])):
                    errors.append(f"active context legacy reference is not in exact patch: {relative}: {line}")
        elif occurrences:
            errors.append(f"active task context retains legacy references: {relative}: {occurrences}")
    for path in active_non_task_files(root):
        relative = str(path.relative_to(root))
        occurrences = _legacy_occurrences(path)
        if mode == "pre":
            for line in occurrences:
                if not any(old in line for old in covered_pre.get(relative, [])):
                    errors.append(f"active legacy reference is not in exact patch: {relative}: {line}")
        elif occurrences:
            errors.append(f"post-move active legacy reference remains: {relative}: {occurrences}")
    return errors


def apply_rewrite_patch(root: Path, patch: dict[str, Any], *, reverse: bool = False) -> None:
    mode = "post" if reverse else "pre"
    errors = validate_rewrite_patch(root, patch, mode=mode)
    if errors:
        raise ValueError("\n".join(errors))
    for entry in patch["files"]:
        path = root / entry["path"]
        text = path.read_text(encoding="utf-8")
        replacements = reversed(entry["replacements"]) if reverse else entry["replacements"]
        for replacement in replacements:
            old, new = replacement["old"], replacement["new"]
            text = text.replace(new, old, 1) if reverse else text.replace(old, new, 1)
        path.write_text(text, encoding="utf-8")
        expected = entry["pre_sha256" if reverse else "post_sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"rewrite output hash mismatch: {entry['path']}")


def validate_active_contexts(root: Path) -> list[str]:
    errors: list[str] = []
    for path in active_task_context_files(root):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid active context JSONL: {path.relative_to(root)}:{number}: {exc}")
                continue
            selected = row.get("file")
            if selected and not (root / selected).is_file():
                errors.append(f"active context file does not exist: {path.relative_to(root)}:{number}: {selected}")
    return errors


def validate_markdown_links(root: Path) -> list[str]:
    errors: list[str] = []
    for path in active_non_task_files(root):
        if path.suffix.lower() != ".md":
            continue
        for target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8", errors="replace")):
            target = target.split()[0].strip("<>").split("#", 1)[0]
            if not target or target == "url" or "://" in target or target.startswith(("mailto:", "/")):
                continue
            if not (path.parent / target).resolve().exists():
                errors.append(f"broken Markdown link: {path.relative_to(root)} -> {target}")
    return errors


def run_focused_post_tests(root: Path) -> list[str]:
    errors: list[str] = []
    for test in FOCUSED_POST_TESTS:
        completed = subprocess.run(
            ["python3", test], cwd=root, text=True, capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if completed.returncode:
            errors.append(f"focused post-move test failed: {test}: {completed.stdout}{completed.stderr}")
    return errors


def manifest_fingerprint(path: Path) -> str:
    return sha256_file(path)


def check(*, mode: str = "pre", run_focused_tests: bool = False) -> dict[str, Any]:
    assert_exact_repo_root(ROOT)
    audit = load_jsonl(AUDIT)
    moves = load_jsonl(MOVES)
    patch = load_rewrite_patch()
    errors = (
        validate_audit(ROOT, audit, mode=mode)
        + validate_move_manifest(ROOT, moves, mode=mode)
        + validate_rewrite_patch(ROOT, patch, mode=mode)
    )
    if mode == "post":
        errors += validate_active_contexts(ROOT) + validate_markdown_links(ROOT)
        if run_focused_tests:
            errors += run_focused_post_tests(ROOT)
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    dispositions = Counter(row["disposition"] for row in audit)
    return {
        "archive_only": dispositions["archive-only"],
        "human_decisions": dispositions["needs-human-decision"],
        "known_error_records": len(audit),
        "mode": mode,
        "move_manifest_sha256": manifest_fingerprint(MOVES),
        "move_records": len(moves),
        "retain_contract": dispositions["retain-contract"],
        "rewrite_files": len(patch["files"]),
        "rewrite_patch_sha256": manifest_fingerprint(REWRITE_PATCH),
        "rewrite_replacements": sum(len(entry["replacements"]) for entry in patch["files"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-move-manifest", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--mode", choices=("pre", "post"), default="pre")
    parser.add_argument("--run-focused-tests", action="store_true")
    args = parser.parse_args()
    if args.generate_move_manifest:
        if args.mode != "pre":
            parser.error("move manifest generation is pre-move only")
        dump_jsonl(MOVES, build_move_manifest(ROOT))
    if args.check:
        print(json.dumps(check(mode=args.mode, run_focused_tests=args.run_focused_tests), sort_keys=True))
    if not args.generate_move_manifest and not args.check:
        parser.error("choose --generate-move-manifest and/or --check")


if __name__ == "__main__":
    main()
