#!/usr/bin/env python3
"""Generate a read-only exact local-object plan for the four live dev modules.

This command probes live Git metadata but never transfers an object, initializes
or updates a submodule, writes a ref/config/FETCH_HEAD, or changes a worktree.
The emitted commands are templates for a separately approved cutover batch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
OBJECT_TOOL = HERE / "local_object_bootstrap.py"
CANONICAL_ROOT = Path("/workspace/remote2/nimloth")
OLD_DEV = Path("/workspace/remote2/nimloth-dev")
COMMON_GIT = CANONICAL_ROOT / ".git"
OLD_DEV_MODULES = COMMON_GIT / "worktrees/dev/modules/external"
CANONICAL_MODULES = COMMON_GIT / "modules/external"
SCOPE_PATHS = (
    (
        "external/RCDM",
        OLD_DEV_MODULES / "RCDM",
        CANONICAL_MODULES / "RCDM",
        CANONICAL_ROOT / "external/RCDM",
    ),
    (
        "external/VAGEN",
        OLD_DEV_MODULES / "VAGEN",
        CANONICAL_MODULES / "VAGEN",
        CANONICAL_ROOT / "external/VAGEN",
    ),
    (
        "external/VAGEN/verl",
        OLD_DEV_MODULES / "VAGEN/modules/verl",
        CANONICAL_MODULES / "VAGEN/modules/verl",
        CANONICAL_ROOT / "external/VAGEN/verl",
    ),
    (
        "external/le-wm",
        OLD_DEV_MODULES / "le-wm",
        CANONICAL_MODULES / "le-wm",
        CANONICAL_ROOT / "external/le-wm",
    ),
)


class PlanError(RuntimeError):
    pass


def load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("local_object_bootstrap", OBJECT_TOOL)
    if spec is None or spec.loader is None:
        raise PlanError("cannot load local_object_bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_git_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_ALLOW_PROTOCOL": "",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    return environment


def run_git(cwd: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=safe_git_environment(),
    )
    if completed.returncode != 0:
        raise PlanError(
            f"read-only Git command failed at {cwd}: {args}: "
            + completed.stderr.decode("utf-8", "replace")
        )
    return completed.stdout


def git_text(cwd: Path, *args: str) -> str:
    return run_git(cwd, *args).decode("utf-8", "surrogateescape").strip()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def exact_real_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise PlanError(f"{label} is not an exact real directory: {path}")


def expected_commits() -> dict[str, str]:
    return {
        "external/RCDM": git_text(OLD_DEV, "rev-parse", "HEAD:external/RCDM"),
        "external/VAGEN": git_text(OLD_DEV, "rev-parse", "HEAD:external/VAGEN"),
        "external/VAGEN/verl": git_text(
            OLD_DEV / "external/VAGEN", "rev-parse", "HEAD:verl"
        ),
        "external/le-wm": git_text(OLD_DEV, "rev-parse", "HEAD:external/le-wm"),
    }


def canonical_object_tool() -> Path:
    return (
        CANONICAL_ROOT
        / ".trellis/tasks/08-28-refactor-local-worktree-layout/tools/local_object_bootstrap.py"
    )


def transfer_command(source: Path, destination: Path, commit: str) -> list[str]:
    return [
        "python3",
        str(canonical_object_tool()),
        "transfer",
        "--source",
        str(source),
        "--destination",
        str(destination),
        "--commit",
        commit,
        "--approved-exact-object-transfer",
    ]


def future_probe_command(source: Path, destination: Path, commit: str) -> list[str]:
    command = transfer_command(source, destination, commit)
    command[2] = "probe"
    command.pop()
    return command


def scope_record(
    tool: Any,
    name: str,
    source: Path,
    destination: Path,
    worktree: Path,
    commit: str,
) -> dict[str, Any]:
    source = tool.exact_git_dir(source, f"live source {name}")
    if tool.optional_head(source) != commit or not tool.object_exists(source, commit):
        raise PlanError(f"live source HEAD/object mismatch for {name}: {commit}")
    object_format = tool.repository_object_format(source)
    source_objects, source_missing = tool.closure_objects(
        source, commit, require_complete=True
    )
    if source_missing or commit not in source_objects:
        raise PlanError(f"live source closure mismatch for {name}: {commit}")
    sorted_source_objects = sorted(source_objects)
    source_closure_sha256 = tool.sha256_bytes(
        ("\n".join(sorted_source_objects) + "\n").encode("ascii")
    )
    record: dict[str, Any] = {
        "name": name,
        "source_git_dir": str(source),
        "destination_module_git_dir": str(destination),
        "destination_worktree": str(worktree),
        "expected_commit": commit,
        "source_head_matches": True,
        "source_exact_commit_exists": True,
        "source_closure_complete": True,
        "source_closure_object_count": len(sorted_source_objects),
        "source_closure_sha256": source_closure_sha256,
        "object_format": object_format,
        "source_is_real_non_symlink_git_dir": True,
        "destination_path_is_symlink": destination.is_symlink(),
    }
    if destination.exists():
        probe = tool.probe(source, destination, commit)
        already_present = probe["destination_closure_complete"]
        record.update(
            {
                "destination_state": "existing-exact-module-git-dir",
                "destination_common_dir": probe["destination"]["common_dir"],
                "destination_head_before": probe["destination"]["head"],
                "destination_exact_commit_exists_before": probe[
                    "destination_exact_commit_exists"
                ],
                "destination_closure_complete_before": already_present,
                "destination_fingerprints_before": probe[
                    "destination_fingerprints"
                ],
                "future_action": "read-only-noop-verification"
                if already_present
                else "approved-exact-object-transfer",
                "future_command": future_probe_command(source, destination, commit)
                if already_present
                else transfer_command(source, destination, commit),
                "ready_after_writer_pause_and_identity_recheck": True,
            }
        )
    else:
        parent = destination.parent
        exact_real_directory(parent, f"prospective destination parent for {name}")
        if os.path.lexists(destination):
            raise PlanError(f"prospective destination is a non-directory collision: {destination}")
        record.update(
            {
                "destination_state": "absent-requires-minimal-local-module-init",
                "destination_common_dir": str(destination),
                "destination_head_before": None,
                "destination_exact_commit_exists_before": False,
                "destination_closure_complete_before": False,
                "destination_fingerprints_before": None,
                "future_action": "approved-minimal-init-then-exact-object-transfer",
                "minimal_init_command": [
                    "python3",
                    str(canonical_object_tool()),
                    "init-empty",
                    "--destination",
                    str(destination),
                    "--worktree",
                    str(worktree),
                    "--object-format",
                    object_format,
                    "--approved-exact-module-init",
                ],
                "post_init_probe_command": future_probe_command(
                    source, destination, commit
                ),
                "future_command": transfer_command(source, destination, commit),
                "ready_after_writer_pause_and_identity_recheck": False,
                "minimal_init_invariants": {
                    "destination_git_dir_must_be_absent": True,
                    "destination_worktree_must_be_exact_real_empty_directory": True,
                    "object_format_is_explicit_and_matches_source": True,
                    "network_transport_available": False,
                    "remote_ref_or_FETCH_HEAD_created": False,
                    "minimal_repository_config_becomes_transfer_baseline": True,
                    "existing_directory_or_nonempty_worktree_is_never_overwritten": True,
                },
                "blocker": "destination module repo does not exist before canonical dev checkout; initialize only in the separately approved cutover after the exact worktree path exists and is empty",
            }
        )
    record["transfer_invariants"] = {
        "source_HEAD_type_object_format_and_closure_rechecked": True,
        "destination_module_common_object_format_and_identity_rechecked": True,
        "pack_checksum_and_exact_closure_object_set_verified": True,
        "HEAD_index_refs_config_remotes_FETCH_HEAD_fingerprinted_before_after": True,
        "destination_exact_closure_complete_after": True,
        "path_overlap_symlink_object_escape_or_commit_mismatch_rejected": True,
        "inherited_GIT_routing_stripped": True,
        "lazy_fetch_and_network_transport_disabled": True,
        "worktree_not_accessed_by_object_transfer": True,
    }
    return record


def generate(
    *,
    expected_canonical_head: str | None = None,
    expected_old_dev_head: str | None = None,
) -> dict[str, Any]:
    tool = load_tool()
    exact_real_directory(CANONICAL_ROOT, "canonical root")
    exact_real_directory(OLD_DEV, "old dev root")
    canonical_head = git_text(CANONICAL_ROOT, "rev-parse", "HEAD")
    canonical_branch = git_text(CANONICAL_ROOT, "branch", "--show-current")
    old_dev_head = git_text(OLD_DEV, "rev-parse", "HEAD")
    old_dev_branch = git_text(OLD_DEV, "branch", "--show-current")
    if canonical_branch != "main":
        raise PlanError("canonical root is no longer on the pre-cutover main branch")
    if old_dev_branch != "dev":
        raise PlanError("old dev root is no longer on the pre-cutover dev branch")
    if expected_canonical_head and canonical_head != expected_canonical_head:
        raise PlanError("canonical root HEAD differs from the explicit expected baseline")
    if expected_old_dev_head and old_dev_head != expected_old_dev_head:
        raise PlanError("old dev HEAD differs from the explicit expected baseline")
    worktree_porcelain = run_git(OLD_DEV, "worktree", "list", "--porcelain")
    registrations = worktree_porcelain.count(b"worktree ")
    if registrations != 39:
        raise PlanError(f"live worktree registration count changed: {registrations}")
    commits = expected_commits()
    scopes = [
        scope_record(tool, name, source, destination, worktree, commits[name])
        for name, source, destination, worktree in SCOPE_PATHS
    ]
    if (
        git_text(CANONICAL_ROOT, "rev-parse", "HEAD") != canonical_head
        or git_text(CANONICAL_ROOT, "branch", "--show-current") != canonical_branch
        or git_text(OLD_DEV, "rev-parse", "HEAD") != old_dev_head
        or git_text(OLD_DEV, "branch", "--show-current") != old_dev_branch
        or run_git(OLD_DEV, "worktree", "list", "--porcelain") != worktree_porcelain
    ):
        raise PlanError("live branch/HEAD/worktree topology changed during plan capture")
    return {
        "schema": "nimloth-live-module-object-bootstrap-plan/v1",
        "capture_mode": "read-only; no live object transfer/init/update/ref/config/FETCH_HEAD mutation",
        "live_baseline": {
            "canonical_root": str(CANONICAL_ROOT),
            "canonical_branch": canonical_branch,
            "canonical_head": canonical_head,
            "old_dev_root": str(OLD_DEV),
            "old_dev_branch": old_dev_branch,
            "old_dev_head": old_dev_head,
            "registered_worktrees": registrations,
            "worktree_porcelain_sha256": sha256(worktree_porcelain),
            "explicit_expected_canonical_head": expected_canonical_head,
            "explicit_expected_old_dev_head": expected_old_dev_head,
            "baseline_rechecked_after_all_scope_probes": True,
        },
        "scopes": scopes,
        "execution_order": [
            "pause all Git writers and re-run this read-only plan",
            "complete separately approved payload clean/detach/canonical checkout to exact dev HEAD",
            "verify external/RCDM is an exact real worktree directory and initialize only its absent module Git dir with the listed no-transport minimal init",
            "run each scope probe/transfer command in order; already-complete le-wm remains a read-only noop",
            "run the separately authorized offline recursive update and verify all four exact commits; this update, unlike object transfer, writes module worktrees/indexes and may register the previously uninitialized RCDM in superproject local config",
            "compare every transfer-scoped module HEAD/index/refs/config/remotes/FETCH_HEAD fingerprint exactly; separately review the expected RCDM init/update config and worktree/index deltas plus topology/HEAD/status protection baselines",
        ],
        "offline_recursive_update_command": [
            "env",
            "-i",
            "PATH=/run/current-system/sw/bin:/usr/bin:/bin",
            "HOME=/nonexistent",
            "LC_ALL=C",
            "LANG=C",
            "GIT_CONFIG_NOSYSTEM=1",
            "GIT_CONFIG_GLOBAL=/dev/null",
            "GIT_TERMINAL_PROMPT=0",
            "GIT_NO_LAZY_FETCH=1",
            "GIT_NO_REPLACE_OBJECTS=1",
            "GIT_ALLOW_PROTOCOL=",
            "git",
            "-C",
            str(CANONICAL_ROOT),
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--no-fetch",
            "--checkout",
        ],
        "policy": {
            "live_transfer_or_update_executed_this_batch": False,
            "local_pack_verified_unpack_only": True,
            "network_transport_and_lazy_fetch_disabled": True,
            "transfer_does_not_modify_HEAD_index_ref_config_remote_FETCH_HEAD_or_worktree": True,
            "RCDM_minimal_init_is_a_distinct_future_cutover_mutation_with_explicit_sha1_format": True,
            "offline_update_worktree_index_and_RCDM_registration_deltas_are_distinct_from_transfer": True,
            "fail_closed_if_git_dir_object_format_closure_path_symlink_identity_or_fingerprint_differs": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-canonical-head")
    parser.add_argument("--expected-old-dev-head")
    args = parser.parse_args()
    try:
        result = generate(
            expected_canonical_head=args.expected_canonical_head,
            expected_old_dev_head=args.expected_old_dev_head,
        )
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(
            "PASS: read-only four-scope live local-object plan captured; "
            "live transfer/init/update executed=false"
        )
        return 0
    except (OSError, PlanError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
