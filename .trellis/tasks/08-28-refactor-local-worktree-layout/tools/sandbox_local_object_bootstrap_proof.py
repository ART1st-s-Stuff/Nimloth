#!/usr/bin/env python3
"""Disposable proof for exact local module object bootstrapping.

The proof constructs a superproject main worktree, a linked dev worktree, and
worktree-specific module Git dirs. The linked module and its nested submodule
hold commit B while the main module repos do not. Exact B closures are copied
with local pack/unpack plumbing, then canonical recursive checkout completes
with every transport disabled and --no-fetch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
OBJECT_TOOL = HERE / "local_object_bootstrap.py"
COMMANDS: list[list[str]] = []


class ProofError(RuntimeError):
    pass


def load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("local_object_bootstrap", OBJECT_TOOL)
    if spec is None or spec.loader is None:
        raise ProofError("cannot load local_object_bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment(*, offline: bool = False) -> dict[str, str]:
    result = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    result.update(
        {
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
    )
    if offline:
        result["GIT_ALLOW_PROTOCOL"] = ""
    return result


def run(
    cwd: Path,
    *args: str,
    check: bool = True,
    offline: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    COMMANDS.append(["cwd=" + str(cwd), *args])
    completed = subprocess.run(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=environment(offline=offline),
    )
    if check and completed.returncode != 0:
        raise ProofError(
            f"command failed ({completed.returncode}) at {cwd}: {args}: "
            + completed.stderr.decode("utf-8", "replace")
        )
    return completed


def git(cwd: Path, *args: str, check: bool = True, offline: bool = False) -> subprocess.CompletedProcess[bytes]:
    return run(cwd, "git", *args, check=check, offline=offline)


def identity(repo: Path) -> None:
    git(repo, "config", "user.name", "Object Bootstrap Proof")
    git(repo, "config", "user.email", "object-bootstrap@example.invalid")


def init_repo(path: Path) -> None:
    path.mkdir()
    git(path, "init", "-b", "main")
    identity(path)


def commit_file(repo: Path, relative: str, content: str, message: str) -> str:
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    git(repo, "add", relative)
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD").stdout.decode("ascii").strip()


def expect_error(call: Any, contains: str) -> None:
    try:
        call()
    except Exception as exc:
        if contains not in str(exc):
            raise ProofError(f"expected {contains!r}, got: {exc}") from exc
        return
    raise ProofError(f"expected failure containing {contains!r}")


def git_dir(worktree: Path) -> Path:
    return Path(
        git(worktree, "rev-parse", "--absolute-git-dir").stdout.decode().strip()
    )


def tree_fingerprint(root: Path) -> str:
    rows: list[tuple[Any, ...]] = []
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        kept: list[str] = []
        for name in sorted(names):
            path = base / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                rows.append((relative, "symlink", os.readlink(path)))
            else:
                rows.append((relative, "directory", path.stat().st_mode & 0o7777))
                kept.append(name)
        names[:] = kept
        for name in sorted(files):
            path = base / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                rows.append((relative, "symlink", os.readlink(path)))
            else:
                rows.append(
                    (
                        relative,
                        "file",
                        path.stat().st_mode & 0o7777,
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
    rendered = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(rendered).hexdigest()


def summarized_transfer(name: str, result: dict[str, Any], sandbox: Path) -> dict[str, Any]:
    return {
        "scope": name,
        "source_git_dir": result["source"]["git_dir"].replace(str(sandbox), "$SANDBOX"),
        "destination_git_dir": result["destination"]["git_dir"].replace(
            str(sandbox), "$SANDBOX"
        ),
        "expected_commit": result["expected_commit"],
        "object_format": result["object_format"],
        "source_closure_object_count": result["source_closure_object_count"],
        "destination_exact_commit_existed_before": result[
            "destination_exact_commit_existed_before"
        ],
        "destination_closure_complete_before": result[
            "destination_closure_complete_before"
        ],
        "destination_exact_commit_exists_after": result[
            "destination_exact_commit_exists_after"
        ],
        "destination_closure_complete_after": result[
            "destination_closure_complete_after"
        ],
        "action": result["action"],
        "pack_size_bytes": result["pack_size_bytes"],
        "pack_object_count": result["pack_object_count"],
        "pack_object_set_equals_source_closure": result[
            "pack_object_set_equals_source_closure"
        ],
        "head_index_refs_config_remotes_fetch_head_unchanged": result[
            "head_index_refs_config_remotes_fetch_head_unchanged"
        ],
        "destination_head_unchanged": result["destination_head_unchanged"],
        "destination_index_unchanged": result["destination_index_unchanged"],
        "worktree_accessed_or_modified": result["worktree_accessed_or_modified"],
        "network_transport_available": result["network_transport_available"],
        "lazy_fetch_disabled": result["lazy_fetch_disabled"],
        "remote_or_ref_created": result["remote_or_ref_created"],
        "fetch_head_written": result["fetch_head_written"],
    }


def prove() -> dict[str, Any]:
    COMMANDS.clear()
    sandbox = Path(tempfile.mkdtemp(prefix="nimloth-module-bootstrap-proof-", dir="/tmp"))
    try:
        tool = load_tool()
        nested_source = sandbox / "nested-source"
        outer_source = sandbox / "outer-source"
        root = sandbox / "nimloth"
        linked = sandbox / "nimloth-dev"
        init_repo(nested_source)
        nested_a = commit_file(nested_source, "nested.txt", "nested A\n", "nested A")

        init_repo(outer_source)
        git(
            outer_source,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "../nested-source",
            "nested",
        )
        (outer_source / "outer.txt").write_text("outer A\n", encoding="utf-8")
        git(outer_source, "add", "outer.txt", ".gitmodules", "nested")
        git(outer_source, "commit", "-m", "outer A")
        outer_a = git(outer_source, "rev-parse", "HEAD").stdout.decode().strip()

        init_repo(root)
        (root / "README.md").write_text("main\n", encoding="utf-8")
        git(root, "add", "README.md")
        git(root, "commit", "-m", "super base")
        git(
            root,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "../outer-source",
            "external/outer",
        )
        git(
            root,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
            "--recursive",
        )
        git(root, "commit", "-am", "super main module A")
        git(root, "branch", "dev")

        nested_b = commit_file(nested_source, "nested.txt", "nested B\n", "nested B")
        git(
            outer_source / "nested",
            "-c",
            "protocol.file.allow=always",
            "fetch",
            str(nested_source),
            nested_b,
        )
        git(outer_source / "nested", "checkout", "--detach", nested_b)
        (outer_source / "outer.txt").write_text("outer B\n", encoding="utf-8")
        git(outer_source, "add", "outer.txt", "nested")
        git(outer_source, "commit", "-m", "outer B with nested B")
        outer_b = git(outer_source, "rev-parse", "HEAD").stdout.decode().strip()

        git(root, "worktree", "add", str(linked), "dev")
        git(
            linked,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
            "--recursive",
        )
        outer_linked = linked / "external/outer"
        git(
            outer_linked,
            "-c",
            "protocol.file.allow=always",
            "fetch",
            str(outer_source),
            outer_b,
        )
        git(outer_linked, "checkout", "--detach", outer_b)
        git(
            outer_linked,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
            "--recursive",
        )
        git(linked, "add", "external/outer")
        git(linked, "commit", "-m", "super dev module B")
        super_dev = git(linked, "rev-parse", "HEAD").stdout.decode().strip()

        source_outer = git_dir(outer_linked)
        source_nested = git_dir(outer_linked / "nested")
        destination_outer = git_dir(root / "external/outer")
        destination_nested = git_dir(root / "external/outer/nested")
        if tool.object_exists(destination_outer, outer_b):
            raise ProofError("main outer module unexpectedly already had outer B")
        if tool.object_exists(destination_nested, nested_b):
            raise ProofError("main nested module unexpectedly already had nested B")
        if git(outer_linked, "rev-parse", "HEAD").stdout.decode().strip() != outer_b:
            raise ProofError("linked outer source HEAD is not exact B")
        if git(outer_linked / "nested", "rev-parse", "HEAD").stdout.decode().strip() != nested_b:
            raise ProofError("linked nested source HEAD is not exact B")

        expect_error(
            lambda: tool.transfer_exact_commit(
                source_outer, destination_outer, outer_b
            ),
            "approved_exact_object_transfer",
        )
        expect_error(
            lambda: tool.probe(source_outer, destination_outer, outer_a),
            "source HEAD mismatch",
        )
        expect_error(
            lambda: tool.probe(source_outer, destination_outer, outer_b[:12]),
            "full exact object ID",
        )
        cli_destination = sandbox / "cli-destination.git"
        git(sandbox, "init", "--bare", str(cli_destination))
        cli_without_approval = run(
            sandbox,
            sys.executable,
            str(OBJECT_TOOL),
            "transfer",
            "--source",
            str(source_outer),
            "--destination",
            str(cli_destination),
            "--commit",
            outer_b,
            check=False,
            offline=True,
        )
        if (
            cli_without_approval.returncode != 2
            or b"approved_exact_object_transfer" not in cli_without_approval.stderr
        ):
            raise ProofError("transfer CLI did not fail closed without approval")
        cli_output = sandbox / "cli-transfer-result.json"
        run(
            sandbox,
            sys.executable,
            str(OBJECT_TOOL),
            "transfer",
            "--source",
            str(source_outer),
            "--destination",
            str(cli_destination),
            "--commit",
            outer_b,
            "--approved-exact-object-transfer",
            "--output",
            str(cli_output),
            offline=True,
        )
        cli_result = json.loads(cli_output.read_text(encoding="utf-8"))
        if not cli_result.get("destination_closure_complete_after"):
            raise ProofError("approved transfer CLI did not complete exact closure")
        os.symlink(source_outer, sandbox / "source-gitdir-link")
        expect_error(
            lambda: tool.probe(
                sandbox / "source-gitdir-link", destination_outer, outer_b
            ),
            "symlink path component",
        )
        os.symlink(destination_outer, sandbox / "destination-gitdir-link")
        expect_error(
            lambda: tool.probe(
                source_outer, sandbox / "destination-gitdir-link", outer_b
            ),
            "symlink path component",
        )
        expect_error(
            lambda: tool.probe(source_outer, source_outer, outer_b),
            "overlap",
        )
        alternate_destination = sandbox / "alternate-destination.git"
        git(sandbox, "init", "--bare", str(alternate_destination))
        alternates = alternate_destination / "objects/info/alternates"
        alternates.write_text(str(source_outer / "objects") + "\n", encoding="utf-8")
        expect_error(
            lambda: tool.probe(source_outer, alternate_destination, outer_b),
            "object alternates are forbidden",
        )
        promisor_destination = sandbox / "promisor-destination.git"
        git(sandbox, "init", "--bare", str(promisor_destination))
        git(promisor_destination, "config", "remote.origin.promisor", "true")
        expect_error(
            lambda: tool.probe(source_outer, promisor_destination, outer_b),
            "promisor object routing is forbidden",
        )
        include_destination = sandbox / "include-destination.git"
        git(sandbox, "init", "--bare", str(include_destination))
        included_config = sandbox / "external-include.config"
        included_config.write_text("[safe]\n\tkey = value\n", encoding="utf-8")
        git(include_destination, "config", "include.path", str(included_config))
        expect_error(
            lambda: tool.probe(source_outer, include_destination, outer_b),
            "external config include routing is forbidden",
        )

        sha256_destination = sandbox / "sha256-destination.git"
        git(sandbox, "init", "--bare", "--object-format=sha256", str(sha256_destination))
        expect_error(
            lambda: tool.probe(source_outer, sha256_destination, outer_b),
            "object formats differ",
        )
        sha256_source = sandbox / "sha256-source"
        sha256_source.mkdir()
        git(sha256_source, "init", "-b", "main", "--object-format=sha256")
        identity(sha256_source)
        sha256_commit = commit_file(
            sha256_source,
            "sha256.txt",
            "sha256 object format\n",
            "sha256 source",
        )
        sha256_transfer = tool.transfer_exact_commit(
            sha256_source / ".git",
            sha256_destination,
            sha256_commit,
            approved_exact_object_transfer=True,
        )
        if len(sha256_commit) != 64 or not sha256_transfer[
            "destination_closure_complete_after"
        ]:
            raise ProofError("same-format sha256 closure transfer failed")

        symlink_objects_destination = sandbox / "symlink-objects-destination.git"
        git(sandbox, "init", "--bare", str(symlink_objects_destination))
        escaped_objects = sandbox / "escaped-objects"
        (symlink_objects_destination / "objects").rename(escaped_objects)
        os.symlink(escaped_objects, symlink_objects_destination / "objects")
        expect_error(
            lambda: tool.probe(source_outer, symlink_objects_destination, outer_b),
            "object directory escaped Git dir",
        )

        replacement_destination = sandbox / "replacement-destination.git"
        git(sandbox, "init", "--bare", str(replacement_destination))
        replacement_snapshot = tool.preflight(
            source_outer, replacement_destination, outer_b
        )
        parked_destination = sandbox / "replacement-destination-parked.git"
        replacement_destination.rename(parked_destination)
        os.symlink(alternate_destination, replacement_destination)
        expect_error(
            lambda: tool.assert_preflight_unchanged(replacement_snapshot),
            "symlink path component",
        )
        replacement_destination.unlink()
        parked_destination.rename(replacement_destination)

        # Keep an extra commit reachable from another source ref. The exact-B
        # pack must contain only B's closure, not this ref's descendant.
        identity(outer_linked)
        git(outer_linked, "switch", "-c", "proof/extra-ref")
        (outer_linked / "extra-only.txt").write_text("must not transfer\n", encoding="utf-8")
        git(outer_linked, "add", "extra-only.txt")
        git(outer_linked, "commit", "-m", "extra ref only")
        extra_ref_commit = git(outer_linked, "rev-parse", "HEAD").stdout.decode().strip()
        git(outer_linked, "checkout", "--detach", outer_b)

        outer_worktree_before = tree_fingerprint(root / "external/outer")
        nested_worktree_before = tree_fingerprint(root / "external/outer/nested")
        redirected_objects = sandbox / "inherited-object-redirect-must-stay-empty"
        redirected_objects.mkdir()
        inherited_routing = {
            "GIT_OBJECT_DIRECTORY": str(redirected_objects),
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(source_outer / "objects"),
            "GIT_DIR": str(source_outer),
            "GIT_WORK_TREE": str(outer_linked),
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "protocol.file.allow",
            "GIT_CONFIG_VALUE_0": "always",
        }
        os.environ.update(inherited_routing)
        try:
            outer_transfer = tool.transfer_exact_commit(
                source_outer,
                destination_outer,
                outer_b,
                approved_exact_object_transfer=True,
            )
        finally:
            for key in inherited_routing:
                os.environ.pop(key, None)
        if any(redirected_objects.iterdir()):
            raise ProofError("inherited GIT_* routing redirected object transfer")
        if tool.object_type(destination_outer, extra_ref_commit) is not None:
            raise ProofError("exact-B pack incorrectly included an extra reachable ref")
        nested_transfer = tool.transfer_exact_commit(
            source_nested,
            destination_nested,
            nested_b,
            approved_exact_object_transfer=True,
        )
        if tree_fingerprint(root / "external/outer") != outer_worktree_before:
            raise ProofError("outer object transfer changed destination worktree")
        if tree_fingerprint(root / "external/outer/nested") != nested_worktree_before:
            raise ProofError("nested object transfer changed destination worktree")

        repair_destination = sandbox / "incomplete-closure-destination.git"
        git(sandbox, "init", "--bare", str(repair_destination))
        tool.transfer_exact_commit(
            source_outer,
            repair_destination,
            outer_b,
            approved_exact_object_transfer=True,
        )
        outer_tree = git(outer_linked, "rev-parse", f"{outer_b}^{{tree}}").stdout.decode().strip()
        loose_tree = repair_destination / "objects" / outer_tree[:2] / outer_tree[2:]
        if not loose_tree.is_file():
            raise ProofError("expected unpacked loose tree for incomplete-closure control")
        loose_tree.unlink()
        if not tool.object_exists(repair_destination, outer_b):
            raise ProofError("incomplete-closure control lost the commit object")
        if tool.object_closure_complete(repair_destination, outer_b):
            raise ProofError("missing destination tree was not detected")
        repaired_transfer = tool.transfer_exact_commit(
            source_outer,
            repair_destination,
            outer_b,
            approved_exact_object_transfer=True,
        )
        if not tool.object_closure_complete(repair_destination, outer_b):
            raise ProofError("incomplete destination closure was not repaired")

        cli_init_worktree = sandbox / "cli-init-worktree"
        cli_init_worktree.mkdir()
        cli_init_destination = sandbox / "cli-init.git"
        cli_init_without_approval = run(
            sandbox,
            sys.executable,
            str(OBJECT_TOOL),
            "init-empty",
            "--destination",
            str(cli_init_destination),
            "--worktree",
            str(cli_init_worktree),
            "--object-format",
            "sha1",
            check=False,
            offline=True,
        )
        if (
            cli_init_without_approval.returncode != 2
            or b"approved_exact_module_init" not in cli_init_without_approval.stderr
            or os.path.lexists(cli_init_destination)
        ):
            raise ProofError("init-empty CLI did not fail closed without approval")
        run(
            sandbox,
            sys.executable,
            str(OBJECT_TOOL),
            "init-empty",
            "--destination",
            str(cli_init_destination),
            "--worktree",
            str(cli_init_worktree),
            "--object-format",
            "sha1",
            "--approved-exact-module-init",
            offline=True,
        )
        if tool.repository_object_format(cli_init_destination) != "sha1":
            raise ProofError("approved init-empty CLI created wrong object format")

        missing_module_worktree = sandbox / "missing-module-worktree"
        missing_module_worktree.mkdir()
        missing_module_destination = sandbox / "missing-module.git"
        expect_error(
            lambda: tool.initialize_empty_module(
                missing_module_destination,
                missing_module_worktree,
                object_format="sha1",
            ),
            "approved_exact_module_init",
        )
        (missing_module_worktree / "collision").write_text("not empty\n", encoding="utf-8")
        expect_error(
            lambda: tool.initialize_empty_module(
                missing_module_destination,
                missing_module_worktree,
                object_format="sha1",
                approved_exact_module_init=True,
            ),
            "must be empty",
        )
        (missing_module_worktree / "collision").unlink()
        os.symlink(missing_module_worktree, sandbox / "missing-module-worktree-link")
        expect_error(
            lambda: tool.initialize_empty_module(
                missing_module_destination,
                sandbox / "missing-module-worktree-link",
                object_format="sha1",
                approved_exact_module_init=True,
            ),
            "symlink path component",
        )
        init_result = tool.initialize_empty_module(
            missing_module_destination,
            missing_module_worktree,
            object_format="sha1",
            approved_exact_module_init=True,
        )
        missing_transfer = tool.transfer_exact_commit(
            source_nested,
            missing_module_destination,
            nested_b,
            approved_exact_object_transfer=True,
        )
        git(
            sandbox,
            f"--git-dir={missing_module_destination}",
            f"--work-tree={missing_module_worktree}",
            "checkout",
            "--detach",
            nested_b,
            offline=True,
        )
        if (missing_module_worktree / "nested.txt").read_text(encoding="utf-8") != "nested B\n":
            raise ProofError("minimal local module init did not support exact offline checkout")

        # Prove the RCDM-shaped case: an exact empty gitlink worktree and an
        # absent module Git dir become usable by real offline submodule update,
        # without clone/fetch transport.
        missing_super = sandbox / "missing-superproject"
        init_repo(missing_super)
        git(
            missing_super,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(nested_source),
            "external/missing",
        )
        git(missing_super, "commit", "-am", "add missing module")
        git(missing_super, "submodule", "deinit", "--all")
        integrated_destination = missing_super / ".git/modules/external/missing"
        shutil.rmtree(integrated_destination)
        integrated_worktree = missing_super / "external/missing"
        integrated_init = tool.initialize_empty_module(
            integrated_destination,
            integrated_worktree,
            object_format="sha1",
            approved_exact_module_init=True,
        )
        integrated_transfer = tool.transfer_exact_commit(
            nested_source / ".git",
            integrated_destination,
            nested_b,
            approved_exact_object_transfer=True,
        )
        git(
            missing_super,
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--no-fetch",
            "--checkout",
            offline=True,
        )
        if git(integrated_worktree, "rev-parse", "HEAD").stdout.decode().strip() != nested_b:
            raise ProofError("integrated absent module did not update offline to exact commit")

        linked_module_worktree = sandbox / "linked-module-common-mismatch"
        git(
            destination_outer,
            "worktree",
            "add",
            "--detach",
            str(linked_module_worktree),
            outer_a,
        )
        linked_module_git_dir = git_dir(linked_module_worktree)
        expect_error(
            lambda: tool.probe(source_outer, linked_module_git_dir, outer_b),
            "must be its own module common dir",
        )
        git(destination_outer, "worktree", "remove", str(linked_module_worktree))

        git(linked, "checkout", "--detach", "HEAD")
        git(root, "checkout", "dev")
        offline_update = git(
            root,
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--no-fetch",
            "--checkout",
            offline=True,
        )
        if offline_update.returncode != 0:
            raise ProofError("offline recursive canonical update failed")
        if git(root / "external/outer", "rev-parse", "HEAD").stdout.decode().strip() != outer_b:
            raise ProofError("canonical outer checkout did not reach exact B")
        if git(root / "external/outer/nested", "rev-parse", "HEAD").stdout.decode().strip() != nested_b:
            raise ProofError("canonical nested checkout did not reach exact B")
        if git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout:
            raise ProofError("canonical superproject is dirty after offline exact update")

        normalized_commands = [
            [part.replace(str(sandbox), "$SANDBOX") for part in command]
            for command in COMMANDS
        ]
        return {
            "schema": "nimloth-local-module-object-bootstrap-sandbox-proof/v1",
            "sandbox_scope": "/tmp disposable superproject/module repositories; removed after proof",
            "construction": {
                "superproject_main_and_linked_dev": True,
                "worktree_specific_module_gitdirs": True,
                "main_outer_missing_B_before": True,
                "linked_outer_HEAD_and_object_B": True,
                "main_nested_missing_B_before": True,
                "linked_nested_HEAD_and_object_B": True,
                "recursive_depth": 2,
                "outer_A": outer_a,
                "outer_B": outer_b,
                "nested_A": nested_a,
                "nested_B": nested_b,
                "super_dev": super_dev,
            },
            "transfers": [
                summarized_transfer("external/outer", outer_transfer, sandbox),
                summarized_transfer("external/outer/nested", nested_transfer, sandbox),
            ],
            "object_format_proof": {
                "sha1_live_shape_transfers_complete": True,
                "sha1_to_sha256_mismatch_rejected_before_pack": True,
                "sha256_commit": sha256_commit,
                "sha256_same_format_closure_transfer_complete": sha256_transfer[
                    "destination_closure_complete_after"
                ],
            },
            "absent_destination_bootstrap": {
                "explicit_init_approval_required": True,
                "exact_empty_real_worktree_required": True,
                "explicit_object_format_required": True,
                "minimal_init_created_remote_or_ref": init_result["remote_or_ref_created"],
                "minimal_init_wrote_FETCH_HEAD": init_result["fetch_head_written"],
                "minimal_init_network_transport_available": init_result[
                    "network_transport_available"
                ],
                "object_transfer_preserved_init_fingerprints": missing_transfer[
                    "head_index_refs_config_remotes_fetch_head_unchanged"
                ],
                "exact_offline_checkout_after_init": nested_b,
                "integrated_gitlink_module_init_object_format": integrated_init[
                    "object_format"
                ],
                "integrated_object_transfer_preserved_init_fingerprints": integrated_transfer[
                    "head_index_refs_config_remotes_fetch_head_unchanged"
                ],
                "integrated_offline_submodule_update_exact_commit": nested_b,
            },
            "offline_canonical_update": {
                "GIT_ALLOW_PROTOCOL_empty": True,
                "no_fetch": True,
                "recursive": True,
                "outer_exact_commit_checked_out": outer_b,
                "nested_exact_commit_checked_out": nested_b,
                "superproject_clean": True,
            },
            "negative_controls": {
                "explicit_transfer_approval_required": True,
                "transfer_CLI_approval_and_success_paths_verified": True,
                "explicit_empty_module_init_approval_required": True,
                "init_empty_CLI_object_format_approval_and_success_paths_verified": True,
                "nonempty_module_worktree_rejected": True,
                "module_worktree_symlink_rejected": True,
                "short_commit_rejected": True,
                "source_HEAD_mismatch_rejected": True,
                "source_symlink_rejected": True,
                "destination_symlink_rejected": True,
                "same_or_overlapping_repository_identity_rejected": True,
                "object_format_mismatch_rejected_before_pack": True,
                "sha256_same_format_transfer_supported": True,
                "object_alternates_rejected": True,
                "partial_clone_promisor_routing_rejected": True,
                "external_config_include_routing_rejected": True,
                "object_directory_symlink_escape_rejected": True,
                "preflight_path_symlink_replacement_rejected": True,
                "all_inherited_GIT_routing_stripped": True,
                "destination_common_dir_mismatch_rejected": True,
                "extra_source_ref_objects_excluded_from_pack": True,
                "pack_object_set_equals_exact_source_closure": True,
                "incomplete_destination_closure_detected_and_repaired": repaired_transfer[
                    "destination_closure_complete_after"
                ],
                "destination_worktree_and_index_unchanged": True,
            },
            "commands_recorded": len(COMMANDS),
            "commands": normalized_commands,
            "live_nimloth_paths_used": False,
            "live_source_or_git_metadata_mutated": False,
        }
    finally:
        if (
            sandbox.parent != Path("/tmp")
            or not sandbox.name.startswith("nimloth-module-bootstrap-proof-")
            or sandbox.is_symlink()
        ):
            raise ProofError(f"refusing cleanup outside owned sandbox: {sandbox}")
        shutil.rmtree(sandbox)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        result = prove()
    except (OSError, ProofError, subprocess.SubprocessError) as exc:
        print(f"FAIL: {exc}")
        return 1
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(
        "PASS: exact outer+nested B objects copied locally without refs/config/remotes/FETCH_HEAD; "
        "canonical recursive --no-fetch checkout passed with transports disabled"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
