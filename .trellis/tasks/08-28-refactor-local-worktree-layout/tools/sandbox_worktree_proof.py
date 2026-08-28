#!/usr/bin/env python3
"""Reproducible RED/GREEN proof for the approved worktree migration method.

All repositories and payload copies live below a fresh /tmp directory.  The
proof deliberately has no force fallback and never edits .git/worktrees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

COMMANDS: list[list[str]] = []


class ProofError(RuntimeError):
    pass


def run(
    cwd: Path,
    *args: str,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    command = [*args]
    if any(
        argument in {"-f", "-ff", "--force"} or argument.startswith("--force=")
        for argument in command
    ):
        raise ProofError("force fallback is forbidden")
    if any(".git/worktrees" in argument.replace("\\", "/") for argument in command):
        raise ProofError("manual .git/worktrees access is forbidden")
    COMMANDS.append(["cwd=" + str(cwd), *command])
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    if check and completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace").strip()
        raise ProofError(f"command failed ({completed.returncode}): {command}: {stderr}")
    return completed


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return run(cwd, "git", *args, check=check)


def configure_identity(repo: Path) -> None:
    git(repo, "config", "user.name", "Sandbox Proof")
    git(repo, "config", "user.email", "sandbox-proof@example.invalid")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def payload_metadata(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        digest = hashlib.sha256(os.fsencode(target)).hexdigest()
        kind = "symlink"
    elif stat.S_ISREG(metadata.st_mode):
        digest = sha256(path)
        kind = "file"
    else:
        raise ProofError(f"unsupported payload kind: {path}")
    return {"kind": kind, "size_bytes": metadata.st_size, "sha256": digest}


def nul_paths(completed: subprocess.CompletedProcess[bytes]) -> list[str]:
    return [os.fsdecode(item) for item in completed.stdout.split(b"\0") if item]


def submodule_paths(repo: Path) -> list[str]:
    completed = git(repo, "submodule", "status", "--recursive")
    paths: list[str] = []
    for line in completed.stdout.decode("utf-8", "surrogateescape").splitlines():
        if line:
            _head, path, *_description = line[1:].split(" ")
            paths.append(path)
    return paths


def scope_paths(repo: Path) -> list[tuple[str, Path]]:
    scopes = [("root", repo)]
    scopes.extend((path, repo / path) for path in submodule_paths(repo))
    return scopes


def copy_payload(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise ProofError(f"archive destination already exists: {destination}")
    destination.mkdir(parents=True)
    manifest: dict[str, Any] = {"scopes": []}
    for index, (name, scope) in enumerate(scope_paths(source)):
        index_patch = git(
            scope,
            "diff",
            "--binary",
            "--full-index",
            "--cached",
            "--ignore-submodules=dirty",
            "HEAD",
            "--",
        ).stdout
        worktree_patch = git(
            scope,
            "diff",
            "--binary",
            "--full-index",
            "--ignore-submodules=dirty",
            "--",
        ).stdout
        patch_records: list[dict[str, Any]] = []
        for patch_kind, payload in (("index", index_patch), ("worktree", worktree_patch)):
            patch_path = destination / f"scope-{index}-{patch_kind}.patch"
            patch_path.write_bytes(payload)
            patch_records.append(
                {
                    "kind": patch_kind,
                    "path": patch_path.name,
                    "sha256": sha256(patch_path),
                    "size_bytes": patch_path.stat().st_size,
                }
            )
        untracked = set(
            nul_paths(git(scope, "ls-files", "--others", "--exclude-standard", "-z"))
        )
        ignored = set(
            nul_paths(
                git(scope, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")
            )
        )
        extras: list[dict[str, Any]] = []
        for relative in sorted(untracked | ignored):
            payload = scope / relative
            if not os.path.lexists(payload):
                raise ProofError(f"payload disappeared during capture: {payload}")
            target = destination / "files" / str(index) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            metadata = payload_metadata(payload)
            if metadata["kind"] == "symlink":
                os.symlink(os.readlink(payload), target)
            else:
                shutil.copy2(payload, target, follow_symlinks=False)
            extras.append(
                {
                    "path": relative,
                    **metadata,
                    "status": "ignored" if relative in ignored else "untracked",
                }
            )
        manifest["scopes"].append(
            {"name": name, "patches": patch_records, "extras": extras}
        )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def remove_exact_payload(scope: Path, relative: str) -> None:
    target = scope / relative
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        raise ProofError(f"refusing recursive deletion for payload path: {target}")
    parent = target.parent
    while parent != scope:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def clean_from_manifest(source: Path, manifest: dict[str, Any]) -> None:
    scopes = {name: path for name, path in scope_paths(source)}
    # Nested repositories first; the superproject patches ignore dirty-only gitlinks.
    for entry in reversed(manifest["scopes"]):
        scope = scopes[entry["name"]]
        git(scope, "restore", "--source=HEAD", "--staged", "--worktree", "--", ".")
        for extra in entry["extras"]:
            remove_exact_payload(scope, extra["path"])
        status = git(scope, "status", "--porcelain=v1", "--untracked-files=all").stdout
        if status:
            raise ProofError(f"scope did not become clean: {entry['name']}")


def restore_payload(destination: Path, archive: Path, manifest: dict[str, Any]) -> None:
    scopes = {name: path for name, path in scope_paths(destination)}
    for index, entry in enumerate(manifest["scopes"]):
        scope = scopes[entry["name"]]
        patches = {patch["kind"]: patch for patch in entry["patches"]}
        for patch_kind in ("index", "worktree"):
            record = patches[patch_kind]
            patch = archive / record["path"]
            if patch.stat().st_size != record["size_bytes"] or sha256(patch) != record["sha256"]:
                raise ProofError(f"archive patch integrity mismatch: {patch}")
            if patch.stat().st_size:
                arguments = ["git", "apply", "--binary"]
                if patch_kind == "index":
                    arguments.append("--index")
                run(scope, *arguments, str(patch))
        for extra in entry["extras"]:
            source = archive / "files" / str(index) / extra["path"]
            target = scope / extra["path"]
            if os.path.lexists(target):
                raise ProofError(f"refusing to overwrite destination payload: {target}")
            if payload_metadata(source) != {
                key: extra[key] for key in ("kind", "size_bytes", "sha256")
            }:
                raise ProofError(f"archive payload integrity mismatch: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if extra["kind"] == "symlink":
                os.symlink(os.readlink(source), target)
            else:
                shutil.copy2(source, target, follow_symlinks=False)
            if payload_metadata(target) != {
                key: extra[key] for key in ("kind", "size_bytes", "sha256")
            }:
                raise ProofError(f"restored payload metadata/hash mismatch: {target}")


def status_snapshot(repo: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name, scope in scope_paths(repo):
        status = git(scope, "status", "--porcelain=v1", "-z", "--untracked-files=all").stdout
        index_patch = git(
            scope,
            "diff",
            "--binary",
            "--full-index",
            "--cached",
            "--ignore-submodules=dirty",
            "HEAD",
            "--",
        ).stdout
        worktree_patch = git(
            scope,
            "diff",
            "--binary",
            "--full-index",
            "--ignore-submodules=dirty",
            "--",
        ).stdout
        untracked = set(
            nul_paths(git(scope, "ls-files", "--others", "--exclude-standard", "-z"))
        )
        ignored = set(
            nul_paths(
                git(scope, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")
            )
        )
        extras: list[dict[str, Any]] = []
        for relative in sorted(untracked | ignored):
            # Canonical .local is machine-owned state, not branch payload; it
            # intentionally exists only at the destination root.
            if relative == ".local" or relative.startswith(".local/"):
                continue
            extras.append(
                {
                    "path": relative,
                    "status": "ignored" if relative in ignored else "untracked",
                    **payload_metadata(scope / relative),
                }
            )
        snapshot[name] = {
            "status_hex": status.hex(),
            "index_patch_sha256": sha256_bytes(index_patch),
            "index_patch_size_bytes": len(index_patch),
            "worktree_patch_sha256": sha256_bytes(worktree_patch),
            "worktree_patch_size_bytes": len(worktree_patch),
            "extras": extras,
        }
    return snapshot


def status_fingerprint(snapshot: dict[str, Any]) -> str:
    rendered = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(rendered).hexdigest()


def create_submodule_source(path: Path) -> None:
    path.mkdir()
    git(path, "init", "-b", "main")
    configure_identity(path)
    (path / ".gitignore").write_text(".cache/\n", encoding="utf-8")
    (path / "model.txt").write_text("base model\n", encoding="utf-8")
    git(path, "add", ".gitignore", "model.txt")
    git(path, "commit", "-m", "submodule base")


def create_main_repo(path: Path) -> None:
    path.mkdir()
    git(path, "init", "-b", "main")
    configure_identity(path)
    (path / ".gitignore").write_text(".worktree\n.local/\n.cache/\n", encoding="utf-8")
    (path / "README.md").write_text("main\n", encoding="utf-8")
    git(path, "add", ".gitignore", "README.md")
    git(path, "commit", "-m", "main base")
    git(path, "branch", "dev")
    (path / ".local").mkdir()
    (path / ".local" / "SERVER.md").write_text("sandbox local state\n", encoding="utf-8")


def prove() -> dict[str, Any]:
    COMMANDS.clear()
    sandbox = Path(tempfile.mkdtemp(prefix="nimloth-worktree-proof-", dir="/tmp"))
    cleaned = False
    try:
        root = sandbox / "nimloth"
        linked_dev = sandbox / "nimloth-dev"
        dirty_only = sandbox / "dirty-remove-proof"
        archive = sandbox / "payload-archive"
        submodule_source = sandbox / "submodule-source"
        create_submodule_source(submodule_source)
        create_main_repo(root)

        # Isolate the normal dirty-tree refusal from Git's separate submodule
        # refusal by using a disposable submodule-free linked worktree.
        git(root, "worktree", "add", "-b", "proof/dirty-remove", str(dirty_only), "main")
        (dirty_only / "README.md").write_text("dirty remove proof\n", encoding="utf-8")
        red_dirty_remove = git(root, "worktree", "remove", str(dirty_only), check=False)
        if red_dirty_remove.returncode == 0 or not dirty_only.exists():
            raise ProofError("RED remove unexpectedly accepted a dirty linked worktree")
        git(dirty_only, "restore", "--worktree", "--", "README.md")
        git(root, "worktree", "remove", str(dirty_only))

        git(root, "worktree", "add", str(linked_dev), "dev")
        git(
            linked_dev,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(submodule_source),
            "external/demo",
        )
        git(linked_dev, "commit", "-am", "add sandbox submodule")

        # Exercise staged + unstaged tracked bytes, untracked/ignored files,
        # a symlink handoff, and the same classes inside a nested submodule.
        readme = linked_dev / "README.md"
        readme.write_text("dev staged payload\n", encoding="utf-8")
        git(linked_dev, "add", "README.md")
        readme.write_text("dev staged plus unstaged payload\n", encoding="utf-8")
        task_file = linked_dev / ".trellis/tasks/proof/progress.md"
        task_file.parent.mkdir(parents=True)
        task_file.write_text("handoff survives\n", encoding="utf-8")
        os.symlink("progress.md", task_file.parent / "latest")
        cache_file = linked_dev / ".cache/runtime.bin"
        cache_file.parent.mkdir()
        cache_file.write_bytes(b"runtime payload\x00")
        nested = linked_dev / "external/demo"
        nested_model = nested / "model.txt"
        nested_model.write_text("staged model\n", encoding="utf-8")
        git(nested, "add", "model.txt")
        nested_model.write_text("staged plus unstaged model\n", encoding="utf-8")
        (nested / "notes.txt").write_text("untracked nested\n", encoding="utf-8")
        (nested / ".cache").mkdir()
        (nested / ".cache/state.bin").write_bytes(b"nested ignored\x00")
        before_snapshot = status_snapshot(linked_dev)
        before = status_fingerprint(before_snapshot)

        red_checkout = git(root, "checkout", "dev", check=False)
        if red_checkout.returncode == 0:
            raise ProofError("RED checkout unexpectedly succeeded while dev was linked")

        manifest = copy_payload(linked_dev, archive)
        clean_from_manifest(linked_dev, manifest)
        # Current Git rejects ordinary removal for a clean worktree whose tree
        # contains a submodule, even after deinit.  This exact refusal remains a
        # stop gate; releasing the branch only needs a clean detach.
        git(linked_dev, "submodule", "deinit", "--all")
        if git(linked_dev, "status", "--porcelain=v1", "--untracked-files=all").stdout:
            raise ProofError("linked worktree became dirty after submodule deinit")
        red_submodule_remove = git(root, "worktree", "remove", str(linked_dev), check=False)
        refusal = red_submodule_remove.stderr.decode("utf-8", "replace")
        if (
            red_submodule_remove.returncode == 0
            or not linked_dev.exists()
            or "containing submodules" not in refusal
        ):
            raise ProofError("submodule worktree removal behavior changed; review the method")
        git(linked_dev, "checkout", "--detach", "HEAD")

        git(root, "checkout", "dev")
        git(root, "-c", "protocol.file.allow=always", "submodule", "update", "--init", "--recursive")
        restore_payload(root, archive, manifest)
        after_snapshot = status_snapshot(root)
        after = status_fingerprint(after_snapshot)
        if before != after:
            raise ProofError(
                "payload status/hash fingerprint changed after restoration: "
                + json.dumps({"before": before_snapshot, "after": after_snapshot}, sort_keys=True)
            )
        if task_file.exists():
            raise ProofError("cleaned linked path unexpectedly retained task payload")
        restored_task = root / ".trellis/tasks/proof/progress.md"
        if restored_task.read_text(encoding="utf-8") != "handoff survives\n":
            raise ProofError("task runtime/handoff payload was not restored")

        # Prove rollback rather than merely describing it: clean the destination
        # from the same exact manifest, release dev there, reattach the retained
        # linked path, and restore the identical staged/unstaged payload.
        clean_from_manifest(root, manifest)
        git(root, "submodule", "deinit", "--all")
        git(root, "checkout", "--detach", "HEAD")
        git(linked_dev, "checkout", "dev")
        git(
            linked_dev,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
            "--recursive",
        )
        restore_payload(linked_dev, archive, manifest)
        rollback_snapshot = status_snapshot(linked_dev)
        rollback = status_fingerprint(rollback_snapshot)
        if before != rollback:
            raise ProofError(
                "rollback payload status/hash fingerprint changed: "
                + json.dumps(
                    {"before": before_snapshot, "rollback": rollback_snapshot}, sort_keys=True
                )
            )
        if task_file.read_text(encoding="utf-8") != "handoff survives\n":
            raise ProofError("task handoff was not restored on rollback path")

        child = root / ".worktree/feat-proof"
        # Use the sandbox's submodule-free main start point to prove ordinary
        # clean cleanup.  A child whose tree contains submodules is covered by
        # the fail-closed RED gate above and requires a separate exact decision.
        git(root, "worktree", "add", "-b", "feat/proof", str(child), "main")
        os.symlink(root / ".local", child / ".local")
        actual_top = git(child, "rev-parse", "--show-toplevel").stdout.decode().strip()
        actual_branch = git(child, "branch", "--show-current").stdout.decode().strip()
        actual_common = (
            git(child, "rev-parse", "--path-format=absolute", "--git-common-dir")
            .stdout.decode()
            .strip()
        )
        if Path(actual_top) != child or actual_branch != "feat/proof":
            raise ProofError("nested child path/branch verification failed")
        if Path(os.path.realpath(child / ".local")) != root / ".local":
            raise ProofError("nested child .local does not resolve to canonical owner")
        if Path(actual_common) != root / ".git":
            raise ProofError("nested child common Git dir is not canonical root .git")
        if Path.cwd() == child:
            raise ProofError("proof incorrectly relied on process cwd changing")
        (child / ".local").unlink()
        git(root, "worktree", "remove", str(child))
        if child.exists():
            raise ProofError("clean nested child path still exists after cleanup")
        registrations = git(root, "worktree", "list", "--porcelain").stdout.decode()
        if str(child) in registrations:
            raise ProofError("cleaned nested child remains registered")

        normalized_commands = [
            [argument.replace(str(sandbox), "$SANDBOX") for argument in command]
            for command in COMMANDS
        ]
        result = {
            "schema": "nimloth-worktree-sandbox-proof/v1",
            "sandbox_scope": "/tmp disposable repository; removed after proof",
            "red": {
                "checkout_dev_while_linked_rejected": True,
                "dirty_submodule_free_linked_remove_rejected": True,
                "ordinary_remove_with_submodule_rejected_after_clean_deinit": True,
                "submodule_refusal_message_verified": True,
                "force_fallback_not_attempted": True,
            },
            "green": {
                "metadata_only_payload_manifest": True,
                "staged_and_unstaged_index_state_restored": True,
                "clean_branch_release_by_detach_without_force": True,
                "main_worktree_checkout_dev": True,
                "tracked_untracked_ignored_symlink_submodule_payload_restored": True,
                "task_handoff_restored": True,
                "rollback_to_retained_linked_path_verified": True,
                "nested_child_created_and_verified": True,
                "child_local_resolves_to_canonical_root": True,
                "clean_child_cleanup": True,
            },
            "before_status_payload_sha256": before,
            "after_status_payload_sha256": after,
            "rollback_status_payload_sha256": rollback,
            "payload_scope_count": len(manifest["scopes"]),
            "payload_patch_count": sum(len(item["patches"]) for item in manifest["scopes"]),
            "payload_extra_count": sum(len(item["extras"]) for item in manifest["scopes"]),
            "commands_recorded": len(COMMANDS),
            "commands": normalized_commands,
            "force_fallback_used": False,
            "manual_git_worktrees_edit_used": False,
            "command_policy_check": "every command had an explicit disposable cwd; -f/--force and .git/worktrees arguments fail closed",
        }
        cleaned = True
        return result
    finally:
        # This recursive cleanup applies only to the newly-created disposable
        # /tmp sandbox, never to a live Nimloth path.
        shutil.rmtree(sandbox, ignore_errors=False)
        if cleaned and sandbox.exists():
            raise ProofError("disposable sandbox cleanup failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="write structured proof evidence to this path")
    args = parser.parse_args()
    try:
        result = prove()
    except (OSError, ProofError) as exc:
        print(f"FAIL: {exc}")
        return 1
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(
        "PASS: RED checkout/dirty-remove/submodule-remove rejected; GREEN staged+unstaged "
        "payload restore, clean detach release, rollback, nested child .local verification, and "
        "ordinary cleanup succeeded; force_fallback=false manual_git_worktrees_edit=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
