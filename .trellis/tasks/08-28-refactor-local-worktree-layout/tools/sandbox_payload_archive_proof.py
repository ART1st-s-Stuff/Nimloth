#!/usr/bin/env python3
"""Disposable end-to-end and negative proof for payload_archive.py.

The proof creates all repositories and archives below a fresh /tmp directory.
It never points clean/restore at a live Nimloth worktree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PAYLOAD_TOOL = HERE / "payload_archive.py"
COMMANDS: list[list[str]] = []


class ProofError(RuntimeError):
    pass


def load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("payload_archive", PAYLOAD_TOOL)
    if spec is None or spec.loader is None:
        raise ProofError("cannot load payload_archive.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    COMMANDS.append(["cwd=" + str(cwd), *args])
    completed = subprocess.run(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={
            **os.environ,
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
        },
    )
    if check and completed.returncode != 0:
        raise ProofError(
            f"command failed at {cwd}: {args}: "
            + completed.stderr.decode("utf-8", "replace")
        )
    return completed


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return run(cwd, "git", *args)


def identity(repo: Path) -> None:
    git(repo, "config", "user.name", "Payload Sandbox")
    git(repo, "config", "user.email", "payload@example.invalid")


def init_repo(path: Path) -> None:
    path.mkdir()
    git(path, "init", "-b", "main")
    identity(path)
    (path / ".gitignore").write_text(
        ".cache/\n.local/\n.venv/\nignored-repo/\n.special/\n", encoding="utf-8"
    )
    (path / "tracked.txt").write_text("base\n", encoding="utf-8")
    (path / "rename-me.txt").write_text("rename-base\n", encoding="utf-8")
    git(path, "add", ".gitignore", "tracked.txt", "rename-me.txt")
    git(path, "commit", "-m", "base")
    (path / ".local").mkdir()
    (path / ".local/SECRET").write_text("machine-only\n", encoding="utf-8")


def dirty(repo: Path, *, prefix: str = "") -> None:
    tracked = repo / "tracked.txt"
    tracked.write_text(prefix + "staged\n", encoding="utf-8")
    git(repo, "add", "tracked.txt")
    tracked.write_text(prefix + "staged+unstaged\n", encoding="utf-8")
    git(repo, "mv", "rename-me.txt", "renamed.txt")
    (repo / "renamed.txt").write_text(prefix + "renamed-unstaged\n", encoding="utf-8")
    payload = repo / "payload dir/extra.txt"
    payload.parent.mkdir()
    payload.write_text(prefix + "untracked\n", encoding="utf-8")
    os.symlink("extra.txt", payload.parent / "extra-link")
    ignored = repo / ".cache/state.bin"
    ignored.parent.mkdir()
    ignored.write_bytes((prefix + "ignored").encode() + b"\0")

    embedded = repo / "ignored-repo"
    (embedded / ".git/objects/pack").mkdir(parents=True)
    (embedded / ".git/refs/heads").mkdir(parents=True)
    (embedded / ".git/HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (embedded / ".git/config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tbare = false\n", encoding="utf-8"
    )
    (embedded / "empty-dir").mkdir()
    (embedded / "payload.bin").write_bytes((prefix + "embedded").encode() + b"\0")
    os.chmod(embedded, 0o750)
    os.chmod(embedded / "empty-dir", 0o711)
    outside = repo.parent / (repo.name + "-outside-target")
    outside.mkdir()
    (outside / "MUST_NOT_ARCHIVE").write_text("outside\n", encoding="utf-8")
    os.symlink(outside, embedded / "outside-dir-link")


def add_disposable_venv_sentinel(repo: Path) -> Path:
    sentinel = repo / ".venv/UNSUPPORTED-HARDLINK-MUST-NOT-BE-READ"
    sentinel.parent.mkdir()
    sentinel.write_text("excluded disposable environment\n", encoding="utf-8")
    os.link(sentinel, sentinel.parent / "UNSUPPORTED-HARDLINK-ALIAS")
    return sentinel


def normalized_symlink_target(path: Path) -> str:
    return re.sub(
        r"/tmp/nimloth-payload-archive-proof-[^/]+",
        "$SANDBOX",
        os.readlink(path),
    )


def tree_fingerprint(root: Path) -> bytes:
    rows: list[tuple[str, str, int, str]] = []
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        kept: list[str] = []
        for name in sorted(names):
            path = base / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                rows.append((relative, "symlink", 0, normalized_symlink_target(path)))
            else:
                rows.append((relative, "directory", path.stat().st_mode & 0o7777, ""))
                kept.append(name)
        names[:] = kept
        for name in sorted(files):
            path = base / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                rows.append((relative, "symlink", 0, normalized_symlink_target(path)))
            else:
                rows.append(
                    (
                        relative,
                        "file",
                        path.stat().st_mode & 0o7777,
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
    return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()


def payload_fingerprint(repo: Path) -> str:
    digest = hashlib.sha256()
    for args in (
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
        ("diff", "--binary", "--cached", "HEAD"),
        ("diff", "--binary"),
    ):
        digest.update(git(repo, *args).stdout)
    for relative in ("payload dir/extra.txt", ".cache/state.bin"):
        digest.update((repo / relative).read_bytes())
    digest.update(os.fsencode(os.readlink(repo / "payload dir/extra-link")))
    digest.update(tree_fingerprint(repo / "ignored-repo"))
    return digest.hexdigest()


def expect_error(call: Any, contains: str) -> None:
    try:
        call()
    except Exception as exc:  # the proof checks exact fail-closed semantics
        if contains not in str(exc):
            raise ProofError(f"expected error containing {contains!r}, got: {exc}") from exc
        return
    raise ProofError(f"expected failure containing {contains!r}")


def rewrite_manifest(tool: Any, archive: Path, mutate: Any) -> None:
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["fingerprint_sha256"] = tool.archive_fingerprint(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def capture_with_venv_pre_read_guard(
    tool: Any,
    repo: Path,
    archive: Path,
) -> dict[str, Any]:
    """Fail if Python payload code touches root .venv before exclusion."""

    venv = repo / ".venv"

    def is_venv_path(value: Any) -> bool:
        if isinstance(value, int):
            return False
        try:
            candidate = Path(os.path.abspath(os.fspath(value)))
            return candidate == venv or venv in candidate.parents
        except TypeError:
            return False

    original_lstat = os.lstat
    original_scandir = os.scandir
    original_sha256_file = tool.sha256_file
    original_copy_extra = tool.copy_extra

    def guarded_lstat(path: Any, *args: Any, **kwargs: Any) -> Any:
        if is_venv_path(path):
            raise ProofError(f".venv reached lstat before exclusion: {path}")
        return original_lstat(path, *args, **kwargs)

    def guarded_scandir(path: Any = ".") -> Any:
        if is_venv_path(path):
            raise ProofError(f".venv reached scandir before exclusion: {path}")
        return original_scandir(path)

    def guarded_sha256_file(path: Path) -> str:
        if is_venv_path(path):
            raise ProofError(f".venv reached hash before exclusion: {path}")
        return original_sha256_file(path)

    def guarded_copy_extra(source: Path, *args: Any, **kwargs: Any) -> None:
        if is_venv_path(source):
            raise ProofError(f".venv reached copy before exclusion: {source}")
        original_copy_extra(source, *args, **kwargs)

    os.lstat = guarded_lstat
    os.scandir = guarded_scandir
    tool.sha256_file = guarded_sha256_file
    tool.copy_extra = guarded_copy_extra
    try:
        return tool.capture(
            repo,
            archive,
            include_ignored=True,
            root_excludes=(".venv",),
        )
    finally:
        os.lstat = original_lstat
        os.scandir = original_scandir
        tool.sha256_file = original_sha256_file
        tool.copy_extra = original_copy_extra


def prove_root_case(tool: Any, sandbox: Path) -> dict[str, Any]:
    repo = sandbox / "root-case-repo"
    archive = sandbox / "root-case-archive"
    init_repo(repo)
    dirty(repo)
    venv_sentinel = add_disposable_venv_sentinel(repo)
    before = payload_fingerprint(repo)

    expect_error(
        lambda: tool.capture(repo, sandbox / "missing-ignored", include_ignored=False, root_excludes=()),
        "--include-ignored",
    )
    expect_error(
        lambda: tool.capture(
            repo,
            sandbox / "unsupported-exclusion",
            include_ignored=True,
            root_excludes=("payload dir",),
        ),
        "forbids unsupported root exclusions",
    )
    expect_error(
        lambda: tool.capture(
            repo,
            sandbox / "venv-not-excluded",
            include_ignored=True,
            root_excludes=(),
        ),
        "hardlinked payload",
    )
    inside_parent = repo / ".trellis/task-artifact"
    inside_parent.mkdir(parents=True)
    expect_error(
        lambda: tool.capture(
            repo,
            inside_parent / "archive",
            include_ignored=True,
            root_excludes=(),
        ),
        "outside registered worktrees",
    )
    inside_parent.rmdir()
    inside_parent.parent.rmdir()

    real_archive_parent = sandbox / "real-archive-parent"
    real_archive_parent.mkdir()
    os.symlink(real_archive_parent, sandbox / "archive-parent-link")
    expect_error(
        lambda: tool.capture(
            repo,
            sandbox / "archive-parent-link/archive",
            include_ignored=True,
            root_excludes=(),
        ),
        "symlink path component",
    )

    special = repo / "ignored-repo"
    os.mkfifo(special / "fifo")
    expect_error(
        lambda: tool.capture(
            repo,
            sandbox / "unsupported-special",
            include_ignored=True,
            root_excludes=(".venv",),
        ),
        "socket/device/FIFO",
    )
    (special / "fifo").unlink()
    os.link(repo / "ignored-repo/payload.bin", special / "hardlink")
    expect_error(
        lambda: tool.capture(
            repo,
            sandbox / "unsupported-hardlink",
            include_ignored=True,
            root_excludes=(".venv",),
        ),
        "hardlinked payload",
    )
    (special / "hardlink").unlink()

    local_mode = stat.S_IMODE((repo / ".local").stat().st_mode)
    os.chmod(repo / ".local", 0)
    try:
        manifest = capture_with_venv_pre_read_guard(tool, repo, archive)
    finally:
        os.chmod(repo / ".local", local_mode)
    if tool.validate_archive(archive):
        raise ProofError("fresh root archive did not validate")
    cli_archive = sandbox / "root-case-cli-archive"
    run(
        repo,
        sys.executable,
        str(PAYLOAD_TOOL),
        "capture",
        "--source",
        str(repo),
        "--archive",
        str(cli_archive),
        "--include-ignored",
        "--exclude-root-prefix",
        ".venv",
    )
    run(
        repo,
        sys.executable,
        str(PAYLOAD_TOOL),
        "validate",
        "--archive",
        str(cli_archive),
        "--live-source",
        str(repo),
    )
    cli_manifest = json.loads((cli_archive / "manifest.json").read_text(encoding="utf-8"))
    if cli_manifest["policy"]["root_excluded_prefixes"] != [".local", ".venv"]:
        raise ProofError("capture CLI did not persist exact no-venv policy")
    excluded_prefixes = tuple(manifest["policy"]["root_excluded_prefixes"])
    if excluded_prefixes != (".local", ".venv"):
        raise ProofError("explicit .venv exclusion was not normalized after mandatory .local")
    if not manifest["policy"].get("disposable_venv_exclusion"):
        raise ProofError("archive policy omitted explicit disposable .venv exclusion")
    for scope in manifest["scopes"]:
        for record in [*scope["extras"], *scope["directories"]]:
            if record["path"].startswith((".local", ".venv")):
                raise ProofError("excluded root payload entered archive manifest")
    if (archive / "files/0/.local").exists() or (archive / "files/0/.venv").exists():
        raise ProofError("excluded root payload entered archive")
    root_scope = manifest["scopes"][0]
    archived_leaves = {entry["path"] for entry in root_scope["extras"]}
    archived_directories = {entry["path"]: entry for entry in root_scope["directories"]}
    required_leaves = {
        "ignored-repo/.git/HEAD",
        "ignored-repo/.git/config",
        "ignored-repo/payload.bin",
        "ignored-repo/outside-dir-link",
    }
    required_directories = {
        "ignored-repo",
        "ignored-repo/.git",
        "ignored-repo/.git/objects/pack",
        "ignored-repo/.git/refs/heads",
        "ignored-repo/empty-dir",
    }
    if not required_leaves.issubset(archived_leaves):
        raise ProofError("embedded ignored Git repo leaves were not expanded")
    if not required_directories.issubset(archived_directories):
        raise ProofError("ignored/empty directory records were not captured")
    if archived_directories["ignored-repo"]["mode"] != 0o750:
        raise ProofError("ignored root directory mode was not captured")
    if archived_directories["ignored-repo/empty-dir"]["mode"] != 0o711:
        raise ProofError("empty directory mode was not captured")
    if any(path.name == "MUST_NOT_ARCHIVE" for path in archive.rglob("*")):
        raise ProofError("archive followed a directory symlink outside the source")

    expect_error(lambda: tool.clean(repo, archive), "approved_exact_clean")
    expect_error(lambda: tool.restore(repo, archive), "approved_exact_restore")
    cli_clean = run(
        repo,
        sys.executable,
        str(PAYLOAD_TOOL),
        "clean",
        "--source",
        str(repo),
        "--archive",
        str(archive),
        check=False,
    )
    if cli_clean.returncode == 0 or b"approved_exact_clean" not in cli_clean.stderr:
        raise ProofError("CLI clean did not require explicit approval")
    cli_restore = run(
        repo,
        sys.executable,
        str(PAYLOAD_TOOL),
        "restore",
        "--destination",
        str(repo),
        "--archive",
        str(archive),
        check=False,
    )
    if cli_restore.returncode == 0 or b"approved_exact_restore" not in cli_restore.stderr:
        raise ProofError("CLI restore did not require explicit approval")
    expect_error(lambda: tool.run_git(repo, "clean", "-n"), "not implemented")
    expect_error(lambda: tool.run_git(repo, "reset", "--hard"), "not implemented")
    expect_error(lambda: tool.run_git(repo, "stash"), "not implemented")
    expect_error(lambda: tool.run_git(repo, "checkout", "--force"), "force")
    expect_error(lambda: tool.run_git(repo, "status", ".git/worktrees/x"), ".git/worktrees")

    malformed_head = sandbox / "tampered-malformed-head"
    shutil.copytree(archive, malformed_head, symlinks=True)
    rewrite_manifest(
        tool,
        malformed_head,
        lambda value: value["source"].update({"head": "a" * 41}),
    )
    malformed_head_errors = tool.validate_archive(malformed_head)
    if not malformed_head_errors or not any(
        "source Git/filesystem identity is malformed" in error
        for error in malformed_head_errors
    ):
        raise ProofError(f"41-hex malformed source HEAD was not rejected: {malformed_head_errors}")

    traversal = sandbox / "tampered-traversal"
    shutil.copytree(archive, traversal, symlinks=True)
    rewrite_manifest(
        tool,
        traversal,
        lambda value: value["scopes"][0]["extras"][0].update(
            {"path": "../escape", "archive_path": "files/0/../escape"}
        ),
    )
    traversal_errors = tool.validate_archive(traversal)
    if not traversal_errors or not any("traversal" in error for error in traversal_errors):
        raise ProofError(f"traversal archive was not rejected: {traversal_errors}")

    symlink_patch = sandbox / "tampered-symlink-patch"
    shutil.copytree(archive, symlink_patch, symlinks=True)
    patch = symlink_patch / "patches/scope-0-index.patch"
    patch.unlink()
    os.symlink("/etc/passwd", patch)
    symlink_errors = tool.validate_archive(symlink_patch)
    if not symlink_errors or not any("regular non-symlink" in error for error in symlink_errors):
        raise ProofError(f"symlink patch was not rejected: {symlink_errors}")

    unknown = sandbox / "tampered-unknown-leaf"
    shutil.copytree(archive, unknown, symlinks=True)
    (unknown / "UNLISTED").write_text("must reject\n", encoding="utf-8")
    unknown_errors = tool.validate_archive(unknown)
    if not unknown_errors or not any("unlisted" in error for error in unknown_errors):
        raise ProofError(f"unlisted archive leaf was not rejected: {unknown_errors}")

    malicious_patch_archive = sandbox / "tampered-patch-payload-path"
    shutil.copytree(archive, malicious_patch_archive, symlinks=True)
    malicious_patch = malicious_patch_archive / "patches/scope-0-index.patch"
    malicious_patch.write_bytes(
        b"diff --git a/.local/ATTACK b/.local/ATTACK\n"
        b"new file mode 100644\n--- /dev/null\n+++ b/.local/ATTACK\n"
        b"@@ -0,0 +1 @@\n+attack\n"
    )
    def update_malicious_patch(value: dict[str, Any]) -> None:
        record = value["scopes"][0]["patches"]["index"]
        record["size_bytes"] = malicious_patch.stat().st_size
        record["sha256"] = tool.sha256_file(malicious_patch)

    rewrite_manifest(tool, malicious_patch_archive, update_malicious_patch)
    if tool.validate_archive(malicious_patch_archive):
        raise ProofError("structurally valid malicious patch archive did not validate")

    (repo / "late-unlisted.txt").write_text("writer race\n", encoding="utf-8")
    expect_error(
        lambda: tool.clean(repo, archive, approved_exact_clean=True),
        "live payload differs",
    )
    if not (repo / "late-unlisted.txt").exists() or (repo / "tracked.txt").read_text() == "base\n":
        raise ProofError("failed clean changed source before rejecting unlisted payload")
    (repo / "late-unlisted.txt").unlink()

    tool.clean(repo, archive, approved_exact_clean=True)
    if not (repo / ".local/SECRET").is_file():
        raise ProofError("clean removed .local")
    if not venv_sentinel.is_file() or venv_sentinel.stat().st_nlink != 2:
        raise ProofError("clean removed or read/replaced excluded source .venv sentinel")
    if (repo / "payload dir/extra.txt").exists() or (repo / ".cache/state.bin").exists():
        raise ProofError("clean retained listed payload")
    # Parent directories are deliberately retained; clean never recursively
    # removes or rmdirs unlisted filesystem entries.
    if not (repo / "payload dir").is_dir() or not (repo / ".cache").is_dir():
        raise ProofError("clean removed an unlisted parent directory")
    for relative in required_directories:
        if not (repo / relative).is_dir():
            raise ProofError(f"clean did not preserve listed directory: {relative}")
    if (repo / "ignored-repo/.git/HEAD").exists() or (
        repo / "ignored-repo/outside-dir-link"
    ).exists():
        raise ProofError("clean retained listed ignored-directory leaves")
    if stat.S_IMODE((repo / "ignored-repo").stat().st_mode) != 0o750:
        raise ProofError("clean changed ignored directory mode")
    if stat.S_IMODE((repo / "ignored-repo/empty-dir").stat().st_mode) != 0o711:
        raise ProofError("clean changed empty directory mode")
    (repo / "ignored-repo/unlisted-empty").mkdir()
    expect_error(
        lambda: tool.restore(repo, archive, approved_exact_restore=True),
        "unlisted existing directory",
    )
    if (repo / "ignored-repo/.git/HEAD").exists():
        raise ProofError("restore mutated before rejecting unlisted empty directory")
    (repo / "ignored-repo/unlisted-empty").rmdir()

    outside = sandbox / "symlink-parent-outside"
    outside.mkdir()
    (repo / "payload dir").rmdir()
    os.symlink(outside, repo / "payload dir")
    expect_error(
        lambda: tool.safe_target(
            repo,
            "payload dir/extra.txt",
            create_parents=False,
            allow_missing_parents=True,
        ),
        "not a real directory",
    )
    expect_error(
        lambda: tool.restore(repo, archive, approved_exact_restore=True),
        "not exact-clean",
    )
    if (outside / "extra.txt").exists():
        raise ProofError("restore wrote through a destination symlink parent")
    (repo / "payload dir").unlink()
    (repo / "payload dir").mkdir()

    (repo / "payload dir/extra.txt").write_text("collision\n", encoding="utf-8")
    expect_error(
        lambda: tool.restore(repo, archive, approved_exact_restore=True),
        "not exact-clean",
    )
    (repo / "payload dir/extra.txt").unlink()
    expect_error(
        lambda: tool.restore(
            repo,
            malicious_patch_archive,
            approved_exact_restore=True,
        ),
        "not covered by listed tracked paths",
    )
    if (repo / ".local/ATTACK").exists():
        raise ProofError("restore applied an unlisted/.local patch payload path")
    tool.restore(repo, archive, approved_exact_restore=True)
    after = payload_fingerprint(repo)
    if before != after or not (repo / ".local/SECRET").is_file():
        raise ProofError("root roundtrip fingerprint/.local changed")
    if not venv_sentinel.is_file() or venv_sentinel.stat().st_nlink != 2:
        raise ProofError("same-worktree restore changed excluded source .venv")
    if stat.S_IMODE((repo / "ignored-repo").stat().st_mode) != 0o750 or stat.S_IMODE(
        (repo / "ignored-repo/empty-dir").stat().st_mode
    ) != 0o711:
        raise ProofError("restore did not reconstruct ignored directory modes")
    return {
        "fingerprint_before": before,
        "fingerprint_after": after,
        "archive_integrity_validated": True,
        "scope_count": len(manifest["scopes"]),
        "extra_count": sum(len(scope["extras"]) for scope in manifest["scopes"]),
        "directory_count": sum(
            len(scope["directories"]) for scope in manifest["scopes"]
        ),
        "root_excluded_prefixes": list(excluded_prefixes),
        "venv_excluded_before_python_lstat_scandir_hash_copy": True,
        "venv_unsupported_hardlink_sentinel_not_read": True,
        "clean_preserved_source_venv": True,
        "local_contract_unchanged": True,
        "remaining_payload_roundtrip_complete": True,
    }


def prove_cross_worktree_case(tool: Any, sandbox: Path) -> dict[str, Any]:
    subsource = sandbox / "submodule-source"
    subsource.mkdir()
    git(subsource, "init", "-b", "main")
    identity(subsource)
    (subsource / ".gitignore").write_text(
        ".cache/\nignored-repo/\n.special/\n", encoding="utf-8"
    )
    (subsource / "tracked.txt").write_text("sub-base\n", encoding="utf-8")
    (subsource / "rename-me.txt").write_text("sub-rename-base\n", encoding="utf-8")
    git(subsource, "add", ".")
    git(subsource, "commit", "-m", "submodule base")

    root = sandbox / "cross-root"
    linked = sandbox / "cross-linked"
    archive = sandbox / "cross-archive"
    init_repo(root)
    git(root, "branch", "dev")
    git(root, "worktree", "add", str(linked), "dev")
    git(
        linked,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(subsource),
        "external/demo",
    )
    git(linked, "commit", "-am", "add submodule")
    dirty(linked, prefix="root-")
    source_venv_sentinel = add_disposable_venv_sentinel(linked)
    nested = linked / "external/demo"
    dirty(nested, prefix="sub-")
    before_root = payload_fingerprint(linked)
    before_submodule = payload_fingerprint(nested)

    manifest = tool.capture(
        linked,
        archive,
        include_ignored=True,
        root_excludes=(".venv",),
    )
    tool.live_snapshot_from_manifest(linked, manifest, require_original_identity=True)
    tool.clean(linked, archive, approved_exact_clean=True)
    if not source_venv_sentinel.is_file() or source_venv_sentinel.stat().st_nlink != 2:
        raise ProofError("cross-worktree clean removed excluded old-source .venv")
    git(linked, "submodule", "deinit", "--all")
    git(linked, "checkout", "--detach", "HEAD")
    git(root, "checkout", "dev")
    git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--recursive",
    )
    destination_venv = root / ".venv"
    destination_venv.mkdir()
    (destination_venv / "collision").write_text("must block restore\n", encoding="utf-8")
    expect_error(
        lambda: tool.restore(root, archive, approved_exact_restore=True),
        "requires absent disposable excluded prefix",
    )
    if (root / "tracked.txt").read_text(encoding="utf-8") != "base\n":
        raise ProofError("blocked cross-worktree restore mutated tracked destination")
    (destination_venv / "collision").unlink()
    destination_venv.rmdir()
    tool.restore(root, archive, approved_exact_restore=True)
    after_root = payload_fingerprint(root)
    after_submodule = payload_fingerprint(root / "external/demo")
    if (before_root, before_submodule) != (after_root, after_submodule):
        raise ProofError("cross-worktree/submodule roundtrip fingerprint changed")
    if not (root / ".local/SECRET").is_file():
        raise ProofError("cross-worktree restore changed canonical .local")
    if os.path.lexists(root / ".venv"):
        raise ProofError("cross-worktree restore created excluded destination .venv")

    foreign = sandbox / "foreign-clone"
    git(sandbox, "clone", "--no-local", str(root), str(foreign))
    git(foreign, "checkout", manifest["source"]["head"])
    git(
        foreign,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--recursive",
    )
    expect_error(
        lambda: tool.restore(foreign, archive, approved_exact_restore=True),
        "common Git dir differs",
    )
    return {
        "root_fingerprint_before": before_root,
        "root_fingerprint_after": after_root,
        "submodule_fingerprint_before": before_submodule,
        "submodule_fingerprint_after": after_submodule,
        "archive_integrity_validated": True,
        "scope_count": len(manifest["scopes"]),
        "extra_count": sum(len(scope["extras"]) for scope in manifest["scopes"]),
        "directory_count": sum(
            len(scope["directories"]) for scope in manifest["scopes"]
        ),
        "root_excluded_prefixes": manifest["policy"]["root_excluded_prefixes"],
        "clean_preserved_old_source_venv": True,
        "restore_omitted_destination_venv": True,
        "local_contract_unchanged": True,
        "remaining_root_and_recursive_submodule_payload_roundtrip_complete": True,
    }


def prove() -> dict[str, Any]:
    COMMANDS.clear()
    sandbox = Path(tempfile.mkdtemp(prefix="nimloth-payload-archive-proof-", dir="/tmp"))
    try:
        tool = load_tool()
        root_case = prove_root_case(tool, sandbox)
        cross_case = prove_cross_worktree_case(tool, sandbox)
        normalized_commands = [
            [
                "$COMMIT"
                if len(normalized) == 40 and all(ch in "0123456789abcdef" for ch in normalized)
                else normalized
                for normalized in (
                    part.replace(str(sandbox), "$SANDBOX") for part in command
                )
            ]
            for command in COMMANDS
        ]
        return {
            "schema": "nimloth-payload-archive-sandbox-proof/v1",
            "sandbox_scope": "/tmp disposable repositories and archives; removed after proof",
            "root_roundtrip": root_case,
            "cross_worktree_submodule_roundtrip": cross_case,
            "negative_controls": {
                "clean_requires_explicit_approval": True,
                "restore_requires_explicit_approval": True,
                "capture_requires_ignored": True,
                "unsupported_payload_exclusions_rejected": True,
                "explicit_disposable_venv_exclusion_supported": True,
                "capture_and_live_validate_CLI_no_venv_paths_verified": True,
                "venv_excluded_before_python_lstat_scandir_hash_copy": True,
                "venv_unsupported_hardlink_sentinel_skipped_before_read": True,
                "malformed_41_hex_object_id_rejected": True,
                "clean_preserves_old_source_venv": True,
                "cross_worktree_existing_destination_venv_blocks_restore": True,
                "cross_worktree_restore_omits_destination_venv": True,
                "archive_inside_worktree_rejected": True,
                "archive_symlink_parent_rejected": True,
                "unsupported_socket_device_fifo_rejected": True,
                "hardlinked_payload_rejected": True,
                "ignored_directory_tree_expanded": True,
                "empty_directory_and_modes_roundtrip": True,
                "unlisted_existing_empty_directory_blocks_restore": True,
                "embedded_git_files_archived_without_git_commands": True,
                "directory_symlink_not_followed": True,
                "local_excluded_and_preserved_contract_unchanged": True,
                "post_capture_recursive_chmod_forbidden_by_policy": True,
                "path_traversal_rejected": True,
                "archive_symlink_patch_rejected": True,
                "unlisted_archive_leaf_rejected": True,
                "patch_payload_path_must_be_listed_and_cannot_touch_local": True,
                "unlisted_live_payload_blocks_clean_without_deletion": True,
                "restore_collision_rejected": True,
                "restore_symlink_parent_rejected_without_external_write": True,
                "foreign_common_git_dir_rejected": True,
                "force_reset_clean_stash_and_manual_metadata_forbidden": True,
            },
            "commands_recorded": len(COMMANDS),
            "commands": normalized_commands,
            "live_nimloth_paths_used": False,
            "archive_payload_embedded_in_evidence": False,
        }
    finally:
        if (
            sandbox.parent != Path("/tmp")
            or not sandbox.name.startswith("nimloth-payload-archive-proof-")
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
        "PASS: root and cross-worktree/submodule payload archive roundtrips; "
        "approval/path/symlink/.local/identity/unlisted/forbidden-command negative controls"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
