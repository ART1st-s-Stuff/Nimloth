#!/usr/bin/env python3
"""Capture, validate, clean, and restore exact worktree payload archives.

Archives must live outside every registered worktree. ``capture`` and
``validate`` do not mutate a source worktree. ``clean`` and ``restore`` are
purposefully unavailable unless the caller supplies the corresponding exact
approval flag. The implementation never invokes force/reset/clean/stash,
never edits ``.git/worktrees``, never follows archive parent symlinks, and
never recursively removes source payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

SCHEMA = "nimloth-worktree-payload-archive/v1"
MANDATORY_ROOT_EXCLUDES = (".local",)
OPTIONAL_DISPOSABLE_ROOT_EXCLUDES = (".venv",)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
FORBIDDEN_GIT_COMMANDS = {"clean", "reset", "stash"}


class ArchiveError(RuntimeError):
    """Raised when an archive operation cannot be proved safe."""


def lexical_absolute(path: Path) -> Path:
    """Return an absolute normalized path without resolving symlinks."""

    return Path(os.path.abspath(os.fspath(path)))


def path_identity(path: Path) -> dict[str, int]:
    metadata = path.lstat()
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IFMT(metadata.st_mode),
    }


def assert_identity(path: Path, expected: dict[str, Any], label: str) -> None:
    if not os.path.lexists(path) or path_identity(path) != expected:
        raise ArchiveError(f"{label} filesystem identity changed: {path}")


def assert_no_symlink_ancestors(path: Path, *, include_leaf: bool) -> None:
    path = lexical_absolute(path)
    parts = path.parts
    current = Path(parts[0])
    limit = len(parts) if include_leaf else len(parts) - 1
    for part in parts[1:limit]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise ArchiveError(f"symlink path component is forbidden: {current}")


def exact_directory(path: Path, label: str) -> Path:
    path = lexical_absolute(path)
    assert_no_symlink_ancestors(path, include_leaf=True)
    if path.is_symlink() or not path.is_dir():
        raise ArchiveError(f"{label} must be a real directory: {path}")
    if path.resolve() != path:
        raise ArchiveError(f"{label} path is not canonical: {path}")
    return path


def safe_relative(value: Any, label: str, *, allow_dot: bool = False) -> str:
    if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
        raise ArchiveError(f"{label} must be a non-empty portable relative path")
    if value == ".":
        if allow_dot:
            return value
        raise ArchiveError(f"{label} cannot name the root itself")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ArchiveError(f"{label} contains an absolute/traversal component: {value!r}")
    if candidate.as_posix() != value:
        raise ArchiveError(f"{label} is not normalized: {value!r}")
    return value


def safe_target(
    root: Path,
    relative: str,
    *,
    create_parents: bool,
    allow_missing_parents: bool = False,
) -> Path:
    relative = safe_relative(relative, "payload path")
    root = exact_directory(root, "payload root")
    target = root.joinpath(*PurePosixPath(relative).parts)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ArchiveError(f"payload escapes root: {relative}") from exc
    current = root
    for component in PurePosixPath(relative).parts[:-1]:
        current /= component
        if os.path.lexists(current):
            if current.is_symlink() or not current.is_dir():
                raise ArchiveError(f"payload parent is not a real directory: {current}")
        elif create_parents:
            current.mkdir(mode=0o700)
        elif not allow_missing_parents:
            raise ArchiveError(f"payload parent is missing: {current}")
    return target


def run_git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    command = ["git", *args]
    if any(arg in {"-f", "-ff", "--force"} or arg.startswith("--force=") for arg in args):
        raise ArchiveError(f"force argument is forbidden: {command}")
    if any(arg in FORBIDDEN_GIT_COMMANDS for arg in args):
        raise ArchiveError(f"destructive Git command is not implemented: {command}")
    normalized_arguments = [os.path.normpath(arg.replace("\\", "/")) for arg in args]
    if any(
        value == ".git/worktrees"
        or value.startswith(".git/worktrees/")
        or "/.git/worktrees/" in value
        for value in normalized_arguments
    ):
        raise ArchiveError("manual .git/worktrees access is forbidden")
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if check and completed.returncode != 0:
        error = completed.stderr.decode("utf-8", "replace").strip()
        raise ArchiveError(f"command failed ({completed.returncode}) at {cwd}: {command}: {error}")
    return completed


def run_scope_git(
    scope: Path,
    name: str,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    prefix = (f"--work-tree={scope}",) if name != "root" else ()
    return run_git(scope, *prefix, *args, check=check)


def git_text(cwd: Path, *args: str) -> str:
    return run_git(cwd, *args).stdout.decode("utf-8", "surrogateescape").strip()


def scope_git_text(scope: Path, name: str, *args: str) -> str:
    return run_scope_git(scope, name, *args).stdout.decode(
        "utf-8", "surrogateescape"
    ).strip()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def path_metadata(path: Path, *, reject_hardlinks: bool = False) -> dict[str, Any]:
    before = path.lstat()
    mode = stat.S_IMODE(before.st_mode)
    if stat.S_ISLNK(before.st_mode):
        target = os.readlink(path)
        result = {
            "kind": "symlink",
            "size_bytes": before.st_size,
            "sha256": sha256_bytes(os.fsencode(target)),
            "mode": mode,
        }
    elif stat.S_ISREG(before.st_mode):
        if reject_hardlinks and before.st_nlink != 1:
            raise ArchiveError(f"hardlinked payload is unsupported: {path}")
        result = {
            "kind": "file",
            "size_bytes": before.st_size,
            "sha256": sha256_file(path),
            "mode": mode,
        }
    else:
        raise ArchiveError(
            f"payload must be a regular file or symlink; socket/device/FIFO are forbidden: {path}"
        )
    after = path.lstat()
    if stat_signature(before) != stat_signature(after):
        raise ArchiveError(f"payload changed identity/content metadata while reading: {path}")
    return result


def source_leaf_metadata(path: Path) -> dict[str, Any]:
    return {**path_metadata(path, reject_hardlinks=True), "filesystem": path_identity(path)}


def directory_metadata(path: Path) -> dict[str, Any]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ArchiveError(f"payload directory must be a real non-symlink directory: {path}")
    after = path.lstat()
    if stat_signature(before) != stat_signature(after):
        raise ArchiveError(f"payload directory changed while reading: {path}")
    return {
        "mode": stat.S_IMODE(before.st_mode),
        "filesystem": path_identity(path),
    }


def git_payload_records(payload: bytes) -> list[tuple[str, bool]]:
    records: list[tuple[str, bool]] = []
    for item in payload.split(b"\0"):
        if not item:
            continue
        rendered = os.fsdecode(item)
        is_directory = rendered.endswith("/")
        normalized = rendered[:-1] if is_directory else rendered
        if not normalized or normalized.endswith("/"):
            raise ArchiveError(f"Git payload directory record is not normalized: {rendered!r}")
        records.append((safe_relative(normalized, "Git payload path"), is_directory))
    return records


def nul_paths(payload: bytes) -> list[str]:
    records = git_payload_records(payload)
    if any(is_directory for _path, is_directory in records):
        raise ArchiveError("directory record is not valid in this Git output")
    return [path for path, _is_directory in records]


def submodule_scopes(root: Path) -> list[tuple[str, Path]]:
    # `submodule status` has no NUL mode and is ambiguous for paths containing
    # spaces/newlines. Git supplies a shell-quoted displaypath to foreach; a
    # fixed printf command gives an exact NUL-delimited initialized-scope list.
    payload = run_git(
        root,
        "submodule",
        "foreach",
        "--quiet",
        "--recursive",
        'printf "%s\\0" "$displaypath"',
    ).stdout
    relatives = nul_paths(payload)
    scopes: list[tuple[str, Path]] = [("root", root)]
    seen = {"root"}
    for relative in relatives:
        if relative in seen:
            raise ArchiveError(f"duplicate initialized submodule scope: {relative}")
        path = root.joinpath(*PurePosixPath(relative).parts)
        marker = path / ".git"
        if not os.path.lexists(marker) or marker.is_symlink():
            raise ArchiveError(f"initialized submodule has no safe .git marker: {path}")
        actual = Path(
            scope_git_text(path, relative, "rev-parse", "--show-toplevel")
        ).resolve()
        if actual != path.resolve():
            raise ArchiveError(f"submodule top-level mismatch: expected={path} actual={actual}")
        scopes.append((relative, path))
        seen.add(relative)
    return scopes


def patch_bytes(scope: Path, name: str, kind: str) -> bytes:
    args = ["diff", "--binary", "--full-index", "--ignore-submodules=dirty"]
    if kind == "index":
        args.extend(["--cached", "HEAD"])
    args.append("--")
    return run_scope_git(scope, name, *args).stdout


def changed_tracked_paths(scope: Path, name: str) -> list[str]:
    fields = run_scope_git(
        scope,
        name,
        "diff",
        "--name-status",
        "-z",
        "--ignore-submodules=dirty",
        "HEAD",
        "--",
    ).stdout.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(fields):
        status_field = fields[index]
        index += 1
        if not status_field:
            continue
        status_code = status_field.decode("ascii", "strict")
        path_count = 2 if status_code.startswith(("R", "C")) else 1
        if index + path_count > len(fields):
            raise ArchiveError("tracked name-status record is truncated")
        for _unused in range(path_count):
            if not fields[index]:
                raise ArchiveError("tracked name-status contains an empty path")
            paths.append(safe_relative(os.fsdecode(fields[index]), "tracked path"))
            index += 1
    return sorted(set(paths))


def normalized_root_excludes(values: Iterable[str]) -> tuple[str, ...]:
    requested = {
        safe_relative(value.rstrip("/"), "root excluded prefix") for value in values
    }
    supported = set(MANDATORY_ROOT_EXCLUDES) | set(OPTIONAL_DISPOSABLE_ROOT_EXCLUDES)
    unsupported = requested - supported
    if unsupported:
        raise ArchiveError(
            "exact archive forbids unsupported root exclusions: "
            + ", ".join(sorted(unsupported))
        )
    return tuple(
        [*MANDATORY_ROOT_EXCLUDES]
        + [prefix for prefix in OPTIONAL_DISPOSABLE_ROOT_EXCLUDES if prefix in requested]
    )


def path_is_excluded(relative: str, prefixes: Iterable[str]) -> bool:
    return any(relative == prefix or relative.startswith(prefix + "/") for prefix in prefixes)


def merge_leaf(
    leaves: dict[str, dict[str, Any]],
    relative: str,
    record: dict[str, Any],
) -> None:
    previous = leaves.get(relative)
    if previous is not None and previous != record:
        raise ArchiveError(f"payload leaf changed between overlapping records: {relative}")
    leaves[relative] = record


def merge_directory(
    directories: dict[str, dict[str, Any]],
    relative: str,
    record: dict[str, Any],
) -> None:
    previous = directories.get(relative)
    if previous is not None and previous != record:
        raise ArchiveError(f"payload directory changed between overlapping records: {relative}")
    directories[relative] = record


def scan_payload_directory(
    scope: Path,
    relative: str,
    *,
    status_name: str,
    excluded_prefixes: tuple[str, ...],
    leaves: dict[str, dict[str, Any]],
    directories: dict[str, dict[str, Any]],
) -> None:
    if path_is_excluded(relative, excluded_prefixes):
        return
    directory = safe_target(scope, relative, create_parents=False)
    before = directory.lstat()
    metadata = directory_metadata(directory)
    merge_directory(
        directories,
        relative,
        {"path": relative, "status": status_name, **metadata},
    )
    try:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: os.fsencode(entry.name))
    except OSError as exc:
        raise ArchiveError(f"cannot enumerate payload directory {directory}: {exc}") from exc
    for entry in entries:
        child_relative = relative + "/" + entry.name
        child_relative = safe_relative(child_relative, "expanded payload path")
        # Mandatory exclusions are applied before lstat/read, and every
        # generated descendant is checked again after expansion.
        if path_is_excluded(child_relative, excluded_prefixes):
            continue
        child = safe_target(scope, child_relative, create_parents=False)
        child_lstat = entry.stat(follow_symlinks=False)
        if stat.S_ISDIR(child_lstat.st_mode):
            scan_payload_directory(
                scope,
                child_relative,
                status_name=status_name,
                excluded_prefixes=excluded_prefixes,
                leaves=leaves,
                directories=directories,
            )
        elif stat.S_ISREG(child_lstat.st_mode) or stat.S_ISLNK(child_lstat.st_mode):
            merge_leaf(
                leaves,
                child_relative,
                {
                    "path": child_relative,
                    "status": status_name,
                    **source_leaf_metadata(child),
                },
            )
        else:
            raise ArchiveError(
                "unsupported socket/device/FIFO in payload directory: " + str(child)
            )
    after = directory.lstat()
    if stat_signature(before) != stat_signature(after):
        raise ArchiveError(f"payload directory changed during expansion: {directory}")


def collect_payload_entries(
    scope: Path,
    name: str,
    *,
    include_ignored: bool,
    root_excludes: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    untracked_records = git_payload_records(
        run_scope_git(
            scope, name, "ls-files", "--others", "--exclude-standard", "-z"
        ).stdout
    )
    ignored_records: list[tuple[str, bool]] = []
    if include_ignored:
        ignored_records = git_payload_records(
            run_scope_git(
                scope,
                name,
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
            ).stdout
        )
    untracked = {path: is_directory for path, is_directory in untracked_records}
    ignored = {path: is_directory for path, is_directory in ignored_records}
    prefixes = root_excludes if name == "root" else ()
    leaves: dict[str, dict[str, Any]] = {}
    directories: dict[str, dict[str, Any]] = {}
    for relative in sorted(set(untracked) | set(ignored)):
        # `.local` is rejected before any safe_target/lstat/scandir call.
        if path_is_excluded(relative, prefixes):
            continue
        directory_flags = {
            mapping[relative]
            for mapping in (untracked, ignored)
            if relative in mapping
        }
        if len(directory_flags) != 1:
            raise ArchiveError(f"Git disagrees on payload record kind: {relative}")
        is_directory = directory_flags.pop()
        status_name = "ignored" if relative in ignored else "untracked"
        if is_directory:
            scan_payload_directory(
                scope,
                relative,
                status_name=status_name,
                excluded_prefixes=prefixes,
                leaves=leaves,
                directories=directories,
            )
        else:
            path = safe_target(scope, relative, create_parents=False)
            merge_leaf(
                leaves,
                relative,
                {"path": relative, "status": status_name, **source_leaf_metadata(path)},
            )

    tracked_files = set(
        nul_paths(run_scope_git(scope, name, "ls-files", "-z").stdout)
    )
    # Record payload-owned parent directory modes required to reconstruct leaf
    # paths. Stop before any directory containing tracked descendants.
    for leaf in list(leaves.values()):
        parent = PurePosixPath(leaf["path"]).parent
        while parent.as_posix() != ".":
            relative = parent.as_posix()
            if path_is_excluded(relative, prefixes):
                break
            if any(
                tracked == relative or tracked.startswith(relative + "/")
                for tracked in tracked_files
            ):
                break
            directory = safe_target(scope, relative, create_parents=False)
            merge_directory(
                directories,
                relative,
                {
                    "path": relative,
                    "status": leaf["status"],
                    **directory_metadata(directory),
                },
            )
            parent = parent.parent

    for relative in [*leaves, *directories]:
        if path_is_excluded(relative, prefixes):
            raise ArchiveError(f"mandatory excluded path entered expanded payload: {relative}")
    for leaf in leaves:
        if leaf in directories or any(
            child.startswith(leaf + "/") for child in [*leaves, *directories] if child != leaf
        ):
            raise ArchiveError(f"payload leaf has a child collision: {leaf}")
    return (
        [leaves[path] for path in sorted(leaves)],
        [directories[path] for path in sorted(directories)],
    )


def scope_snapshot(
    name: str,
    scope: Path,
    *,
    include_ignored: bool,
    root_excludes: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    scope = exact_directory(scope, f"scope {name}")
    patches = {kind: patch_bytes(scope, name, kind) for kind in ("index", "worktree")}
    tracked_paths = changed_tracked_paths(scope, name)
    if name == "root" and any(
        path_is_excluded(relative, root_excludes) for relative in tracked_paths
    ):
        raise ArchiveError("tracked payload intersects mandatory root exclusions")
    extras, directories = collect_payload_entries(
        scope,
        name,
        include_ignored=include_ignored,
        root_excludes=root_excludes,
    )
    return (
        {
            "name": name,
            "relative_path": "." if name == "root" else name,
            "head": scope_git_text(scope, name, "rev-parse", "HEAD"),
            "branch": scope_git_text(scope, name, "branch", "--show-current") or None,
            "top_level": scope_git_text(scope, name, "rev-parse", "--show-toplevel"),
            "common_dir": scope_git_text(
                scope, name, "rev-parse", "--path-format=absolute", "--git-common-dir"
            ),
            "filesystem": path_identity(scope),
            "tracked_paths": tracked_paths,
            "patches": {
                kind: {"size_bytes": len(payload), "sha256": sha256_bytes(payload)}
                for kind, payload in patches.items()
            },
            "extras": extras,
            "directories": directories,
        },
        patches,
    )


def source_identity(root: Path) -> dict[str, Any]:
    return {
        "root": str(root),
        "head": git_text(root, "rev-parse", "HEAD"),
        "branch": git_text(root, "branch", "--show-current") or None,
        "top_level": git_text(root, "rev-parse", "--show-toplevel"),
        "common_dir": git_text(root, "rev-parse", "--path-format=absolute", "--git-common-dir"),
        "filesystem": path_identity(root),
    }


def archive_fingerprint(manifest: dict[str, Any]) -> str:
    stable = {
        "source": manifest["source"],
        "policy": manifest["policy"],
        "scopes": manifest["scopes"],
    }
    rendered = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(rendered)


def registered_worktree_paths(source: Path) -> list[Path]:
    fields = run_git(source, "worktree", "list", "--porcelain", "-z").stdout.split(b"\0")
    return [
        lexical_absolute(Path(os.fsdecode(field[len(b"worktree ") :])))
        for field in fields
        if field.startswith(b"worktree ")
    ]


def assert_archive_outside_worktrees(source: Path, archive: Path) -> None:
    archive = lexical_absolute(archive)
    for worktree in registered_worktree_paths(source):
        if archive == worktree or worktree in archive.parents:
            raise ArchiveError(f"archive must remain outside registered worktrees: {archive}")


def assert_archive_outside_root(root: Path, archive: Path) -> None:
    root = lexical_absolute(root)
    archive = lexical_absolute(archive)
    if archive == root or root in archive.parents:
        raise ArchiveError(f"archive cannot be inside the source/destination worktree: {archive}")


def copy_extra(
    source: Path,
    destination: Path,
    expected: dict[str, Any],
    *,
    destination_root: Path,
    require_source_identity: bool,
) -> None:
    actual_source = (
        source_leaf_metadata(source) if require_source_identity else path_metadata(source)
    )
    wanted_source = {
        key: expected[key] for key in ("kind", "size_bytes", "sha256", "mode")
    }
    if require_source_identity:
        wanted_source["filesystem"] = expected["filesystem"]
    if actual_source != wanted_source:
        raise ArchiveError(f"source payload changed before copy: {source}")
    relative = destination.relative_to(destination_root).as_posix()
    destination = safe_target(destination_root, relative, create_parents=True)
    if os.path.lexists(destination):
        raise ArchiveError(f"copy refuses to overwrite: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".payload-copy-", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if expected["kind"] == "symlink":
            temporary.unlink()
            os.symlink(os.readlink(source), temporary)
        elif expected["kind"] == "file":
            with source.open("rb") as source_stream, temporary.open("wb") as output_stream:
                shutil.copyfileobj(source_stream, output_stream, length=1024 * 1024)
            os.chmod(temporary, expected["mode"])
        else:
            raise ArchiveError(f"unsupported payload kind: {expected['kind']}")
        actual = path_metadata(temporary)
        wanted = {key: expected[key] for key in ("kind", "size_bytes", "sha256", "mode")}
        if actual != wanted:
            raise ArchiveError(f"copied payload verification failed: {destination}")
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise ArchiveError(f"copy destination appeared concurrently: {destination}") from exc
        if path_metadata(destination) != wanted:
            destination.unlink(missing_ok=True)
            raise ArchiveError(f"installed payload verification failed: {destination}")
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def ensure_payload_directory(
    root: Path,
    relative: str,
    *,
    allow_existing: bool,
) -> Path:
    target = safe_target(
        root,
        relative,
        create_parents=True,
        allow_missing_parents=True,
    )
    if os.path.lexists(target):
        if target.is_symlink() or not target.is_dir():
            raise ArchiveError(f"payload directory collides with non-directory: {target}")
        if not allow_existing:
            raise ArchiveError(f"payload directory unexpectedly exists: {target}")
    else:
        target.mkdir(mode=0o700)
    return target


def apply_directory_modes(
    root: Path,
    directories: list[dict[str, Any]],
) -> None:
    for record in sorted(
        directories,
        key=lambda item: len(PurePosixPath(item["path"]).parts),
        reverse=True,
    ):
        target = safe_target(root, record["path"], create_parents=False)
        if target.is_symlink() or not target.is_dir():
            raise ArchiveError(f"payload directory disappeared before chmod: {target}")
        os.chmod(target, record["mode"])
        if stat.S_IMODE(target.lstat().st_mode) != record["mode"]:
            raise ArchiveError(f"payload directory mode restore failed: {target}")


def cleanup_owned_temporary(path: Path, parent: Path, identity: dict[str, Any]) -> None:
    if not path.exists():
        return
    if path.parent != parent or not path.name.startswith("payload-"):
        raise ArchiveError(f"refusing to remove unrecognized temporary archive: {path}")
    assert_identity(path, identity, "temporary archive")
    shutil.rmtree(path)


def capture(
    source: Path,
    archive: Path,
    *,
    include_ignored: bool,
    root_excludes: tuple[str, ...],
) -> dict[str, Any]:
    if not include_ignored:
        raise ArchiveError("exact capture requires --include-ignored")
    source = exact_directory(source, "source")
    if Path(git_text(source, "rev-parse", "--show-toplevel")).resolve() != source:
        raise ArchiveError(f"source is not the exact worktree top-level: {source}")
    archive = lexical_absolute(archive)
    assert_no_symlink_ancestors(archive, include_leaf=True)
    if os.path.lexists(archive):
        raise ArchiveError(f"archive target already exists: {archive}")
    archive_parent = exact_directory(archive.parent, "archive parent")
    assert_archive_outside_worktrees(source, archive)
    root_excludes = normalized_root_excludes(root_excludes)
    parent_identity = path_identity(archive_parent)
    source_before = source_identity(source)
    temporary = Path(tempfile.mkdtemp(prefix="payload-", dir=archive_parent))
    temporary_identity = path_identity(temporary)
    installed = False
    try:
        scopes = submodule_scopes(source)
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "source": source_before,
            "policy": {
                "include_ignored": include_ignored,
                "root_excluded_prefixes": list(root_excludes),
                "mandatory_local_exclusion": True,
                "disposable_venv_exclusion": ".venv" in root_excludes,
                "archive_post_capture_recursive_chmod_forbidden": True,
                "archive_outside_registered_worktrees": True,
                "force_fallback": False,
                "manual_git_worktrees_edit": False,
                "clean_requires_exact_approval": True,
                "restore_requires_exact_approval": True,
                "restore_refuses_overwrite": True,
                "cross_worktree_restore_requires_absent_disposable_excludes": True,
                "clean_preserves_listed_directories": True,
                "restore_recreates_directory_modes": True,
                "recursive_unlisted_payload_delete": False,
            },
            "scopes": [],
            "recovery": {
                "clean": "validate exact live fingerprint, restore only listed tracked paths, unlink only listed leaves, and preserve listed directories plus excluded root prefixes",
                "restore": "require same common Git dir and exact HEADs, then apply verified patches, recreate listed directories/modes, and restore listed leaves without overwrite or excluded root prefixes",
                "orchestration": "never run recursive chmod on a captured archive: leaf modes are integrity metadata; repair only an exact reviewed path before CLI validation",
            },
        }
        for index, (name, scope) in enumerate(scopes):
            snapshot, patches = scope_snapshot(
                name,
                scope,
                include_ignored=include_ignored,
                root_excludes=root_excludes,
            )
            patch_records: dict[str, dict[str, Any]] = {}
            for kind in ("index", "worktree"):
                relative = f"patches/scope-{index}-{kind}.patch"
                target = safe_target(temporary, relative, create_parents=True)
                target.write_bytes(patches[kind])
                patch_records[kind] = {"archive_path": relative, **snapshot["patches"][kind]}
            snapshot["patches"] = patch_records
            for extra in snapshot["extras"]:
                source_extra = safe_target(scope, extra["path"], create_parents=False)
                relative = f"files/{index}/{extra['path']}"
                destination = temporary.joinpath(*PurePosixPath(relative).parts)
                copy_extra(
                    source_extra,
                    destination,
                    extra,
                    destination_root=temporary,
                    require_source_identity=True,
                )
                extra["archive_path"] = relative
            for directory in snapshot["directories"]:
                relative = f"files/{index}/{directory['path']}"
                # Archive container directories stay owner-accessible; the
                # source mode is fingerprinted in the manifest and restored on
                # the destination only after all leaves are installed.
                ensure_payload_directory(temporary, relative, allow_existing=True)
                directory["archive_path"] = relative
            manifest["scopes"].append(snapshot)
        manifest["fingerprint_sha256"] = archive_fingerprint(manifest)
        manifest_path = safe_target(temporary, "manifest.json", create_parents=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        errors = validate_archive(temporary, manifest=manifest)
        if errors:
            raise ArchiveError("archive self-validation failed: " + "; ".join(errors))
        live_snapshot_from_manifest(source, manifest, require_original_identity=True)
        assert_identity(source, source_before["filesystem"], "source")
        assert_identity(archive_parent, parent_identity, "archive parent")
        if os.path.lexists(archive):
            raise ArchiveError(f"archive target appeared concurrently: {archive}")
        temporary.rename(archive)
        installed = True
        return manifest
    finally:
        if not installed and temporary.exists():
            cleanup_owned_temporary(temporary, archive_parent, temporary_identity)


def load_manifest(archive: Path) -> dict[str, Any]:
    archive = exact_directory(archive, "archive")
    manifest_path = safe_target(archive, "manifest.json", create_parents=False)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ArchiveError(f"archive manifest must be a regular file: {manifest_path}")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"cannot read archive manifest: {archive}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArchiveError("archive manifest root must be an object")
    return value


def valid_identity(value: Any, expected_type: int) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"device", "inode", "mode"}
        and all(isinstance(value[key], int) and value[key] >= 0 for key in value)
        and value["mode"] == expected_type
    )


def archive_tree(archive: Path) -> tuple[set[str], set[str]]:
    leaves: set[str] = set()
    directories: set[str] = {"."}
    for directory, names, files in os.walk(archive, topdown=True, followlinks=False):
        base = Path(directory)
        kept: list[str] = []
        for name in names:
            path = base / name
            relative = path.relative_to(archive).as_posix()
            if path.is_symlink():
                leaves.add(relative)
            else:
                directories.add(relative)
                kept.append(name)
        names[:] = kept
        leaves.update((base / name).relative_to(archive).as_posix() for name in files)
    return leaves, directories


def expected_archive_directories(leaves: Iterable[str]) -> set[str]:
    directories = {"."}
    for leaf in leaves:
        parent = PurePosixPath(leaf).parent
        while parent.as_posix() != ".":
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def validate_archive(archive: Path, *, manifest: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    try:
        archive = exact_directory(archive, "archive")
        manifest = manifest if manifest is not None else load_manifest(archive)
    except ArchiveError as exc:
        return [str(exc)]
    if manifest.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    try:
        if manifest.get("fingerprint_sha256") != archive_fingerprint(manifest):
            errors.append("manifest fingerprint mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"manifest fingerprint fields are malformed: {exc}")
    policy = manifest.get("policy", {})
    if not isinstance(policy, dict):
        errors.append("policy must be an object")
        policy = {}
    required_true = {
        "include_ignored",
        "mandatory_local_exclusion",
        "archive_post_capture_recursive_chmod_forbidden",
        "archive_outside_registered_worktrees",
        "clean_requires_exact_approval",
        "restore_requires_exact_approval",
        "restore_refuses_overwrite",
        "cross_worktree_restore_requires_absent_disposable_excludes",
        "clean_preserves_listed_directories",
        "restore_recreates_directory_modes",
    }
    for key in sorted(required_true):
        if policy.get(key) is not True:
            errors.append(f"archive policy must enable {key}")
    if policy.get("force_fallback") or policy.get("manual_git_worktrees_edit"):
        errors.append("archive policy does not prohibit force/manual metadata edits")
    if policy.get("recursive_unlisted_payload_delete") is not False:
        errors.append("archive policy must prohibit recursive unlisted payload deletion")
    try:
        excludes = normalized_root_excludes(policy.get("root_excluded_prefixes", []))
        if list(excludes) != policy.get("root_excluded_prefixes"):
            errors.append("root exclusions are not normalized or omit .local")
        if policy.get("disposable_venv_exclusion") is not (".venv" in excludes):
            errors.append("disposable .venv policy differs from root exclusions")
    except ArchiveError as exc:
        errors.append(str(exc))
        excludes = MANDATORY_ROOT_EXCLUDES
    source = manifest.get("source", {})
    required_source_keys = {
        "root",
        "head",
        "branch",
        "top_level",
        "common_dir",
        "filesystem",
    }
    if (
        not isinstance(source, dict)
        or set(source) != required_source_keys
        or not isinstance(source.get("root"), str)
        or not HEX_OBJECT.fullmatch(str(source.get("head", "")))
        or not valid_identity(source.get("filesystem"), stat.S_IFDIR)
    ):
        errors.append("source Git/filesystem identity is malformed")
    else:
        try:
            source_root = lexical_absolute(Path(source["root"]))
            if source.get("top_level") != str(source_root):
                errors.append("source top_level differs from source root")
            if archive == source_root or source_root in archive.parents:
                errors.append("archive is inside the recorded source worktree")
            source_root = exact_directory(source_root, "recorded source")
            for worktree in registered_worktree_paths(source_root):
                if archive == worktree or worktree in archive.parents:
                    errors.append("archive is inside a registered worktree")
        except (ArchiveError, OSError, ValueError) as exc:
            errors.append(f"recorded source identity cannot be verified: {exc}")
    scopes = manifest.get("scopes")
    if not isinstance(scopes, list) or not scopes:
        errors.append("scopes must be a non-empty list")
        return errors
    expected_leaves = {"manifest.json"}
    listed_archive_directories: set[str] = set()
    scope_names: set[str] = set()
    for scope_index, scope in enumerate(scopes):
        if not isinstance(scope, dict):
            errors.append(f"scope {scope_index} must be an object")
            continue
        name = scope.get("name")
        try:
            if scope_index == 0:
                if name != "root" or scope.get("relative_path") != ".":
                    raise ArchiveError("first scope must be root")
            else:
                name = safe_relative(name, f"scope {scope_index} name")
                if scope.get("relative_path") != name:
                    raise ArchiveError(f"scope {scope_index} relative_path differs from name")
            if name in scope_names:
                raise ArchiveError(f"duplicate scope name: {name}")
            scope_names.add(name)
        except ArchiveError as exc:
            errors.append(str(exc))
        if not HEX_OBJECT.fullmatch(str(scope.get("head", ""))):
            errors.append(f"scope {scope_index} has invalid HEAD")
        if not valid_identity(scope.get("filesystem"), stat.S_IFDIR):
            errors.append(f"scope {scope_index} filesystem identity is malformed")
        if not isinstance(scope.get("top_level"), str) or not os.path.isabs(
            scope.get("top_level", "")
        ):
            errors.append(f"scope {scope_index} top_level is malformed")
        if not isinstance(scope.get("common_dir"), str) or not os.path.isabs(
            scope.get("common_dir", "")
        ):
            errors.append(f"scope {scope_index} common_dir is malformed")
        tracked_paths = scope.get("tracked_paths")
        if not isinstance(tracked_paths, list):
            errors.append(f"scope {scope_index} tracked_paths must be a list")
            tracked_paths = []
        normalized_tracked: list[str] = []
        for tracked in tracked_paths:
            try:
                normalized_tracked.append(safe_relative(tracked, "tracked path"))
            except ArchiveError as exc:
                errors.append(str(exc))
        if normalized_tracked != tracked_paths or len(set(normalized_tracked)) != len(normalized_tracked):
            errors.append(f"scope {scope_index} tracked_paths are not normalized and unique")
        if scope_index == 0 and any(
            path_is_excluded(relative, excludes) for relative in normalized_tracked
        ):
            errors.append("root tracked payload intersects mandatory exclusions")
        patches = scope.get("patches")
        if not isinstance(patches, dict) or set(patches) != {"index", "worktree"}:
            errors.append(f"scope {scope_index} patches must contain index/worktree")
            patches = {}
        for kind in ("index", "worktree"):
            record = patches.get(kind, {})
            expected_relative = f"patches/scope-{scope_index}-{kind}.patch"
            if (
                not isinstance(record, dict)
                or record.get("archive_path") != expected_relative
                or not isinstance(record.get("size_bytes"), int)
                or record.get("size_bytes", -1) < 0
                or not HEX_SHA256.fullmatch(str(record.get("sha256", "")))
            ):
                errors.append(f"scope {scope_index} has unsafe/malformed {kind} patch record")
                continue
            expected_leaves.add(expected_relative)
            try:
                path = safe_target(archive, expected_relative, create_parents=False)
                metadata = path.lstat()
                if not stat.S_ISREG(metadata.st_mode):
                    raise ArchiveError(f"patch must be a regular non-symlink file: {path}")
                if (
                    metadata.st_size != record.get("size_bytes")
                    or sha256_file(path) != record.get("sha256")
                    or not HEX_SHA256.fullmatch(str(record.get("sha256", "")))
                ):
                    errors.append(f"{kind} patch integrity mismatch for scope {scope_index}")
            except (ArchiveError, OSError) as exc:
                errors.append(str(exc))
        extras = scope.get("extras")
        if not isinstance(extras, list):
            errors.append(f"scope {scope_index} extras must be a list")
            extras = []
        extra_paths: list[str] = []
        for extra_index, extra in enumerate(extras):
            if not isinstance(extra, dict):
                errors.append(f"extra {scope_index}:{extra_index} must be an object")
                continue
            try:
                payload_path = safe_relative(extra.get("path"), "extra payload path")
                if extra.get("status") not in {"untracked", "ignored"}:
                    raise ArchiveError(f"extra status is invalid: {payload_path}")
                if extra.get("kind") not in {"file", "symlink"}:
                    raise ArchiveError(f"extra kind is invalid: {payload_path}")
                if (
                    not isinstance(extra.get("size_bytes"), int)
                    or extra.get("size_bytes", -1) < 0
                    or not isinstance(extra.get("mode"), int)
                    or not valid_identity(
                        extra.get("filesystem"),
                        stat.S_IFREG if extra.get("kind") == "file" else stat.S_IFLNK,
                    )
                ):
                    raise ArchiveError(f"extra metadata/identity is malformed: {payload_path}")
                expected_relative = f"files/{scope_index}/{payload_path}"
                if extra.get("archive_path") != expected_relative:
                    raise ArchiveError(
                        f"extra archive path mismatch {scope_index}:{extra_index}"
                    )
                if scope_index == 0 and path_is_excluded(payload_path, excludes):
                    raise ArchiveError(f"excluded root payload entered archive: {payload_path}")
                expected_leaves.add(expected_relative)
                extra_paths.append(payload_path)
                path = safe_target(archive, expected_relative, create_parents=False)
                if not os.path.lexists(path):
                    raise ArchiveError(f"missing extra {scope_index}:{extra_index}")
                actual = path_metadata(path)
                wanted = {
                    key: extra.get(key) for key in ("kind", "size_bytes", "sha256", "mode")
                }
                if actual != wanted or not HEX_SHA256.fullmatch(str(extra.get("sha256", ""))):
                    errors.append(f"extra integrity mismatch {scope_index}:{extra_index}")
            except (ArchiveError, OSError) as exc:
                errors.append(str(exc))
        if len(extra_paths) != len(set(extra_paths)):
            errors.append(f"scope {scope_index} extras contain duplicate paths")
        for left in extra_paths:
            if any(right.startswith(left + "/") for right in extra_paths if right != left):
                errors.append(f"scope {scope_index} extras contain parent/child collision: {left}")

        directory_records = scope.get("directories")
        if not isinstance(directory_records, list):
            errors.append(f"scope {scope_index} directories must be a list")
            directory_records = []
        directory_paths: list[str] = []
        for directory_index, directory in enumerate(directory_records):
            if not isinstance(directory, dict):
                errors.append(f"directory {scope_index}:{directory_index} must be an object")
                continue
            try:
                payload_path = safe_relative(directory.get("path"), "payload directory path")
                if directory.get("status") not in {"untracked", "ignored"}:
                    raise ArchiveError(f"directory status is invalid: {payload_path}")
                if not isinstance(directory.get("mode"), int) or not valid_identity(
                    directory.get("filesystem"), stat.S_IFDIR
                ):
                    raise ArchiveError(f"directory metadata is malformed: {payload_path}")
                expected_relative = f"files/{scope_index}/{payload_path}"
                if directory.get("archive_path") != expected_relative:
                    raise ArchiveError(
                        f"directory archive path mismatch {scope_index}:{directory_index}"
                    )
                if scope_index == 0 and path_is_excluded(payload_path, excludes):
                    raise ArchiveError(f"excluded root directory entered archive: {payload_path}")
                path = safe_target(archive, expected_relative, create_parents=False)
                if path.is_symlink() or not path.is_dir():
                    raise ArchiveError(f"archived payload directory is missing/unsafe: {path}")
                listed_archive_directories.add(expected_relative)
                directory_paths.append(payload_path)
            except (ArchiveError, OSError) as exc:
                errors.append(str(exc))
        if len(directory_paths) != len(set(directory_paths)):
            errors.append(f"scope {scope_index} directories contain duplicate paths")
        for leaf in extra_paths:
            if leaf in directory_paths or any(path.startswith(leaf + "/") for path in directory_paths):
                errors.append(f"scope {scope_index} leaf/directory collision: {leaf}")
    try:
        actual_leaves, actual_directories = archive_tree(archive)
        if actual_leaves != expected_leaves:
            errors.append(
                "archive contains missing/unlisted leaves: "
                f"missing={sorted(expected_leaves - actual_leaves)} "
                f"unlisted={sorted(actual_leaves - expected_leaves)}"
            )
        expected_directories = expected_archive_directories(expected_leaves)
        for listed_directory in listed_archive_directories:
            current = PurePosixPath(listed_directory)
            while current.as_posix() != ".":
                expected_directories.add(current.as_posix())
                current = current.parent
        if actual_directories != expected_directories:
            errors.append(
                "archive contains missing/unlisted directories: "
                f"missing={sorted(expected_directories - actual_directories)} "
                f"unlisted={sorted(actual_directories - expected_directories)}"
            )
    except OSError as exc:
        errors.append(f"cannot enumerate archive tree: {exc}")
    return errors


def validated_manifest(archive: Path) -> dict[str, Any]:
    manifest = load_manifest(archive)
    errors = validate_archive(archive, manifest=manifest)
    if errors:
        raise ArchiveError("archive validation failed: " + "; ".join(errors))
    return manifest


def normalized_scopes(scopes: list[dict[str, Any]], *, portable: bool) -> list[dict[str, Any]]:
    normalized = json.loads(json.dumps(scopes))
    for scope in normalized:
        if portable:
            # Linked-worktree submodules use worktree-private Git dirs while a
            # main-worktree restore uses the common modules dir. Root common
            # identity is checked separately before restore.
            for key in ("branch", "top_level", "common_dir", "filesystem"):
                scope.pop(key, None)
        scope["patches"] = {
            kind: {key: value for key, value in record.items() if key != "archive_path"}
            for kind, record in scope["patches"].items()
        }
        for extra in scope["extras"]:
            extra.pop("archive_path", None)
            if portable:
                extra.pop("filesystem", None)
        for directory in scope["directories"]:
            directory.pop("archive_path", None)
            if portable:
                directory.pop("filesystem", None)
    return normalized


def live_snapshot_from_manifest(
    root: Path,
    manifest: dict[str, Any],
    *,
    require_original_identity: bool,
) -> list[dict[str, Any]]:
    root = exact_directory(root, "live root")
    current_scopes = submodule_scopes(root)
    current = {name: path for name, path in current_scopes}
    expected_names = [scope["name"] for scope in manifest["scopes"]]
    if set(current) != set(expected_names):
        raise ArchiveError(
            f"live initialized scopes differ: expected={sorted(expected_names)} actual={sorted(current)}"
        )
    policy = manifest["policy"]
    snapshots: list[dict[str, Any]] = []
    for expected in manifest["scopes"]:
        snapshot, _patches = scope_snapshot(
            expected["name"],
            current[expected["name"]],
            include_ignored=bool(policy["include_ignored"]),
            root_excludes=tuple(policy["root_excluded_prefixes"]),
        )
        snapshots.append(snapshot)
    if require_original_identity:
        if source_identity(root) != manifest["source"]:
            raise ArchiveError("live source Git/filesystem identity differs from archive manifest")
    else:
        if git_text(root, "rev-parse", "HEAD") != manifest["source"]["head"]:
            raise ArchiveError("destination HEAD differs from archived source HEAD")
        if (
            git_text(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
            != manifest["source"]["common_dir"]
        ):
            raise ArchiveError("destination common Git dir differs from archive source")
    portable = not require_original_identity
    if normalized_scopes(snapshots, portable=portable) != normalized_scopes(
        manifest["scopes"], portable=portable
    ):
        raise ArchiveError("live payload differs from exact archive manifest")
    return snapshots


def remove_exact_file(scope: Path, relative: str, expected: dict[str, Any]) -> None:
    target = safe_target(scope, relative, create_parents=False)
    if not (target.is_symlink() or target.is_file()):
        if target.exists():
            raise ArchiveError(f"refusing directory/recursive payload deletion: {target}")
        raise ArchiveError(f"listed payload disappeared before cleanup: {target}")
    actual = source_leaf_metadata(target)
    wanted = {
        key: expected[key]
        for key in ("kind", "size_bytes", "sha256", "mode", "filesystem")
    }
    if actual != wanted:
        raise ArchiveError(f"listed payload changed before cleanup: {target}")
    target.unlink()
    if os.path.lexists(target):
        raise ArchiveError(f"listed payload still exists after unlink: {target}")


def assert_scope_clean(
    name: str,
    scope: Path,
    *,
    root_excludes: tuple[str, ...],
) -> None:
    snapshot, _patches = scope_snapshot(
        name,
        scope,
        include_ignored=True,
        root_excludes=root_excludes,
    )
    if (
        snapshot["tracked_paths"]
        or snapshot["extras"]
        or snapshot["patches"]["index"]["size_bytes"]
        or snapshot["patches"]["worktree"]["size_bytes"]
    ):
        raise ArchiveError(f"scope is not exact-clean outside exclusions: {name}")


def preflight_restore_directories(
    scope: Path,
    directories: list[dict[str, Any]],
) -> None:
    expected = {record["path"] for record in directories}
    roots = [
        relative
        for relative in expected
        if PurePosixPath(relative).parent.as_posix() not in expected
    ]
    for root_relative in roots:
        root = safe_target(
            scope,
            root_relative,
            create_parents=False,
            allow_missing_parents=True,
        )
        if not os.path.lexists(root):
            continue
        if root.is_symlink() or not root.is_dir():
            raise ArchiveError(f"restore directory root collides with non-directory: {root}")
        for directory, names, files in os.walk(root, topdown=True, followlinks=False):
            base = Path(directory)
            relative_base = base.relative_to(scope).as_posix()
            if relative_base not in expected:
                raise ArchiveError(
                    f"restore found unlisted existing directory: {relative_base}"
                )
            for name in [*names, *files]:
                path = base / name
                relative = path.relative_to(scope).as_posix()
                if path.is_symlink() or not path.is_dir():
                    raise ArchiveError(f"restore found unlisted existing leaf: {relative}")
                if relative not in expected:
                    raise ArchiveError(f"restore found unlisted existing directory: {relative}")


def verify_preserved_directories(
    scope: Path,
    directories: list[dict[str, Any]],
    *,
    require_source_identity: bool,
) -> None:
    for record in directories:
        target = safe_target(scope, record["path"], create_parents=False)
        if target.is_symlink() or not target.is_dir():
            raise ArchiveError(f"listed payload directory was not preserved: {target}")
        if stat.S_IMODE(target.lstat().st_mode) != record["mode"]:
            raise ArchiveError(f"listed payload directory mode changed: {target}")
        if require_source_identity:
            assert_identity(target, record["filesystem"], "listed payload directory")


def clean(
    root: Path,
    archive: Path,
    *,
    approved_exact_clean: bool = False,
) -> None:
    if not approved_exact_clean:
        raise ArchiveError("clean requires explicit approved_exact_clean=True")
    root = exact_directory(root, "clean source")
    archive = exact_directory(archive, "archive")
    assert_archive_outside_root(root, archive)
    manifest = validated_manifest(archive)
    if manifest["policy"].get("include_ignored") is not True:
        raise ArchiveError("clean requires an archive captured with ignored payload")
    live_snapshot_from_manifest(root, manifest, require_original_identity=True)
    scopes = {name: path for name, path in submodule_scopes(root)}
    root_excludes = tuple(manifest["policy"]["root_excluded_prefixes"])
    for expected in reversed(manifest["scopes"]):
        name = expected["name"]
        scope = scopes[name]
        assert_identity(scope, expected["filesystem"], f"scope {name}")
        fresh, _patches = scope_snapshot(
            name,
            scope,
            include_ignored=True,
            root_excludes=root_excludes,
        )
        if normalized_scopes([fresh], portable=False) != normalized_scopes(
            [expected], portable=False
        ):
            raise ArchiveError(f"scope changed immediately before cleanup: {name}")
        tracked_paths = expected["tracked_paths"]
        if tracked_paths:
            run_scope_git(
                scope,
                name,
                "restore",
                "--source=HEAD",
                "--staged",
                "--worktree",
                "--",
                *tracked_paths,
            )
        for extra in expected["extras"]:
            remove_exact_file(scope, extra["path"], extra)
        assert_scope_clean(name, scope, root_excludes=root_excludes)
        verify_preserved_directories(
            scope,
            expected["directories"],
            require_source_identity=True,
        )
    assert_identity(root, manifest["source"]["filesystem"], "clean source")


def verify_patch(archive: Path, record: dict[str, Any]) -> Path:
    relative = safe_relative(record["archive_path"], "patch archive path")
    path = safe_target(archive, relative, create_parents=False)
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size != record["size_bytes"]
        or sha256_file(path) != record["sha256"]
    ):
        raise ArchiveError(f"patch changed after archive validation: {path}")
    return path


def parsed_patch_paths(scope: Path, name: str, patch: Path) -> set[str]:
    payload = run_scope_git(
        scope,
        name,
        "apply",
        "--numstat",
        "-z",
        str(patch),
    ).stdout
    fields = payload.split(b"\0")
    paths: set[str] = set()
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        parts = field.split(b"\t", 2)
        if len(parts) != 3:
            raise ArchiveError(f"cannot parse patch numstat record: {field!r}")
        rendered = parts[2]
        if rendered:
            paths.add(safe_relative(os.fsdecode(rendered), "patch payload path"))
            continue
        # With -z, rename/copy records store old and new names as the next two
        # NUL fields after an empty pathname in the numstat prefix.
        if index + 1 >= len(fields) or not fields[index] or not fields[index + 1]:
            raise ArchiveError("patch rename/copy numstat record is truncated")
        paths.add(safe_relative(os.fsdecode(fields[index]), "patch source path"))
        paths.add(safe_relative(os.fsdecode(fields[index + 1]), "patch destination path"))
        index += 2
    return paths


def assert_cross_worktree_disposable_excludes_absent(
    root: Path,
    manifest: dict[str, Any],
) -> None:
    if str(root) == manifest["source"]["root"]:
        return
    configured = set(manifest["policy"]["root_excluded_prefixes"])
    for prefix in OPTIONAL_DISPOSABLE_ROOT_EXCLUDES:
        if prefix not in configured:
            continue
        target = safe_target(
            root,
            prefix,
            create_parents=False,
            allow_missing_parents=True,
        )
        if os.path.lexists(target):
            raise ArchiveError(
                f"cross-worktree restore requires absent disposable excluded prefix: {target}"
            )


def restore(
    root: Path,
    archive: Path,
    *,
    approved_exact_restore: bool = False,
) -> None:
    if not approved_exact_restore:
        raise ArchiveError("restore requires explicit approved_exact_restore=True")
    root = exact_directory(root, "restore destination")
    archive = exact_directory(archive, "archive")
    assert_archive_outside_root(root, archive)
    destination_identity = path_identity(root)
    manifest = validated_manifest(archive)
    assert_cross_worktree_disposable_excludes_absent(root, manifest)
    if (
        git_text(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
        != manifest["source"]["common_dir"]
    ):
        raise ArchiveError("destination common Git dir differs from archive source")
    if git_text(root, "rev-parse", "HEAD") != manifest["source"]["head"]:
        raise ArchiveError("destination HEAD differs from archived source HEAD")
    scopes = {name: path for name, path in submodule_scopes(root)}
    expected_names = {scope["name"] for scope in manifest["scopes"]}
    if set(scopes) != expected_names:
        raise ArchiveError("destination initialized submodule scopes differ from archive")
    root_excludes = tuple(manifest["policy"]["root_excluded_prefixes"])
    plans: dict[str, dict[str, Path]] = {}
    destination_scope_identities: dict[str, dict[str, int]] = {}
    # Preflight every scope and every archive patch before the first mutation,
    # so a malformed nested scope cannot leave an earlier scope half-restored.
    for expected in manifest["scopes"]:
        name = expected["name"]
        scope = scopes[name]
        destination_scope_identities[name] = path_identity(scope)
        if scope_git_text(scope, name, "rev-parse", "HEAD") != expected["head"]:
            raise ArchiveError(f"destination scope HEAD differs: {name}")
        assert_scope_clean(name, scope, root_excludes=root_excludes)
        for extra in expected["extras"]:
            target = safe_target(
                scope,
                extra["path"],
                create_parents=False,
                allow_missing_parents=True,
            )
            if os.path.lexists(target):
                raise ArchiveError(f"restore refuses to overwrite: {target}")
        for directory in expected["directories"]:
            target = safe_target(
                scope,
                directory["path"],
                create_parents=False,
                allow_missing_parents=True,
            )
            if os.path.lexists(target) and (target.is_symlink() or not target.is_dir()):
                raise ArchiveError(f"restore directory collides with non-directory: {target}")
        preflight_restore_directories(scope, expected["directories"])
        patch_plan: dict[str, Path] = {}
        patch_paths: set[str] = set()
        for kind in ("index", "worktree"):
            patch = verify_patch(archive, expected["patches"][kind])
            patch_plan[kind] = patch
            if patch.stat().st_size:
                patch_paths.update(parsed_patch_paths(scope, name, patch))
        expected_paths = set(expected["tracked_paths"])
        # Rename/copy numstat may report only the destination while cleanup
        # must list both source and destination. Every patch path must still be
        # an approved listed tracked path, and an empty/non-empty mismatch is
        # forbidden.
        if not patch_paths.issubset(expected_paths) or bool(patch_paths) != bool(
            expected_paths
        ):
            raise ArchiveError(
                f"patch paths are not covered by listed tracked paths for {name}: "
                f"patch={sorted(patch_paths)} listed={sorted(expected_paths)}"
            )
        if name == "root" and any(
            path_is_excluded(relative, root_excludes) for relative in patch_paths
        ):
            raise ArchiveError("root patch intersects mandatory exclusions")
        plans[name] = patch_plan

    for expected in manifest["scopes"]:
        name = expected["name"]
        scope = scopes[name]
        assert_identity(scope, destination_scope_identities[name], f"restore scope {name}")
        for kind in ("index", "worktree"):
            patch = verify_patch(archive, expected["patches"][kind])
            if patch != plans[name][kind]:
                raise ArchiveError("patch path identity changed after restore preflight")
            if patch.stat().st_size:
                arguments = ["apply", "--binary"]
                if kind == "index":
                    arguments.append("--index")
                run_scope_git(scope, name, *arguments, "--check", str(patch))
                run_scope_git(scope, name, *arguments, str(patch))
        for directory in sorted(
            expected["directories"],
            key=lambda item: len(PurePosixPath(item["path"]).parts),
        ):
            ensure_payload_directory(scope, directory["path"], allow_existing=True)
        for extra in expected["extras"]:
            source = safe_target(archive, extra["archive_path"], create_parents=False)
            target = scope.joinpath(*PurePosixPath(extra["path"]).parts)
            copy_extra(
                source,
                target,
                extra,
                destination_root=scope,
                require_source_identity=False,
            )
        apply_directory_modes(scope, expected["directories"])
        verify_preserved_directories(
            scope,
            expected["directories"],
            require_source_identity=False,
        )
    assert_identity(root, destination_identity, "restore destination")
    live_snapshot_from_manifest(root, manifest, require_original_identity=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--source", required=True)
    capture_parser.add_argument("--archive", required=True)
    capture_parser.add_argument("--include-ignored", action="store_true")
    capture_parser.add_argument("--exclude-root-prefix", action="append", default=[])

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--archive", required=True)
    validate_parser.add_argument("--live-source")

    clean_parser = subparsers.add_parser("clean")
    clean_parser.add_argument("--source", required=True)
    clean_parser.add_argument("--archive", required=True)
    clean_parser.add_argument("--approved-exact-clean", action="store_true")

    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("--destination", required=True)
    restore_parser.add_argument("--archive", required=True)
    restore_parser.add_argument("--approved-exact-restore", action="store_true")

    args = parser.parse_args()
    try:
        archive = lexical_absolute(Path(args.archive))
        if args.command == "capture":
            manifest = capture(
                lexical_absolute(Path(args.source)),
                archive,
                include_ignored=args.include_ignored,
                root_excludes=tuple(args.exclude_root_prefix),
            )
            extras = sum(len(scope["extras"]) for scope in manifest["scopes"])
            print(
                f"PASS: archive captured outside all worktrees and verified: {archive}; "
                f"scopes={len(manifest['scopes'])} extras={extras} "
                f"fingerprint={manifest['fingerprint_sha256']}"
            )
            return 0
        if args.command == "validate":
            manifest = validated_manifest(archive)
            if args.live_source:
                live_snapshot_from_manifest(
                    lexical_absolute(Path(args.live_source)),
                    manifest,
                    require_original_identity=True,
                )
            print(f"PASS: archive integrity, layout, identity, and policy validated: {archive}")
            return 0
        if args.command == "clean":
            clean(
                lexical_absolute(Path(args.source)),
                archive,
                approved_exact_clean=args.approved_exact_clean,
            )
            print(f"PASS: exact listed payload cleaned from {args.source}")
            return 0
        if args.command == "restore":
            restore(
                lexical_absolute(Path(args.destination)),
                archive,
                approved_exact_restore=args.approved_exact_restore,
            )
            print(f"PASS: exact archived payload restored to {args.destination}")
            return 0
        raise ArchiveError(f"unknown command: {args.command}")
    except (ArchiveError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
