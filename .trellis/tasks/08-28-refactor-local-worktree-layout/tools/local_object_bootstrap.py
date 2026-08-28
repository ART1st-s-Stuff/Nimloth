#!/usr/bin/env python3
"""Transfer one exact commit closure between existing local module Git dirs.

The transfer uses only local Git plumbing (``pack-objects`` -> verified pack ->
``unpack-objects``). It cannot contact a transport, does not create a remote or
ref, and does not write FETCH_HEAD, HEAD, the index, or a worktree. Both Git
dirs, object formats, source HEAD/closure, exact pack object set, and
non-object destination fingerprints are checked before and after the
object-only mutation. Live use remains unavailable without the explicit
approval flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

SCHEMA = "nimloth-local-module-object-bootstrap/v1"
OBJECT_FORMAT_LENGTHS = {"sha1": 40, "sha256": 64}
HEX_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class BootstrapError(RuntimeError):
    """Raised when an exact object-only transfer cannot be proved safe."""


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def assert_no_symlink_components(path: Path) -> None:
    path = lexical_absolute(path)
    current = Path(path.parts[0])
    for component in path.parts[1:]:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            raise BootstrapError(f"symlink path component is forbidden: {current}")


def path_identity(path: Path) -> dict[str, int]:
    metadata = path.lstat()
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "kind": stat.S_IFMT(metadata.st_mode),
    }


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


def exact_real_directory(path: Path, label: str) -> Path:
    path = lexical_absolute(path)
    assert_no_symlink_components(path)
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise BootstrapError(f"{label} must be an exact real directory: {path}")
    return path


def assert_directory_empty(path: Path, label: str) -> None:
    path = exact_real_directory(path, label)
    before = path.lstat()
    with os.scandir(path) as entries:
        if next(entries, None) is not None:
            raise BootstrapError(f"{label} must be empty: {path}")
    if stat_signature(before) != stat_signature(path.lstat()):
        raise BootstrapError(f"{label} changed while checking emptiness: {path}")


def git_environment() -> dict[str, str]:
    # Strip inherited Git repository/object/config routing before setting the
    # exact local plumbing environment. GIT_NO_LAZY_FETCH and the empty
    # protocol allowlist make a partial-clone/network dependency fail closed.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_ALLOW_PROTOCOL": "",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    return environment


def run_git(
    git_dir: Path,
    *args: str,
    check: bool = True,
    input_bytes: bytes | None = None,
    stdin: Any = None,
    stdout: Any = subprocess.PIPE,
) -> subprocess.CompletedProcess[bytes]:
    command = ["git", f"--git-dir={git_dir}", *args]
    completed = subprocess.run(
        command,
        input=input_bytes,
        stdin=stdin,
        stdout=stdout,
        stderr=subprocess.PIPE,
        check=False,
        env=git_environment(),
    )
    if check and completed.returncode != 0:
        error = completed.stderr.decode("utf-8", "replace").strip()
        raise BootstrapError(
            f"Git command failed ({completed.returncode}): {command}: {error}"
        )
    return completed


def git_text(git_dir: Path, *args: str, check: bool = True) -> str:
    return run_git(git_dir, *args, check=check).stdout.decode(
        "utf-8", "surrogateescape"
    ).strip()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def exact_object_format(value: str) -> str:
    if value not in OBJECT_FORMAT_LENGTHS:
        raise BootstrapError(f"unsupported Git object format: {value!r}")
    return value


def repository_object_format(git_dir: Path) -> str:
    return exact_object_format(git_text(git_dir, "rev-parse", "--show-object-format"))


def exact_commit(value: str, object_format: str | None = None) -> str:
    if not HEX_OBJECT.fullmatch(value):
        raise BootstrapError(f"commit must be one full exact object ID: {value!r}")
    if object_format is not None:
        object_format = exact_object_format(object_format)
        if len(value) != OBJECT_FORMAT_LENGTHS[object_format]:
            raise BootstrapError(
                f"commit length does not match {object_format}: {value!r}"
            )
    return value


def strict_file_fingerprint(path: Path) -> dict[str, Any]:
    if not os.path.lexists(path):
        return {"kind": "absent", "size_bytes": None, "sha256": None}
    assert_no_symlink_components(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise BootstrapError(f"Git control path must be a regular file: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        opened_before = os.fstat(descriptor)
        if stat_signature(opened_before) != stat_signature(before):
            raise BootstrapError(f"Git control file changed before read: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        stat_signature(opened_before) != stat_signature(opened_after)
        or stat_signature(before) != stat_signature(after)
    ):
        raise BootstrapError(f"Git control file changed while reading: {path}")
    return {
        "kind": "file",
        "size_bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }


def control_tree_fingerprint(git_dir: Path) -> dict[str, Any]:
    rows: list[tuple[Any, ...]] = []
    for directory, names, files in os.walk(git_dir, topdown=True, followlinks=False):
        base = Path(directory)
        relative_base = base.relative_to(git_dir)
        if relative_base == Path(".") and "objects" in names:
            names.remove("objects")
        kept: list[str] = []
        for name in sorted(names, key=os.fsencode):
            path = base / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                rows.append(
                    (
                        path.relative_to(git_dir).as_posix(),
                        "symlink",
                        os.readlink(path),
                        metadata.st_dev,
                        metadata.st_ino,
                    )
                )
                continue
            if not stat.S_ISDIR(metadata.st_mode):
                raise BootstrapError(f"unsafe Git control directory: {path}")
            rows.append(
                (
                    path.relative_to(git_dir).as_posix(),
                    "directory",
                    stat.S_IMODE(metadata.st_mode),
                    metadata.st_dev,
                    metadata.st_ino,
                )
            )
            kept.append(name)
        names[:] = kept
        for name in sorted(files, key=os.fsencode):
            path = base / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                rows.append(
                    (
                        path.relative_to(git_dir).as_posix(),
                        "symlink",
                        os.readlink(path),
                        metadata.st_dev,
                        metadata.st_ino,
                    )
                )
                continue
            fingerprint = strict_file_fingerprint(path)
            rows.append(
                (
                    path.relative_to(git_dir).as_posix(),
                    "file",
                    stat.S_IMODE(metadata.st_mode),
                    metadata.st_dev,
                    metadata.st_ino,
                    fingerprint["size_bytes"],
                    fingerprint["sha256"],
                )
            )
    rendered = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"entries": len(rows), "sha256": sha256_bytes(rendered)}


def exact_object_directory(git_dir: Path, label: str) -> Path:
    objects = Path(
        git_text(
            git_dir,
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "objects",
        )
    )
    expected = git_dir / "objects"
    if objects != expected:
        raise BootstrapError(
            f"{label} object directory escaped Git dir: expected={expected} actual={objects}"
        )
    objects = exact_real_directory(objects, f"{label} object directory")
    # A symlink below objects could redirect unpacked loose objects or local
    # reads outside the approved module object database. Reject it before
    # probing alternates so even objects/info cannot redirect that lstat.
    for directory, names, files in os.walk(objects, topdown=True, followlinks=False):
        base = Path(directory)
        kept: list[str] = []
        for name in names:
            path = base / name
            if path.is_symlink() or not path.is_dir():
                raise BootstrapError(f"unsafe object directory entry: {path}")
            kept.append(name)
        names[:] = kept
        for name in files:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise BootstrapError(f"unsafe object file entry: {path}")
    for alternate_name in ("alternates", "http-alternates"):
        alternate = objects / "info" / alternate_name
        if os.path.lexists(alternate):
            raise BootstrapError(f"{label} object alternates are forbidden: {alternate}")
    return objects


def local_config_items(git_dir: Path) -> list[tuple[str, str]]:
    payload = run_git(git_dir, "config", "--local", "--null", "--list").stdout
    items: list[tuple[str, str]] = []
    for record in payload.split(b"\0"):
        if not record:
            continue
        raw_key, separator, raw_value = record.partition(b"\n")
        if not separator:
            raise BootstrapError(f"malformed local config record: {record!r}")
        items.append(
            (
                raw_key.decode("utf-8", "surrogateescape"),
                raw_value.decode("utf-8", "surrogateescape"),
            )
        )
    return items


def exact_git_dir(path: Path, label: str) -> Path:
    path = exact_real_directory(path, label)
    absolute_git_dir = git_text(path, "rev-parse", "--absolute-git-dir")
    common_dir = git_text(
        path, "rev-parse", "--path-format=absolute", "--git-common-dir"
    )
    if Path(absolute_git_dir) != path:
        raise BootstrapError(
            f"{label} module Git dir mismatch: expected={path} actual={absolute_git_dir}"
        )
    if Path(common_dir) != path:
        raise BootstrapError(
            f"{label} must be its own module common dir: expected={path} actual={common_dir}"
        )
    exact_object_directory(path, label)
    for key, value in local_config_items(path):
        normalized_key = key.lower()
        normalized_value = value.strip().lower()
        if normalized_key == "include.path" or (
            normalized_key.startswith("includeif.")
            and normalized_key.endswith(".path")
        ):
            raise BootstrapError(f"{label} external config include routing is forbidden")
        if normalized_key == "extensions.partialclone" or normalized_key.endswith(
            ".partialclonefilter"
        ):
            raise BootstrapError(
                f"{label} partial-clone object routing is forbidden"
            )
        if normalized_key.startswith("remote.") and normalized_key.endswith(
            ".promisor"
        ) and normalized_value in {"1", "true", "yes", "on"}:
            raise BootstrapError(f"{label} promisor object routing is forbidden")
    return path


def object_type(git_dir: Path, object_id: str) -> str | None:
    completed = run_git(git_dir, "cat-file", "-t", object_id, check=False)
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("ascii", "strict").strip()


def object_exists(git_dir: Path, commit: str) -> bool:
    return object_type(git_dir, commit) == "commit"


def closure_objects(
    git_dir: Path,
    commit: str,
    *,
    require_complete: bool,
) -> tuple[set[str], set[str]]:
    commit_type = object_type(git_dir, commit)
    if commit_type is None:
        if require_complete:
            raise BootstrapError(f"repository lacks exact commit object: {commit}")
        return set(), {commit}
    if commit_type != "commit":
        raise BootstrapError(
            f"exact object is not a commit: {commit} type={commit_type}"
        )
    completed = run_git(
        git_dir,
        "rev-list",
        "--objects",
        "--no-object-names",
        "--missing=print",
        commit,
        check=False,
    )
    present: set[str] = set()
    missing: set[str] = set()
    object_format = repository_object_format(git_dir)
    expected_length = OBJECT_FORMAT_LENGTHS[object_format]
    for raw in completed.stdout.decode("ascii", "strict").splitlines():
        marker = raw.startswith("?")
        object_id = raw[1:] if marker else raw
        if len(object_id) != expected_length or not HEX_OBJECT.fullmatch(object_id):
            raise BootstrapError(f"malformed rev-list object record: {raw!r}")
        (missing if marker else present).add(object_id)
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", "replace").strip()
        if require_complete:
            raise BootstrapError(f"commit closure enumeration failed: {error}")
        missing.add(commit)
    if missing and require_complete:
        raise BootstrapError(
            f"commit closure has missing objects: count={len(missing)} first={sorted(missing)[0]}"
        )
    return present, missing


def object_closure_complete(git_dir: Path, commit: str) -> bool:
    try:
        _present, missing = closure_objects(git_dir, commit, require_complete=False)
    except BootstrapError:
        return False
    return not missing


def optional_head(git_dir: Path) -> str | None:
    completed = run_git(git_dir, "rev-parse", "--verify", "HEAD", check=False)
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("ascii", "strict").strip()


def semantic_fingerprints(git_dir: Path) -> dict[str, Any]:
    refs = run_git(
        git_dir,
        "for-each-ref",
        "--format=%(refname)%00%(objectname)%00%(objecttype)",
    ).stdout
    config = run_git(git_dir, "config", "--local", "--null", "--list").stdout
    remotes = run_git(git_dir, "remote", "-v").stdout
    fetch_head = Path(
        git_text(
            git_dir,
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "FETCH_HEAD",
        )
    )
    index = Path(
        git_text(
            git_dir,
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "index",
        )
    )
    head_file = git_dir / "HEAD"
    return {
        "refs_sha256": sha256_bytes(refs),
        "refs_size_bytes": len(refs),
        "config_sha256": sha256_bytes(config),
        "config_size_bytes": len(config),
        "remotes_sha256": sha256_bytes(remotes),
        "remotes_size_bytes": len(remotes),
        "head_file": strict_file_fingerprint(head_file),
        "index": strict_file_fingerprint(index),
        "fetch_head": strict_file_fingerprint(fetch_head),
        "non_object_control_tree": control_tree_fingerprint(git_dir),
    }


def repository_identity(git_dir: Path) -> dict[str, Any]:
    objects = exact_object_directory(git_dir, "repository")
    return {
        "git_dir": str(git_dir),
        "common_dir": git_text(
            git_dir, "rev-parse", "--path-format=absolute", "--git-common-dir"
        ),
        "head": optional_head(git_dir),
        "object_format": repository_object_format(git_dir),
        "filesystem": path_identity(git_dir),
        "objects_filesystem": path_identity(objects),
        "non_object_control_tree": control_tree_fingerprint(git_dir),
    }


def public_preflight(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if not key.startswith("_")}


def preflight(source: Path, destination: Path, commit: str) -> dict[str, Any]:
    source = exact_git_dir(source, "source")
    destination = exact_git_dir(destination, "destination")
    if source == destination or source in destination.parents or destination in source.parents:
        raise BootstrapError("source and destination module Git dirs overlap")
    if path_identity(source) == path_identity(destination):
        raise BootstrapError("source and destination module Git dirs are identical")
    source_format = repository_object_format(source)
    destination_format = repository_object_format(destination)
    if source_format != destination_format:
        raise BootstrapError(
            "source/destination object formats differ: "
            f"source={source_format} destination={destination_format}"
        )
    commit = exact_commit(commit, source_format)
    source_identity = repository_identity(source)
    destination_identity = repository_identity(destination)
    if source_identity["head"] != commit:
        raise BootstrapError(
            f"source HEAD mismatch: expected={commit} actual={source_identity['head']}"
        )
    source_objects, source_missing = closure_objects(
        source, commit, require_complete=True
    )
    if source_missing or commit not in source_objects:
        raise BootstrapError("source exact commit closure is incomplete")
    destination_object_type = object_type(destination, commit)
    if destination_object_type not in {None, "commit"}:
        raise BootstrapError(
            "destination exact object has the wrong type: "
            f"{commit} type={destination_object_type}"
        )
    destination_complete = object_closure_complete(destination, commit)
    sorted_source_objects = sorted(source_objects)
    closure_digest = sha256_bytes(("\n".join(sorted_source_objects) + "\n").encode("ascii"))
    return {
        "source": source_identity,
        "destination": destination_identity,
        "expected_commit": commit,
        "object_format": source_format,
        "source_exact_commit_exists": True,
        "source_closure_complete": True,
        "source_closure_object_count": len(sorted_source_objects),
        "source_closure_sha256": closure_digest,
        "destination_exact_commit_exists": destination_object_type == "commit",
        "destination_closure_complete": destination_complete,
        "source_fingerprints": semantic_fingerprints(source),
        "destination_fingerprints": semantic_fingerprints(destination),
        "_source_object_ids": sorted_source_objects,
    }


def assert_preflight_unchanged(before: dict[str, Any]) -> None:
    current = preflight(
        Path(before["source"]["git_dir"]),
        Path(before["destination"]["git_dir"]),
        before["expected_commit"],
    )
    for key in (
        "source",
        "destination",
        "expected_commit",
        "object_format",
        "source_closure_complete",
        "source_closure_object_count",
        "source_closure_sha256",
        "destination_exact_commit_exists",
        "destination_closure_complete",
        "source_fingerprints",
        "destination_fingerprints",
    ):
        if current[key] != before[key]:
            raise BootstrapError(f"repository state changed after preflight: {key}")


def verified_pack(
    source: Path,
    commit: str,
    expected_objects: list[str],
    directory: Path,
) -> tuple[Path, int, str]:
    pack_path = directory / "exact-closure.pack"
    index_path = directory / "exact-closure.idx"
    with pack_path.open("xb") as pack_stream:
        run_git(
            source,
            "pack-objects",
            "--stdout",
            "--revs",
            input_bytes=(commit + "\n").encode("ascii"),
            stdout=pack_stream,
        )
        pack_stream.flush()
        os.fsync(pack_stream.fileno())
    pack_size = pack_path.stat().st_size
    if pack_size <= 0:
        raise BootstrapError("source produced an empty exact object pack")
    run_git(
        source,
        "index-pack",
        "--no-rev-index",
        "-o",
        str(index_path),
        str(pack_path),
    )
    with index_path.open("rb") as index_stream:
        listing = run_git(source, "show-index", stdin=index_stream).stdout
    object_ids: set[str] = set()
    object_records = 0
    object_format = repository_object_format(source)
    expected_length = OBJECT_FORMAT_LENGTHS[object_format]
    for raw in listing.decode("ascii", "strict").splitlines():
        fields = raw.split()
        if len(fields) < 2:
            raise BootstrapError(f"malformed pack index record: {raw!r}")
        object_id = fields[1]
        if len(object_id) != expected_length or not HEX_OBJECT.fullmatch(object_id):
            raise BootstrapError(f"malformed packed object ID: {object_id!r}")
        object_records += 1
        object_ids.add(object_id)
    if object_records != len(object_ids):
        raise BootstrapError("verified pack index contains duplicate object IDs")
    expected_set = set(expected_objects)
    if object_ids != expected_set:
        raise BootstrapError(
            "pack object set differs from exact commit closure: "
            f"missing={len(expected_set - object_ids)} extra={len(object_ids - expected_set)}"
        )
    pack_fingerprint = strict_file_fingerprint(pack_path)
    return pack_path, pack_size, str(pack_fingerprint["sha256"])


def initialize_empty_module(
    destination: Path,
    worktree: Path,
    *,
    object_format: str,
    approved_exact_module_init: bool = False,
) -> dict[str, Any]:
    if not approved_exact_module_init:
        raise BootstrapError("module init requires approved_exact_module_init=True")
    object_format = exact_object_format(object_format)
    destination = lexical_absolute(destination)
    worktree = lexical_absolute(worktree)
    assert_no_symlink_components(destination)
    if os.path.lexists(destination):
        raise BootstrapError(f"destination module Git dir must not exist: {destination}")
    parent = exact_real_directory(destination.parent, "destination module parent")
    assert_directory_empty(worktree, "destination module worktree")
    parent_identity = path_identity(parent)
    worktree_identity = path_identity(worktree)
    init_help = run_git(destination, "init", "-h", check=False)
    help_payload = init_help.stdout + init_help.stderr
    if b"--initial-branch" not in help_payload or b"--object-format" not in help_payload:
        raise BootstrapError("Git lacks required init object-format/initial-branch capability")
    run_git(
        destination,
        f"--work-tree={worktree}",
        "init",
        f"--object-format={object_format}",
        "--initial-branch=main",
    )
    destination = exact_git_dir(destination, "initialized destination")
    if repository_object_format(destination) != object_format:
        raise BootstrapError("minimal module init created the wrong object format")
    configured_worktree = git_text(
        destination, "config", "--local", "--get", "core.worktree"
    )
    configured_path = Path(configured_worktree)
    if not configured_path.is_absolute():
        configured_path = lexical_absolute(destination / configured_path)
    if configured_path != worktree:
        raise BootstrapError(
            "minimal module init core.worktree mismatch: "
            f"expected={worktree} actual={configured_path}"
        )
    assert_directory_empty(worktree, "destination module worktree after init")
    if path_identity(parent) != parent_identity or path_identity(worktree) != worktree_identity:
        raise BootstrapError("module init changed parent/worktree filesystem identity")
    fingerprints = semantic_fingerprints(destination)
    if optional_head(destination) is not None:
        raise BootstrapError("minimal module init unexpectedly created a HEAD object")
    if fingerprints["refs_size_bytes"] or fingerprints["remotes_size_bytes"]:
        raise BootstrapError("minimal module init unexpectedly created a ref or remote")
    if fingerprints["fetch_head"]["kind"] != "absent":
        raise BootstrapError("minimal module init unexpectedly wrote FETCH_HEAD")
    return {
        "schema": SCHEMA,
        "operation": "minimal-empty-module-init",
        "destination": repository_identity(destination),
        "worktree": str(worktree),
        "object_format": object_format,
        "fingerprints_after_init": fingerprints,
        "worktree_remained_empty": True,
        "network_transport_available": False,
        "remote_or_ref_created": False,
        "fetch_head_written": False,
        "config_change": "minimal repository initialization only; object transfer fingerprints this baseline and must not change it",
    }


def probe(source: Path, destination: Path, commit: str) -> dict[str, Any]:
    snapshot = public_preflight(preflight(source, destination, commit))
    return {
        "schema": SCHEMA,
        "operation": "read-only-probe",
        **snapshot,
        "network_transport_available": False,
        "lazy_fetch_disabled": True,
        "object_transfer_command": [
            "git --git-dir=$EXACT_SOURCE pack-objects --stdout --revs",
            "git --git-dir=$EXACT_DESTINATION unpack-objects",
        ],
        "transfer_approval_required": True,
    }


def transfer_exact_commit(
    source: Path,
    destination: Path,
    commit: str,
    *,
    approved_exact_object_transfer: bool = False,
) -> dict[str, Any]:
    if not approved_exact_object_transfer:
        raise BootstrapError(
            "object transfer requires approved_exact_object_transfer=True"
        )
    before = preflight(source, destination, commit)
    source = Path(before["source"]["git_dir"])
    destination = Path(before["destination"]["git_dir"])
    source_identity = before["source"]
    destination_identity = before["destination"]
    source_fingerprints = before["source_fingerprints"]
    fingerprints_before = before["destination_fingerprints"]
    action = "already-present-noop"
    pack_size = 0
    pack_sha256: str | None = None
    if not before["destination_closure_complete"]:
        action = "local-pack-unpack"
        # Close the preflight-to-mutation window as far as a path-based Git CLI
        # permits. Future live use additionally requires the recorded writer
        # pause; any identity/symlink/control-state replacement stops here.
        assert_preflight_unchanged(before)
        with tempfile.TemporaryDirectory(
            prefix="nimloth-exact-object-pack-", dir="/tmp"
        ) as temporary_directory:
            pack_path, pack_size, pack_sha256 = verified_pack(
                source,
                before["expected_commit"],
                before["_source_object_ids"],
                Path(temporary_directory),
            )
            if repository_identity(source) != source_identity:
                raise BootstrapError("source module identity/HEAD changed while packing")
            if semantic_fingerprints(source) != source_fingerprints:
                raise BootstrapError("source refs/config/index/control state changed while packing")
            if repository_identity(destination) != destination_identity:
                raise BootstrapError("destination identity changed before unpack")
            if semantic_fingerprints(destination) != fingerprints_before:
                raise BootstrapError("destination refs/config/index/control state changed before unpack")
            with pack_path.open("rb") as pack_stream:
                run_git(destination, "unpack-objects", stdin=pack_stream)
    if repository_identity(source) != source_identity:
        raise BootstrapError("source module identity/HEAD changed during transfer")
    if semantic_fingerprints(source) != source_fingerprints:
        raise BootstrapError("source refs/config/index/control state changed during transfer")
    if repository_identity(destination) != destination_identity:
        raise BootstrapError("destination module identity/HEAD changed during transfer")
    destination_objects, destination_missing = closure_objects(
        destination, before["expected_commit"], require_complete=True
    )
    if destination_missing or set(before["_source_object_ids"]) != destination_objects:
        raise BootstrapError("destination exact commit closure is incomplete after transfer")
    fingerprints_after = semantic_fingerprints(destination)
    if fingerprints_after != fingerprints_before:
        raise BootstrapError(
            "object transfer changed destination HEAD/index/refs/config/remotes/FETCH_HEAD/control state"
        )
    return {
        "schema": SCHEMA,
        "operation": "exact-object-transfer",
        "action": action,
        "source": source_identity,
        "destination": destination_identity,
        "expected_commit": before["expected_commit"],
        "object_format": before["object_format"],
        "source_exact_commit_exists": True,
        "source_closure_complete": True,
        "source_closure_object_count": before["source_closure_object_count"],
        "source_closure_sha256": before["source_closure_sha256"],
        "destination_exact_commit_existed_before": before[
            "destination_exact_commit_exists"
        ],
        "destination_closure_complete_before": before[
            "destination_closure_complete"
        ],
        "destination_exact_commit_exists_after": True,
        "destination_closure_complete_after": True,
        "pack_size_bytes": pack_size,
        "pack_sha256": pack_sha256,
        "pack_object_count": before["source_closure_object_count"] if pack_size else 0,
        "pack_object_set_equals_source_closure": True,
        "fingerprints_before": fingerprints_before,
        "fingerprints_after": fingerprints_after,
        "head_index_refs_config_remotes_fetch_head_unchanged": True,
        "destination_head_unchanged": True,
        "destination_index_unchanged": True,
        "worktree_accessed_or_modified": False,
        "network_transport_available": False,
        "lazy_fetch_disabled": True,
        "remote_or_ref_created": False,
        "fetch_head_written": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "transfer"):
        command = subparsers.add_parser(name)
        command.add_argument("--source", required=True)
        command.add_argument("--destination", required=True)
        command.add_argument("--commit", required=True)
        command.add_argument("--output")
        if name == "transfer":
            command.add_argument(
                "--approved-exact-object-transfer", action="store_true"
            )
    init_command = subparsers.add_parser("init-empty")
    init_command.add_argument("--destination", required=True)
    init_command.add_argument("--worktree", required=True)
    init_command.add_argument("--object-format", required=True, choices=sorted(OBJECT_FORMAT_LENGTHS))
    init_command.add_argument("--approved-exact-module-init", action="store_true")
    init_command.add_argument("--output")
    args = parser.parse_args()
    try:
        destination = lexical_absolute(Path(args.destination))
        if args.command == "init-empty":
            result = initialize_empty_module(
                destination,
                lexical_absolute(Path(args.worktree)),
                object_format=args.object_format,
                approved_exact_module_init=args.approved_exact_module_init,
            )
        else:
            source = lexical_absolute(Path(args.source))
            if args.command == "probe":
                result = probe(source, destination, args.commit)
            else:
                result = transfer_exact_commit(
                    source,
                    destination,
                    args.commit,
                    approved_exact_object_transfer=args.approved_exact_object_transfer,
                )
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        detail = (
            f"commit={result['expected_commit']}"
            if "expected_commit" in result
            else f"destination={result['destination']['git_dir']}"
        )
        print(
            "PASS: exact local module bootstrap state verified; "
            f"operation={result['operation']} {detail}"
        )
        return 0
    except (BootstrapError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
