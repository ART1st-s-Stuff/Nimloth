#!/usr/bin/env python3
"""Generate one non-overwriting, Git-computed reconstruction runtime contract."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from experiments.training.sft1.vagen_step60_collect import (
    APPROVED_RECONSTRUCTION_DIFF_SHA256,
    APPROVED_RECONSTRUCTION_HEAD,
    APPROVED_RECONSTRUCTION_TREE,
    RECONSTRUCTION_BASE_COMMIT,
    build_source_runtime_contract,
    reconstruction_git_identity,
    validate_reconstruction_git_identity,
    validate_source_runtime_contract,
)


def _publish_no_replace(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"runtime contract output already exists: {path}")
    fd, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(
                f"runtime contract output already exists: {path}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-diff-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    supplied = {
        "runtime_head": args.expected_head,
        "runtime_tree": args.expected_tree,
        "diff_sha256": args.expected_diff_sha256,
    }
    approved = {
        "runtime_head": APPROVED_RECONSTRUCTION_HEAD,
        "runtime_tree": APPROVED_RECONSTRUCTION_TREE,
        "diff_sha256": APPROVED_RECONSTRUCTION_DIFF_SHA256,
    }
    if supplied != approved:
        raise ValueError("runtime contract inputs differ from approved VAGEN literals")
    actual = reconstruction_git_identity(
        args.runtime_root,
        base_commit=RECONSTRUCTION_BASE_COMMIT,
    )
    expected = {
        **actual,
        "base_commit": RECONSTRUCTION_BASE_COMMIT,
        "runtime_parent": RECONSTRUCTION_BASE_COMMIT,
        **approved,
        "commit_count": 1,
        "parent_count": 1,
    }
    validate_reconstruction_git_identity(actual, expected=expected)
    contract = build_source_runtime_contract(
        runtime_root=args.runtime_root,
        reconstruction_identity=actual,
    )
    validate_source_runtime_contract(
        contract,
        expected_reconstruction_identity=expected,
        expected_runtime_root=args.runtime_root,
    )
    _publish_no_replace(args.output, contract)
    print(contract["contract_payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
