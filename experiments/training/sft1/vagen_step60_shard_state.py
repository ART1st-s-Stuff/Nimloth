#!/usr/bin/env python3
"""Fail-closed run-root and batched shard checks for step60 orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INSPECTION_HANDOFF_FORMAT = "vagen_step60_gate_inspection_handoff_v1"
NON_RESUMABLE_MARKER = "NON_RESUMABLE.json"


@dataclass(frozen=True)
class OutputItem:
    label: str
    output_dir: Path
    run_id: str
    selector: str
    index: int
    format_failure_policy: str
    concurrency: int


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_new(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def mark_run_non_resumable(
    run_root: Path,
    *,
    exit_code: int,
    attempt_id: str,
    orchestrator_signal: str | None = None,
) -> None:
    """Durably prohibit resume unless rc143 came from an observed TERM/INT."""
    if orchestrator_signal not in {None, "TERM", "INT"}:
        raise ValueError("orchestrator_signal must be TERM, INT, or absent")
    if exit_code == 0 or (
        exit_code == 143 and orchestrator_signal in {"TERM", "INT"}
    ):
        return
    payload = {
        "format": "vagen_step60_gate_non_resumable_v1",
        "attempt_id": attempt_id,
        "exit_code": exit_code,
        "orchestrator_signal": orchestrator_signal,
    }
    content = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    marker = run_root.absolute() / NON_RESUMABLE_MARKER
    if marker.exists() or marker.is_symlink():
        return
    _atomic_write_new(marker, content)


def initialize_run_root(run_root: Path, *, run_mode: str, identity: bytes) -> None:
    """Create or validate the durable, non-overwriting gate run layout."""
    run_root = run_root.absolute()
    parent = run_root.parent
    if run_mode == "fresh":
        run_root.mkdir()
        for child in ("attempts", "smoke", "gate"):
            (run_root / child).mkdir()
        identity_tmp = run_root / f".RUN_IDENTITY.tmp-{os.getpid()}"
        fd = os.open(identity_tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with os.fdopen(fd, "wb", closefd=True) as stream:
                stream.write(identity)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(identity_tmp, run_root / "RUN_IDENTITY")
        finally:
            if identity_tmp.exists():
                identity_tmp.unlink()
        identity_fd = os.open(run_root / "RUN_IDENTITY", os.O_RDONLY)
        try:
            os.fsync(identity_fd)
        finally:
            os.close(identity_fd)
        for child in ("attempts", "smoke", "gate"):
            _fsync_directory(run_root / child)
        _fsync_directory(run_root)
        _fsync_directory(parent)
        return
    if run_mode != "resume":
        raise ValueError("run_mode must be fresh or resume")
    if run_root.is_symlink() or not run_root.is_dir():
        raise ValueError("resume requires an existing real RUN_ROOT")
    non_resumable = run_root / NON_RESUMABLE_MARKER
    if non_resumable.exists() or non_resumable.is_symlink():
        raise ValueError("resume rejected because RUN_ROOT is non-resumable")
    identity_path = run_root / "RUN_IDENTITY"
    if identity_path.is_symlink() or not identity_path.is_file():
        raise ValueError("resume requires a regular RUN_IDENTITY")
    if identity_path.read_bytes() != identity:
        raise ValueError("resume RUN_ROOT identity mismatch")
    for child in ("attempts", "smoke", "gate"):
        path = run_root / child
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"resume RUN_ROOT layout is invalid: {child}")


def verify_nimloth_checkout(root: Path, expected_commit: str) -> None:
    root = root.absolute()
    top = Path(
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    if top != root:
        raise ValueError("NIMLOTH_ROOT is not the Git worktree top-level")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != expected_commit:
        raise ValueError("Nimloth HEAD differs from EXPECTED_NIMLOTH_COMMIT")
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    if status:
        raise ValueError("NIMLOTH_ROOT must be clean")


def _identity_bytes(fields: Sequence[str]) -> bytes:
    values: dict[str, str] = {}
    for field in fields:
        key, separator, value = field.partition("=")
        if not separator or not key or key in values:
            raise ValueError("identity fields must be unique key=value pairs")
        values[key] = value
    payload = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    envelope = {
        "format": "vagen_step60_gate_run_identity_v2",
        "payload": values,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return (
        json.dumps(
            envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )


def _parse_item(value: str) -> OutputItem:
    parts = value.split("|")
    if len(parts) != 7 or any("|" in part for part in parts):
        raise ValueError("--item must have seven pipe-delimited fields")
    label, output, run_id, selector, index, policy, concurrency = parts
    if selector not in {"source-index", "shard-index"}:
        raise ValueError("item selector must be source-index or shard-index")
    return OutputItem(
        label=label,
        output_dir=Path(output),
        run_id=run_id,
        selector=selector,
        index=int(index),
        format_failure_policy=policy,
        concurrency=int(concurrency),
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _episode_payload(spec: Any) -> dict[str, Any]:
    return {
        "source_index": spec.source_index,
        "eval_set": spec.eval_set,
        "seed": spec.seed,
        "dataset_split": spec.dataset_split,
        "source_key": spec.source_key,
    }


def _handoff_bindings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "partition_manifest": str(args.partition_manifest.absolute()),
        "partition_manifest_file_sha256": hashlib.sha256(
            args.partition_manifest.read_bytes()
        ).hexdigest(),
        "shard_size": args.shard_size,
        "source_runtime_root": str(args.source_runtime_root.absolute()),
        "source_runtime_contract": str(args.source_runtime_contract.absolute()),
        "source_runtime_contract_file_sha256": hashlib.sha256(
            args.source_runtime_contract.read_bytes()
        ).hexdigest(),
        "model_path": str(args.model_path.absolute()),
        "expected_runtime_contract_payload_sha256": args.expected_runtime_contract_payload_sha256,
        "expected_reconstruction_head": args.expected_reconstruction_head,
        "expected_reconstruction_tree": args.expected_reconstruction_tree,
        "expected_reconstruction_diff_sha256": args.expected_reconstruction_diff_sha256,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "engine_seed": args.engine_seed,
    }


def _write_inspection_handoff(
    path: Path,
    *,
    bindings: dict[str, Any],
    context: dict[str, Any],
    items: Sequence[tuple[OutputItem, Sequence[Any]]],
) -> None:
    payload = {
        "bindings": bindings,
        "inspection_context": context,
        "items": [
            {
                "label": item.label,
                "output_dir": str(item.output_dir.absolute()),
                "run_id": item.run_id,
                "selector": item.selector,
                "index": item.index,
                "format_failure_policy": item.format_failure_policy,
                "concurrency": item.concurrency,
                "ordered_episode_specs": [_episode_payload(spec) for spec in specs],
            }
            for item, specs in items
        ],
    }
    envelope = {
        "format": INSPECTION_HANDOFF_FORMAT,
        "payload": payload,
        "payload_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
    }
    _atomic_write_new(path.absolute(), _canonical_bytes(envelope) + b"\n")


def _run_batch(args: argparse.Namespace) -> None:
    # Run-root ownership checks stay stdlib-only; deep rollout dependencies load here.
    from experiments.training.sft1.vagen_step60_collect import (
        EpisodeSpec,
        batch1_shard_specs_from_manifest,
        batch1_smoke_spec_from_manifest,
        build_inspected_collector,
        prepare_collection_inspection_context,
    )
    from experiments.training.sft1.vagen_step60_data import (
        load_published_partition_manifest,
    )

    items = [_parse_item(value) for value in args.item]
    if not items or len({item.label for item in items}) != len(items):
        raise ValueError("batch items must have unique labels")
    if args.command == "classify-batch":
        context = prepare_collection_inspection_context(
            model_path=args.model_path,
            source_runtime_root=args.source_runtime_root,
            source_runtime_contract=args.source_runtime_contract,
            expected_reconstruction_head=args.expected_reconstruction_head,
            expected_reconstruction_tree=args.expected_reconstruction_tree,
            expected_reconstruction_diff_sha256=args.expected_reconstruction_diff_sha256,
            expected_runtime_contract_payload_sha256=args.expected_runtime_contract_payload_sha256,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            engine_seed=args.engine_seed,
        )
        partition = load_published_partition_manifest(args.partition_manifest)
        inspected_items: list[tuple[OutputItem, Sequence[Any]]] = []
    else:
        from experiments.training.sft1.vagen_step60_collect import (
            load_inspection_handoff,
        )

        handoff = load_inspection_handoff(
            args.inspection_handoff,
            expected_file_sha256=args.expected_inspection_handoff_sha256,
            expected_bindings=_handoff_bindings(args),
        )
        context = handoff["inspection_context"]
        by_label = {entry["label"]: entry for entry in handoff["items"]}
        partition = None
        inspected_items = []
    for item in items:
        if args.command == "classify-batch":
            specs = (
                [batch1_smoke_spec_from_manifest(partition, source_index=item.index)]
                if item.selector == "source-index"
                else batch1_shard_specs_from_manifest(
                    partition, shard_index=item.index, shard_size=args.shard_size
                )
            )
        else:
            entry = by_label.get(item.label)
            if entry is None:
                raise ValueError(f"inspection handoff lacks item: {item.label}")
            expected_item = {
                "label": item.label,
                "output_dir": str(item.output_dir.absolute()),
                "run_id": item.run_id,
                "selector": item.selector,
                "index": item.index,
                "format_failure_policy": item.format_failure_policy,
                "concurrency": item.concurrency,
            }
            if {key: entry.get(key) for key in expected_item} != expected_item:
                raise ValueError(f"inspection handoff item binding mismatch: {item.label}")
            specs = [
                EpisodeSpec(**payload) for payload in entry["ordered_episode_specs"]
            ]
        collector = build_inspected_collector(
            context,
            shard_index=0 if item.selector == "source-index" else item.index,
            run_id=item.run_id,
            format_failure_policy=item.format_failure_policy,
            concurrency=item.concurrency,
        )
        if args.command == "classify-batch":
            state = collector.inspect_output_state(specs, output_dir=item.output_dir)
            inspected_items.append((item, specs))
            print(f"{item.label}\t{state}")
        elif args.command == "validate-batch":
            collector.validate_output(specs, output_dir=item.output_dir)
    if args.command == "classify-batch":
        _write_inspection_handoff(
            args.inspection_handoff,
            bindings=_handoff_bindings(args),
            context=context,
            items=inspected_items,
        )


def _add_common_batch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--item", action="append", required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--source-runtime-root", type=Path, required=True)
    parser.add_argument("--source-runtime-contract", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--expected-runtime-contract-payload-sha256", required=True)
    parser.add_argument("--expected-reconstruction-head", required=True)
    parser.add_argument("--expected-reconstruction-tree", required=True)
    parser.add_argument("--expected-reconstruction-diff-sha256", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--engine-seed", type=int, required=True)
    parser.add_argument("--inspection-handoff", type=Path, required=True)
    parser.add_argument("--expected-inspection-handoff-sha256")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize-run")
    initialize.add_argument("run_root", type=Path)
    initialize.add_argument("--run-mode", choices=("fresh", "resume"), required=True)
    initialize.add_argument("--nimloth-root", type=Path, required=True)
    initialize.add_argument("--expected-nimloth-commit", required=True)
    initialize.add_argument("--identity-field", action="append", default=[])
    finalize = subparsers.add_parser("finalize-run")
    finalize.add_argument("run_root", type=Path)
    finalize.add_argument("--exit-code", type=int, required=True)
    finalize.add_argument("--attempt-id", required=True)
    finalize.add_argument("--orchestrator-signal", choices=("TERM", "INT"))
    verify = subparsers.add_parser("verify-nimloth")
    verify.add_argument("--nimloth-root", type=Path, required=True)
    verify.add_argument("--expected-nimloth-commit", required=True)
    for command in ("classify-batch", "verify-handoff", "validate-batch"):
        _add_common_batch_arguments(subparsers.add_parser(command))
    args = parser.parse_args()
    if args.command == "initialize-run":
        verify_nimloth_checkout(args.nimloth_root, args.expected_nimloth_commit)
        initialize_run_root(
            args.run_root,
            run_mode=args.run_mode,
            identity=_identity_bytes(args.identity_field),
        )
    elif args.command == "finalize-run":
        mark_run_non_resumable(
            args.run_root,
            exit_code=args.exit_code,
            attempt_id=args.attempt_id,
            orchestrator_signal=args.orchestrator_signal,
        )
    elif args.command == "verify-nimloth":
        verify_nimloth_checkout(args.nimloth_root, args.expected_nimloth_commit)
    else:
        if args.command == "classify-batch" and args.expected_inspection_handoff_sha256:
            parser.error("classify-batch does not accept a handoff SHA")
        if args.command in {"verify-handoff", "validate-batch"} and not args.expected_inspection_handoff_sha256:
            parser.error(f"{args.command} requires the expected handoff SHA")
        _run_batch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
