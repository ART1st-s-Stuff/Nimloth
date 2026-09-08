#!/usr/bin/env python3
"""Audit and materialize traceable Stage2 inputs with real, nonempty CoT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

SCHEMA = "nimloth_stage2_nonempty_cot_inputs_v1"
VERSION = "stage2-nonempty-cot-v1"
EXCLUSION_POLICY = "exclude if any assistant turn lacks a recorded nonempty CoT"
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
IM_START_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n")
IM_END = "<|im_end|>"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def cot_issue(content: Any) -> str | None:
    if not isinstance(content, str):
        return "missing_think_tag"
    match = THINK_RE.search(content)
    if match is None:
        return "missing_think_tag"
    if not match.group(1).strip():
        return "empty_think_body"
    return None


def _parse_im_messages(text: str) -> list[dict[str, str]]:
    messages = []
    position = 0
    while match := IM_START_RE.search(text, position):
        start = match.end()
        end = text.find(IM_END, start)
        messages.append(
            {
                "role": match.group(1),
                "content": text[start:] if end < 0 else text[start:end],
            }
        )
        position = len(text) if end < 0 else end + len(IM_END)
    return messages


def _parse_output_messages(text: str) -> list[dict[str, str]]:
    if text.startswith("<|im_start|>"):
        return _parse_im_messages(text)
    first = text.find("<|im_start|>")
    leading, rest = (text, "") if first < 0 else (text[:first], text[first:])
    leading = leading.removesuffix(IM_END).strip("\n")
    messages = [{"role": "assistant", "content": leading}] if leading else []
    return messages + _parse_im_messages(rest)


@cache
def _source_rows(path: Path) -> dict[int, dict[str, Any]]:
    return {line_number - 1: row for line_number, row in _jsonl(path)}


def _source_assistants(record: dict[str, Any]) -> list[str] | None:
    raw_path = record.get("source_jsonl")
    raw_index = record.get("source_line_index")
    if not isinstance(raw_path, str) or not isinstance(raw_index, int):
        return None
    path = Path(raw_path)
    if not path.is_file():
        return None
    source = _source_rows(path).get(raw_index)
    if source is None:
        raise ValueError(f"source row is missing: {path}:{raw_index + 1}")
    input_messages = _parse_im_messages(source.get("input", ""))
    if (
        input_messages
        and input_messages[-1]["role"] == "assistant"
        and not input_messages[-1]["content"].strip()
    ):
        input_messages.pop()
    messages = input_messages + _parse_output_messages(source.get("output", ""))
    return [
        message["content"] for message in messages if message["role"] == "assistant"
    ]


def audit_record(
    record: dict[str, Any], *, split: str, line_number: int, source_path: Path
) -> list[dict[str, Any]]:
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"{source_path}:{line_number}: missing record id")
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise TypeError(f"{source_path}:{line_number}: messages must be a list")
    assistants = [
        (index, message)
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not assistants:
        raise ValueError(f"{source_path}:{line_number}: no assistant turns")
    source_assistants = _source_assistants(record)
    if source_assistants is not None and len(source_assistants) != len(assistants):
        raise ValueError(
            f"{source_path}:{line_number}: source/converted assistant count mismatch "
            f"{len(source_assistants)} != {len(assistants)}"
        )
    invalid = []
    for ordinal, (message_index, message) in enumerate(assistants):
        converted_content = message.get("content")
        converted_issue = cot_issue(converted_content)
        source_content = (
            source_assistants[ordinal] if source_assistants is not None else None
        )
        source_issue = (
            cot_issue(source_content)
            if source_assistants is not None
            else "source_unavailable"
        )
        converted_cot = (
            THINK_RE.search(converted_content).group(1).strip()
            if converted_issue is None and isinstance(converted_content, str)
            else None
        )
        source_cot = (
            THINK_RE.search(source_content).group(1).strip()
            if source_issue is None and isinstance(source_content, str)
            else None
        )
        if (
            converted_issue is None
            and source_issue is None
            and converted_cot == source_cot
        ):
            continue
        reason = source_issue
        if source_issue is None and converted_issue is not None:
            reason = "converted_cot_missing_from_nonempty_source"
        elif source_issue is None and converted_cot != source_cot:
            reason = "converted_cot_differs_from_source"
        invalid.append(
            {
                "assistant_ordinal": ordinal,
                "message_index": message_index,
                "converted_reason": converted_issue,
                "reason": reason,
            }
        )
    if not invalid:
        return []
    return [
        {
            "split": split,
            "record_id": record_id,
            "input_path": str(source_path.resolve()),
            "input_line": line_number,
            "original_source_jsonl": record.get("source_jsonl"),
            "original_source_line": (
                record.get("source_line_index") + 1
                if isinstance(record.get("source_line_index"), int)
                else None
            ),
            "invalid_turns": invalid,
        }
    ]


def _count(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "trajectories": len(records),
        "assistant_turns": sum(
            message.get("role") == "assistant"
            for record in records
            for message in record["messages"]
        ),
        "image_paths": sum(len(record.get("image_paths", [])) for record in records),
    }


def materialize(args: argparse.Namespace) -> int:
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing Stage2 inputs: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "conversion_version": VERSION,
            "exclusion_unit": "whole_trajectory",
            "exclusion_policy": EXCLUSION_POLICY,
            "splits": {},
        }
        exclusions: list[dict[str, Any]] = []
        for split, source_arg in (
            ("train", args.train_source),
            ("val", args.val_source),
        ):
            source = Path(source_arg)
            records = [record for _, record in _jsonl(source)]
            ids = [record.get("id") for record in records]
            if any(
                not isinstance(record_id, str) or not record_id for record_id in ids
            ):
                raise ValueError(f"{source}: every trajectory requires a nonempty id")
            if len(set(ids)) != len(ids):
                raise ValueError(f"{source}: duplicate trajectory id")
            split_exclusions = [
                exclusion
                for line_number, record in _jsonl(source)
                for exclusion in audit_record(
                    record,
                    split=split,
                    line_number=line_number,
                    source_path=source,
                )
            ]
            excluded_ids = {row["record_id"] for row in split_exclusions}
            retained = [row for row in records if row["id"] not in excluded_ids]
            if not retained:
                raise ValueError(f"{split} Stage2 split has no eligible trajectories")
            destination = temporary / f"{split}.jsonl"
            with destination.open("w", encoding="utf-8") as handle:
                for record in retained:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            exclusions.extend(split_exclusions)
            manifest["splits"][split] = {
                "source": str(source.resolve()),
                "source_sha256": sha256(source),
                "before": _count(records),
                "after": _count(retained),
                "excluded_trajectories": len(split_exclusions),
                "output": destination.name,
                "output_sha256": sha256(destination),
            }
        exclusions_path = temporary / "exclusions.jsonl"
        with exclusions_path.open("w", encoding="utf-8") as handle:
            for exclusion in exclusions:
                handle.write(json.dumps(exclusion, ensure_ascii=False) + "\n")
        manifest["exclusions"] = {
            "path": exclusions_path.name,
            "sha256": sha256(exclusions_path),
            "trajectories": len(exclusions),
            "invalid_turns": sum(len(row["invalid_turns"]) for row in exclusions),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, sort_keys=True))
    return 0


def validate(args: argparse.Namespace) -> int:
    root = Path(args.input_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("conversion_version") != VERSION
    ):
        raise ValueError("Stage2 input manifest schema/version mismatch")
    if (
        manifest.get("exclusion_unit") != "whole_trajectory"
        or manifest.get("exclusion_policy") != EXCLUSION_POLICY
    ):
        raise ValueError("Stage2 input exclusion policy mismatch")
    required_images: set[str] = set()
    expected_exclusions: list[dict[str, Any]] = []
    for split in ("train", "val"):
        entry = manifest.get("splits", {}).get(split, {})
        source = Path(entry.get("source", ""))
        output = root / entry.get("output", "")
        expected_source = Path(getattr(args, f"expected_{split}_source")).resolve()
        if source.resolve() != expected_source:
            raise ValueError(f"{split} Stage2 source identity mismatch")
        if output.resolve() != (root / f"{split}.jsonl").resolve():
            raise ValueError(f"{split} Stage2 output identity mismatch")
        if sha256(source) != entry.get("source_sha256"):
            raise ValueError(f"{split} Stage2 source hash mismatch")
        if sha256(output) != entry.get("output_sha256"):
            raise ValueError(f"{split} Stage2 output hash mismatch")
        source_records = [record for _, record in _jsonl(source)]
        records = [record for _, record in _jsonl(output)]
        source_ids = [record.get("id") for record in source_records]
        if any(
            not isinstance(record_id, str) or not record_id for record_id in source_ids
        ) or len(source_ids) != len(set(source_ids)):
            raise ValueError(f"{split} Stage2 source trajectory IDs are invalid")
        if not records:
            raise ValueError(f"{split} Stage2 split has no eligible trajectories")
        split_exclusions = [
            exclusion
            for line_number, record in _jsonl(source)
            for exclusion in audit_record(
                record, split=split, line_number=line_number, source_path=source
            )
        ]
        excluded_ids = {row["record_id"] for row in split_exclusions}
        expected_records = [
            row for row in source_records if row.get("id") not in excluded_ids
        ]
        if records != expected_records:
            raise ValueError(
                f"{split} Stage2 output does not match whole-trajectory audit"
            )
        if entry.get("before") != _count(source_records):
            raise ValueError(f"{split} Stage2 source counts mismatch")
        if _count(records) != entry.get("after"):
            raise ValueError(f"{split} Stage2 output counts mismatch")
        if entry.get("excluded_trajectories") != len(split_exclusions):
            raise ValueError(f"{split} Stage2 exclusion count mismatch")
        expected_exclusions.extend(split_exclusions)
        for line_number, record in _jsonl(output):
            if audit_record(
                record, split=split, line_number=line_number, source_path=output
            ):
                raise ValueError(
                    f"{split} audited Stage2 output still contains invalid CoT"
                )
            required_images.update(
                str(Path(path).expanduser().resolve())
                for path in record.get("image_paths", [])
            )
    exclusion = manifest.get("exclusions", {})
    exclusions_path = root / exclusion.get("path", "")
    if exclusions_path.resolve() != (root / "exclusions.jsonl").resolve():
        raise ValueError("Stage2 exclusions output identity mismatch")
    exclusions = [row for _, row in _jsonl(exclusions_path)]
    if (
        sha256(exclusions_path) != exclusion.get("sha256")
        or exclusions != expected_exclusions
        or len(exclusions) != exclusion.get("trajectories")
        or sum(len(row["invalid_turns"]) for row in exclusions)
        != exclusion.get("invalid_turns")
    ):
        raise ValueError("Stage2 exclusions sidecar mismatch")
    from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY, CachedDINOGridTargets

    targets = CachedDINOGridTargets.from_cache_root(
        args.dino_cache_root, identity=DINOV2_LARGE_IDENTITY, grid_size=4
    )
    missing = sorted(required_images - set(targets.path_to_feature))
    if missing:
        raise ValueError(
            f"DINO cache misses {len(missing)} retained images; first={missing[0]}"
        )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "required_images": len(required_images),
                "dino_missing": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("materialize")
    create.add_argument("--train-source", required=True)
    create.add_argument("--val-source", required=True)
    create.add_argument("--output-root", required=True)
    create.set_defaults(func=materialize)
    check = commands.add_parser("validate")
    check.add_argument("--input-root", required=True)
    check.add_argument("--dino-cache-root", required=True)
    check.add_argument("--expected-train-source", required=True)
    check.add_argument("--expected-val-source", required=True)
    check.set_defaults(func=validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
