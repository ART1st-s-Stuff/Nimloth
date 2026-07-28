"""独立复核 trajectory JSONL 迁移产物及其 manifest。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from itertools import zip_longest
from pathlib import Path
from typing import Any, TextIO

from nimloth.rollout.migration import migrate_trajectory_record
from nimloth.rollout.record_format import (
    REWARD_PROVENANCE_VALUES,
    TRAJECTORY_RECORD_FORMAT,
    require_trajectory_record,
)
from nimloth.rollout.transitions import expand_record_transitions


def _open_jsonl(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _records(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    with _open_jsonl(path) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield line_number, record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_trajectory_migration(
    *,
    source_path: Path,
    output_path: Path,
    missing_action_space_id: str | None,
    missing_action_space_version: int | None,
    missing_reward_provenance: str | None,
) -> dict[str, Any]:
    """逐记录确认迁移产物与确定性迁移结果完全一致。"""

    source_path = source_path.resolve()
    output_path = output_path.resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if not source_path.is_file() or not output_path.is_file():
        raise FileNotFoundError("source and migrated JSONL must both exist")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"migration manifest is missing: {manifest_path}")

    record_count = 0
    transition_count = 0
    seen_ids: set[str] = set()
    pairs = zip_longest(_records(source_path), _records(output_path))
    for source_item, output_item in pairs:
        if source_item is None or output_item is None:
            raise ValueError("source and migrated JSONL record counts differ")
        source_line, source_record = source_item
        output_line, output_record = output_item
        expected = migrate_trajectory_record(
            source_record,
            missing_action_space_id=missing_action_space_id,
            missing_action_space_version=missing_action_space_version,
            missing_reward_provenance=missing_reward_provenance,
        )
        require_trajectory_record(output_record)
        record_id = str(output_record["id"])
        if output_record != expected:
            raise ValueError(
                "migrated record differs from deterministic migration: "
                f"source_line={source_line} output_line={output_line} id={record_id}"
            )
        if record_id in seen_ids:
            raise ValueError(f"duplicate trajectory id in migrated JSONL: {record_id}")
        seen_ids.add(record_id)

        actions = source_record["action_indices"]
        if output_record["action_indices"] != actions:
            raise ValueError(f"action sequence changed during migration: {record_id}")
        if output_record.get("terminal_assistant_prefix") != source_record.get(
            "terminal_assistant_prefix"
        ):
            raise ValueError(f"terminal CoT changed during migration: {record_id}")
        transitions = expand_record_transitions(output_record, value_gamma=1.0)
        if len(transitions) != len(actions):
            raise ValueError(f"transition count changed during migration: {record_id}")
        if "terminal_assistant_prefix" in source_record:
            if not transitions:
                raise ValueError(
                    f"terminal CoT trajectory has no transition: {record_id}"
                )
            terminal_messages = transitions[-1].next_prefix_messages
            if (
                terminal_messages is None
                or terminal_messages[-1]["content"]
                != source_record["terminal_assistant_prefix"]
            ):
                raise ValueError(
                    f"final transition does not target terminal CoT: {record_id}"
                )
        record_count += 1
        transition_count += len(transitions)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "format": "nimloth_trajectory_migration_v1",
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "record_count": record_count,
        "target_record_format": TRAJECTORY_RECORD_FORMAT,
        "declared_missing_fields": {
            "action_space_id": missing_action_space_id,
            "action_space_version": missing_action_space_version,
            "reward_provenance": missing_reward_provenance,
        },
    }
    if manifest != expected_manifest:
        raise ValueError("migration manifest does not match source/output artifacts")
    return {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "source_sha256": expected_manifest["source_sha256"],
        "output_sha256": expected_manifest["output_sha256"],
        "record_count": record_count,
        "transition_count": transition_count,
        "ids_unique": True,
        "actions_equal": True,
        "terminal_cot_equal": True,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--missing-action-space-id")
    parser.add_argument("--missing-action-space-version", type=int)
    parser.add_argument(
        "--missing-reward-provenance",
        choices=sorted(REWARD_PROVENANCE_VALUES),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    result = validate_trajectory_migration(
        source_path=args.source,
        output_path=args.output,
        missing_action_space_id=args.missing_action_space_id,
        missing_action_space_version=args.missing_action_space_version,
        missing_reward_provenance=args.missing_reward_provenance,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["main", "parse_args", "validate_trajectory_migration"]
