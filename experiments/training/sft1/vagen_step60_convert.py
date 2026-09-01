#!/usr/bin/env python3
"""Convert verified VAGEN step60 raw shards into dual-view SFT1/SFT2 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from experiments.training.sft1.vagen_step60_data import (
    CONVERSION_FORMAT,
    REJECTION_FORMAT,
    atomic_publish_directory,
    build_source_audit,
    convert_source_record,
    validate_complete_shard,
    validate_partition_manifest,
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path} contains blank line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path} line {line_number} is not a mapping")
            rows.append(value)
    return rows


def _validate_sft1_record(record: dict[str, Any], *, latent_token_count: int) -> None:
    messages = record.get("messages")
    actions = record.get("action_indices")
    images = record.get("image_paths")
    if not isinstance(messages, list) or not messages:
        raise ValueError("SFT1 record has no messages")
    if messages[0].get("role") != "system":
        raise ValueError("SFT1 record does not start with system")
    if not isinstance(actions, list) or not isinstance(images, list):
        raise TypeError("SFT1 record action/image fields must be lists")
    assistants = [message for message in messages if message.get("role") == "assistant"]
    users = [message for message in messages if message.get("role") == "user"]
    if len(assistants) != len(actions) or len(users) != len(actions):
        raise ValueError("SFT1 assistant/user/action counts do not align")
    if len(images) != len(actions):
        raise ValueError("SFT1 image/action counts do not align")
    if not record.get("source_audit") or not record.get("conversion_provenance"):
        raise ValueError("SFT1 record has no source/conversion audit")
    for message in messages:
        content = str(message.get("content", ""))
        if "<answer>" in content or "</answer>" in content:
            raise ValueError("SFT1 training view still contains source answer tags")
    for response in assistants:
        content = str(response["content"])
        if content.count("<|latent_state|>") != 1:
            raise ValueError("SFT1 assistant has no canonical first latent token")
        if content.count("<|latent_state_") != latent_token_count - 1:
            raise ValueError("SFT1 assistant has the wrong K latent tokens")
        if "<|action_start|>" not in content or "<|action_end|>" not in content:
            raise ValueError("SFT1 assistant has no complete action envelope")


def _load_verified_records(
    shard_dirs: Sequence[Path],
    *,
    expected_by_index: dict[int, dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], Path]], list[dict[str, Any]]]:
    records: list[tuple[dict[str, Any], Path]] = []
    shard_evidence: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    common_shard_contract: dict[str, Any] | None = None
    for shard_dir_value in shard_dirs:
        shard_dir = shard_dir_value.resolve()
        raw_manifest = json.loads(
            (shard_dir / "shard_manifest.json").read_text(encoding="utf-8")
        )
        source_indices = {int(value) for value in raw_manifest["source_indices"]}
        manifest = validate_complete_shard(
            shard_dir,
            expected_source_indices=source_indices,
        )
        shard_contract = {
            field_name: manifest.get(field_name)
            for field_name in (
                "unavailable_source_commit",
                "reconstruction_identity",
                "source_runtime_contract",
                "policy_artifact",
                "policy_runtime_contract",
                "format_failure_policy",
            )
        }
        if common_shard_contract is None:
            common_shard_contract = shard_contract
        elif shard_contract != common_shard_contract:
            raise ValueError("complete shards mix runtime or policy provenance")
        duplicate = seen_indices & source_indices
        if duplicate:
            raise ValueError(f"duplicate source indices across shards: {sorted(duplicate)}")
        unexpected = source_indices - set(expected_by_index)
        if unexpected:
            raise ValueError(f"shard contains non-batch1 indices: {sorted(unexpected)}")
        rows = _read_jsonl(shard_dir / "raw.jsonl")
        for row in rows:
            source_index = int(row["source_index"])
            expected = expected_by_index[source_index]
            if str(row["eval_set"]) != str(expected["eval_set"]):
                raise ValueError(f"source index {source_index} eval_set drift")
            if int(row["seed"]) != int(expected["seed"]):
                raise ValueError(f"source index {source_index} seed drift")
            if str(row["split"]) != str(expected["dataset_split"]):
                raise ValueError(f"source index {source_index} split drift")
            if str(row.get("source_key", "")) != str(expected["source_key"]):
                raise ValueError(f"source index {source_index} source_key drift")
            if int(row.get("batch", -1)) != int(expected["batch"]):
                raise ValueError(f"source index {source_index} batch drift")
            records.append((row, shard_dir))
        seen_indices.update(source_indices)
        shard_evidence.append(
            {
                "path": str(shard_dir),
                "manifest_sha256": _file_sha256(
                    shard_dir / "shard_manifest.json"
                ),
                "raw_jsonl_sha256": manifest["raw_jsonl"]["sha256"],
                "source_indices": manifest["source_indices"],
            }
        )
    expected_indices = set(expected_by_index)
    if seen_indices != expected_indices:
        raise ValueError(
            "complete shards do not cover exact batch1: "
            f"missing={sorted(expected_indices - seen_indices)}, "
            f"extra={sorted(seen_indices - expected_indices)}"
        )
    if len(records) != len(expected_indices):
        raise ValueError("raw record count does not equal unique batch1 identities")
    return records, shard_evidence


def _dataset_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "success": sum(bool(record.get("success")) for record in records),
        "assistant_turns": sum(
            len(record.get("assistant_responses", []))
            if "assistant_responses" in record
            else sum(
                message.get("role") == "assistant"
                for message in record.get("messages", [])
            )
            for record in records
        ),
        "actions": sum(len(record.get("action_indices", [])) for record in records),
        "images": sum(len(record.get("image_paths", [])) for record in records),
        "transitions": sum(len(record.get("action_indices", [])) for record in records),
    }


def validate_conversion_output(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "conversion_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("conversion output has no manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != CONVERSION_FORMAT:
        raise ValueError("conversion manifest format mismatch")
    claimed = manifest.get("manifest_payload_sha256")
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_payload_sha256"
    }
    if claimed != _canonical_sha256(payload):
        raise ValueError("conversion manifest payload hash mismatch")
    source_shards = manifest.get("source_shards")
    if not isinstance(source_shards, list) or not source_shards:
        raise ValueError("conversion manifest has no source shard evidence")
    source_shard_indices: set[int] = set()
    raw_source_by_index: dict[int, dict[str, Any]] = {}
    for evidence in source_shards:
        if not isinstance(evidence, dict):
            raise TypeError("conversion source shard evidence must be a mapping")
        shard_dir = Path(str(evidence.get("path", ""))).resolve()
        manifest_path = shard_dir / "shard_manifest.json"
        if not manifest_path.is_file() or _file_sha256(manifest_path) != evidence.get(
            "manifest_sha256"
        ):
            raise ValueError("conversion source shard manifest hash/path mismatch")
        expected_indices = {int(value) for value in evidence.get("source_indices", [])}
        verified = validate_complete_shard(
            shard_dir,
            expected_source_indices=expected_indices,
        )
        if verified["raw_jsonl"]["sha256"] != evidence.get("raw_jsonl_sha256"):
            raise ValueError("conversion source shard raw hash mismatch")
        if source_shard_indices & expected_indices:
            raise ValueError("conversion source shard indices overlap")
        raw_rows = _read_jsonl(shard_dir / "raw.jsonl")
        for raw_row in raw_rows:
            source_index = int(raw_row["source_index"])
            if source_index in raw_source_by_index:
                raise ValueError("conversion source raw identity duplicates")
            raw_source_by_index[source_index] = {
                "id": raw_row["id"],
                "raw_record_sha256": raw_row["raw_record_sha256"],
                "source_audit": build_source_audit(raw_row),
                "source_key": raw_row["source_key"],
                "eval_set": raw_row["eval_set"],
                "seed": int(raw_row["seed"]),
                "split": raw_row["split"],
            }
        if set(raw_source_by_index) - (source_shard_indices | expected_indices):
            raise ValueError("conversion source raw identities drift from shard")
        source_shard_indices.update(expected_indices)
    outputs = manifest.get("outputs")
    expected_output_files = {
        "sft1_train_all.jsonl",
        "sft1_train_success.jsonl",
        "sft1_heldout_all.jsonl",
        "sft2_train.jsonl",
        "sft2_heldout.jsonl",
        "rejections.jsonl",
    }
    if not isinstance(outputs, dict) or set(outputs) != expected_output_files:
        raise ValueError("conversion manifest output set mismatch")
    counts: dict[str, int] = {}
    rows_by_file: dict[str, list[dict[str, Any]]] = {}
    for filename, evidence in outputs.items():
        if Path(filename).name != filename or not isinstance(evidence, dict):
            raise ValueError("invalid conversion output evidence")
        path = output_dir / filename
        if not path.is_file():
            raise ValueError(f"conversion output is missing: {filename}")
        if path.stat().st_size != int(evidence.get("size_bytes", -1)):
            raise ValueError(f"conversion output size mismatch: {filename}")
        if _file_sha256(path) != evidence.get("sha256"):
            raise ValueError(f"conversion output hash mismatch: {filename}")
        rows = _read_jsonl(path)
        rows_by_file[filename] = rows
        counts[filename] = len(rows)
        if len(rows) != int(evidence.get("count", -1)):
            raise ValueError(f"conversion output count mismatch: {filename}")
        if filename == "rejections.jsonl":
            required_rejection_fields = {
                "format",
                "id",
                "source_index",
                "source_key",
                "eval_set",
                "seed",
                "split",
                "source_record_sha256",
                "error_type",
                "reason",
            }
            for row in rows:
                if set(row) != required_rejection_fields or row.get(
                    "format"
                ) != REJECTION_FORMAT:
                    raise ValueError("conversion rejection envelope format mismatch")
                if not isinstance(row.get("source_index"), int) or not isinstance(
                    row.get("seed"), int
                ):
                    raise TypeError("conversion rejection identity type mismatch")
                if not re.fullmatch(
                    r"[0-9a-f]{64}", str(row.get("source_record_sha256", ""))
                ):
                    raise ValueError("conversion rejection source hash mismatch")
                if any(
                    not isinstance(row.get(key), str) or not row[key]
                    for key in (
                        "id",
                        "source_key",
                        "eval_set",
                        "split",
                        "error_type",
                        "reason",
                    )
                ):
                    raise ValueError("conversion rejection text field mismatch")
        if filename != "rejections.jsonl" and evidence.get("stats") != (
            _dataset_stats(rows)
        ):
            raise ValueError(f"conversion output statistics mismatch: {filename}")
    valid = counts.get("sft1_train_all.jsonl", 0) + counts.get(
        "sft1_heldout_all.jsonl", 0
    )
    if valid != int(manifest.get("result", {}).get("valid", -1)):
        raise ValueError("conversion valid output count mismatch")
    if counts.get("rejections.jsonl", 0) != int(
        manifest.get("result", {}).get("excluded", -1)
    ):
        raise ValueError("conversion rejection count mismatch")
    if counts.get("sft2_train.jsonl", 0) + counts.get(
        "sft2_heldout.jsonl", 0
    ) != valid:
        raise ValueError("conversion SFT1/SFT2 valid counts disagree")

    from nimloth.rollout import RolloutTrajectory, validate_rollout_trajectory
    from nimloth.rollout.transitions import expand_record_transitions

    sft1_rows = [
        *rows_by_file.get("sft1_train_all.jsonl", []),
        *rows_by_file.get("sft1_heldout_all.jsonl", []),
    ]
    sft2_rows = [
        *rows_by_file.get("sft2_train.jsonl", []),
        *rows_by_file.get("sft2_heldout.jsonl", []),
    ]
    for row in sft1_rows:
        _validate_sft1_record(row, latent_token_count=16)
        provenance = row.get("conversion_provenance")
        if not isinstance(provenance, dict) or provenance.get(
            "converted_sha256"
        ) != _canonical_sha256(
            {
                key: value
                for key, value in row.items()
                if key != "conversion_provenance"
            }
        ):
            raise ValueError("published SFT1 converted hash mismatch")
    for row in sft2_rows:
        trajectory = RolloutTrajectory.from_record(row)
        validate_rollout_trajectory(trajectory)
        if len(expand_record_transitions(row)) != len(row.get("action_indices", [])):
            raise ValueError("published SFT2 transition count mismatch")
    sft1_by_id = {str(row.get("id")): row for row in sft1_rows}
    sft2_by_id = {str(row.get("id")): row for row in sft2_rows}
    if len(sft1_by_id) != len(sft1_rows) or len(sft2_by_id) != len(sft2_rows):
        raise ValueError("published conversion contains duplicate IDs")
    if set(sft1_by_id) != set(sft2_by_id):
        raise ValueError("published SFT1/SFT2 IDs disagree")
    for record_id, left in sft1_by_id.items():
        right = sft2_by_id[record_id]
        if left.get("source_identity") != right.get("source_identity") or left.get(
            "source_audit"
        ) != right.get("source_audit"):
            raise ValueError("published SFT1/SFT2 source linkage mismatch")
        source_audit = right.get("source_audit")
        source_identity = right.get("source_identity")
        if not isinstance(source_identity, dict) or not isinstance(
            source_audit, dict
        ):
            raise TypeError("published source identity/audit must be mappings")
        source_index = int(source_identity["source_index"])
        raw_source = raw_source_by_index.get(source_index)
        if raw_source is None or source_audit != raw_source["source_audit"]:
            raise ValueError("published source audit does not match raw row")
        if record_id != str(raw_source["id"]):
            raise ValueError("published record ID does not match raw row")
        for key in ("source_key", "eval_set", "seed", "split"):
            if source_identity.get(key) != raw_source[key]:
                raise ValueError("published source identity does not match raw row")
        artifacts = source_audit.get("image_artifacts") if isinstance(
            source_audit, dict
        ) else None
        image_paths = right.get("image_paths")
        if not isinstance(artifacts, list) or not isinstance(image_paths, list) or (
            len(artifacts) != len(image_paths)
        ):
            raise ValueError("published source image audit alignment mismatch")
        for image_path_text, artifact in zip(image_paths, artifacts, strict=True):
            if not isinstance(artifact, dict):
                raise TypeError("published source image artifact must be a mapping")
            image_path = Path(str(image_path_text)).resolve()
            if not image_path.is_file() or image_path.stat().st_size != int(
                artifact.get("size_bytes", -1)
            ):
                raise ValueError("published source image size/path mismatch")
            if _file_sha256(image_path) != artifact.get("sha256"):
                raise ValueError("published source image hash mismatch")
            if not image_path.as_posix().endswith(str(artifact.get("path", ""))):
                raise ValueError("published source image relative path mismatch")
    expected_success = {
        str(row["id"]): row
        for row in rows_by_file.get("sft1_train_all.jsonl", [])
        if bool(row.get("success"))
    }
    actual_success = {
        str(row.get("id")): row
        for row in rows_by_file.get("sft1_train_success.jsonl", [])
    }
    if actual_success != expected_success:
        raise ValueError("published SFT1 success subset mismatch")
    train_seeds = {
        int(row["source_identity"]["seed"])
        for row in rows_by_file.get("sft2_train.jsonl", [])
    }
    heldout_seeds = {
        int(row["source_identity"]["seed"])
        for row in rows_by_file.get("sft2_heldout.jsonl", [])
    }
    if train_seeds & heldout_seeds:
        raise ValueError("published train/heldout seeds overlap")

    partition_contract = manifest.get("partition_manifest")
    if not isinstance(partition_contract, dict):
        raise TypeError("conversion manifest has no partition contract")
    partition_path = Path(str(partition_contract.get("path", ""))).resolve()
    if not partition_path.is_file() or _file_sha256(partition_path) != (
        partition_contract.get("sha256")
    ):
        raise ValueError("published conversion partition hash/path mismatch")
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    validate_partition_manifest(partition, require_published=True)
    expected_by_index = {
        int(row["source_index"]): row
        for row in partition.get("rows", [])
        if int(row.get("batch", -1)) == 1
    }
    if len(expected_by_index) != 2_000:
        raise ValueError("published partition does not contain exact batch1")
    observed: dict[int, dict[str, Any]] = {}
    for row in sft2_rows:
        identity = row.get("source_identity")
        if not isinstance(identity, dict):
            raise TypeError("published SFT2 row has no source identity")
        source_index = int(identity["source_index"])
        if source_index in observed:
            raise ValueError("published conversion duplicates a source index")
        observed[source_index] = identity
    for row in rows_by_file.get("rejections.jsonl", []):
        source_index = int(row["source_index"])
        if source_index in observed:
            raise ValueError("published conversion duplicates a source index")
        raw_source = raw_source_by_index.get(source_index)
        if raw_source is None or row.get("source_record_sha256") != raw_source.get(
            "raw_record_sha256"
        ):
            raise ValueError("published rejection does not match raw source row")
        if row.get("id") != raw_source["id"] or row.get("source_key") != raw_source[
            "source_key"
        ]:
            raise ValueError("published rejection identity does not match raw row")
        observed[source_index] = {
            "source_index": source_index,
            "eval_set": str(row["eval_set"]),
            "seed": int(row["seed"]),
            "split": str(row["split"]),
        }
    if set(observed) != set(expected_by_index):
        raise ValueError("published conversion does not cover exact batch1 identities")
    for source_index, expected in expected_by_index.items():
        actual = observed[source_index]
        for key in ("eval_set", "seed", "split"):
            if actual.get(key) != expected.get(
                "dataset_split" if key == "split" else key
            ):
                raise ValueError(
                    f"published source identity drift at {source_index}: {key}"
                )
    return manifest


def convert_complete_batch1(
    *,
    partition_manifest: Path,
    shard_dirs: Sequence[Path],
    output_dir: Path,
    latent_token_count: int = 16,
) -> dict[str, Any]:
    """Convert exact complete batch1 once and atomically publish all views."""

    from nimloth.rollout import RolloutTrajectory, validate_rollout_trajectory
    from nimloth.rollout.transitions import expand_record_transitions

    if latent_token_count != 16:
        raise ValueError("step60 conversion is fixed to K16")
    partition_manifest = partition_manifest.resolve()
    partition = json.loads(partition_manifest.read_text(encoding="utf-8"))
    validate_partition_manifest(partition, require_published=True)
    batch1_rows = [row for row in partition.get("rows", []) if int(row["batch"]) == 1]
    if len(batch1_rows) != 2_000:
        raise ValueError(f"partition manifest batch1 count drift: {len(batch1_rows)}")
    expected_by_index = {int(row["source_index"]): row for row in batch1_rows}
    if len(expected_by_index) != 2_000:
        raise ValueError("partition manifest batch1 has duplicate source indices")
    split_counts = Counter(str(row["dataset_split"]) for row in batch1_rows)
    if split_counts != {"train": 1_800, "heldout": 200}:
        raise ValueError(f"partition manifest batch1 split drift: {dict(split_counts)}")
    records, shard_evidence = _load_verified_records(
        shard_dirs,
        expected_by_index=expected_by_index,
    )

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"conversion output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(
        f".{output_dir.name}.tmp-{uuid.uuid4().hex[:12]}"
    )
    temporary.mkdir()
    valid_sft1: list[dict[str, Any]] = []
    valid_sft2: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    try:
        for source, shard_dir in records:
            try:
                converted = convert_source_record(
                    source,
                    latent_token_count=latent_token_count,
                    source_root=shard_dir,
                )
                sft1 = converted["sft1"]
                sft2 = converted["sft2"]
                _validate_sft1_record(
                    sft1,
                    latent_token_count=latent_token_count,
                )
                trajectory = RolloutTrajectory.from_record(sft2)
                validate_rollout_trajectory(trajectory)
                transitions = expand_record_transitions(sft2)
                if len(transitions) != len(sft2["action_indices"]):
                    raise ValueError("SFT2 terminal draft created an extra transition")
                valid_sft1.append(sft1)
                valid_sft2.append(sft2)
            except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
                rejections.append(
                    {
                        "format": REJECTION_FORMAT,
                        "id": source.get("id"),
                        "source_index": source.get("source_index"),
                        "source_key": source.get("source_key"),
                        "eval_set": source.get("eval_set"),
                        "seed": source.get("seed"),
                        "split": source.get("split"),
                        "source_record_sha256": source.get("raw_record_sha256"),
                        "error_type": type(error).__name__,
                        "reason": str(error),
                    }
                )

        if len(records) != len(valid_sft1) + len(rejections):
            raise RuntimeError("conversion input does not equal valid + excluded")
        if len(valid_sft1) != len(valid_sft2):
            raise RuntimeError("SFT1/SFT2 valid record counts diverged")
        valid_ids1 = [record["id"] for record in valid_sft1]
        valid_ids2 = [record["id"] for record in valid_sft2]
        if valid_ids1 != valid_ids2:
            raise RuntimeError("SFT1/SFT2 valid record identities diverged")

        sft1_train = [record for record in valid_sft1 if record["split"] == "train"]
        sft1_success = [record for record in sft1_train if record["success"]]
        sft1_heldout = [record for record in valid_sft1 if record["split"] == "heldout"]
        sft2_train = [record for record in valid_sft2 if record["split"] == "train"]
        sft2_heldout = [record for record in valid_sft2 if record["split"] == "heldout"]
        output_rows = {
            "sft1_train_all.jsonl": sft1_train,
            "sft1_train_success.jsonl": sft1_success,
            "sft1_heldout_all.jsonl": sft1_heldout,
            "sft2_train.jsonl": sft2_train,
            "sft2_heldout.jsonl": sft2_heldout,
            "rejections.jsonl": rejections,
        }
        output_evidence: dict[str, Any] = {}
        for filename, rows in output_rows.items():
            path = temporary / filename
            count = _write_jsonl(path, rows)
            output_evidence[filename] = {
                "count": count,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
                "stats": _dataset_stats(rows) if filename != "rejections.jsonl" else None,
            }

        train_seeds = {
            int(record["source_identity"]["seed"])
            for record in valid_sft2
            if record["split"] == "train"
        }
        heldout_seeds = {
            int(record["source_identity"]["seed"])
            for record in valid_sft2
            if record["split"] == "heldout"
        }
        if train_seeds & heldout_seeds:
            raise RuntimeError("converted train/heldout bare seeds overlap")
        manifest = {
            "format": CONVERSION_FORMAT,
            "partition_manifest": {
                "path": str(partition_manifest),
                "sha256": _file_sha256(partition_manifest),
            },
            "source_shards": shard_evidence,
            "reward_provenance_counts": dict(
                Counter(
                    str(record["reward_provenance"])
                    for record, _shard_dir in records
                )
            ),
            "latent_token_count": latent_token_count,
            "input": {
                "records": len(records),
                "train": 1_800,
                "heldout": 200,
            },
            "result": {
                "valid": len(valid_sft1),
                "excluded": len(rejections),
                "input_equals_valid_plus_excluded": (
                    len(records) == len(valid_sft1) + len(rejections)
                ),
                "train_heldout_bare_seed_overlap": 0,
            },
            "outputs": output_evidence,
            "limitations": [
                "internal heldout is unseen seed within batch1, not unseen environment distribution",
                "source test.parquet is excluded because all 128 eval_set/seed keys overlap train",
                "terminal generation is state-only and has no LLM-backbone supervision",
                "source behavior action/token log-probabilities are unavailable and not invented",
            ],
        }
        manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
        _write_json(temporary / "conversion_manifest.json", manifest)
        atomic_publish_directory(temporary, output_dir)
        return validate_conversion_output(output_dir)
    except Exception:
        # Conversion has no partial-resume semantics. Preserve the unique failed
        # temp directory and add an explicit failure marker for end recording.
        failure = temporary / "FAILED"
        if temporary.exists() and not failure.exists():
            failure.write_text("conversion failed\n", encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latent-token-count", type=int, default=16)
    args = parser.parse_args()
    manifest = convert_complete_batch1(
        partition_manifest=args.partition_manifest,
        shard_dirs=args.shard_dir,
        output_dir=args.output_dir,
        latent_token_count=args.latent_token_count,
    )
    print(json.dumps(manifest["result"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
