"""Partition trajectory records by pinned original VAGEN (eval_set, seed) keys."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from nimloth.rollout.record_format import require_trajectory_record


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _pair(value: Any) -> tuple[str, int]:
    if not isinstance(value, dict):
        raise ValueError("pair identity must be a mapping")
    category, seed = value.get("eval_set"), value.get("seed")
    if not isinstance(category, str) or not category or type(seed) is not int:
        raise ValueError("pair requires explicit eval_set string and integer seed")
    return category, seed


def record_pair(record: dict[str, Any]) -> tuple[str, int]:
    pair = _pair(record.get("source_identity"))
    if "eval_set" in record or "seed" in record:
        if _pair(record) != pair:
            raise ValueError("top-level pair disagrees with source_identity")
    return pair


def _write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.rename(path)


def split_by_eval_keys(source: Path, eval_keys_manifest: Path, output_root: Path) -> dict[str, Any]:
    """Change only split and add its audit; source rows/checkpoints are never mutated."""
    source_bytes = source.read_bytes()
    keys_bytes = eval_keys_manifest.read_bytes()
    keys_manifest = json.loads(keys_bytes)
    if keys_manifest.get("format") != "vagen_eval_keys_v1":
        raise ValueError("unsupported eval key manifest format")
    test_hash = _digest(keys_manifest.get("source_sha256"), "source_sha256")
    input_hash = _digest(keys_manifest.get("input_sha256"), "input_sha256")
    if input_hash != _sha256(source_bytes):
        raise ValueError("source JSONL hash does not match eval key manifest input_sha256")
    raw_keys = keys_manifest.get("keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ValueError("eval keys must be a nonempty list")
    pairs = [_pair(item) for item in raw_keys]
    if type(keys_manifest.get("count")) is not int or keys_manifest["count"] != len(pairs):
        raise ValueError("eval key count does not match keys")
    eval_keys = set(pairs)
    if len(eval_keys) != len(pairs):
        raise ValueError("duplicate eval keys")
    groups: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    split_keys: dict[str, set[tuple[str, int]]] = {"train": set(), "eval": set()}
    ids = set()
    for line_number, line in enumerate(source_bytes.splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        require_trajectory_record(record)
        record_id = record["id"]
        if not isinstance(record_id, str) or not record_id or record_id in ids:
            raise ValueError(f"missing or duplicate trajectory ID at line {line_number}")
        ids.add(record_id)
        if "split_provenance" in record:
            raise ValueError("record has already been split with provenance")
        pair = record_pair(record)
        split = "eval" if pair in eval_keys else "train"
        result = copy.deepcopy(record)
        result["split"] = split
        result["split_provenance"] = {
            "format": "original_vagen_eval_pair_split_v1",
            "original_split": record["split"],
            "source_jsonl_sha256": input_hash,
            "source_row_sha256": _sha256(line),
            "eval_keys_manifest_sha256": _sha256(keys_bytes),
            "original_vagen_eval_source_sha256": test_hash,
            "eval_set": pair[0], "seed": pair[1],
        }
        # Verify this partition leaves reward, returns, CoT, images and labels untouched.
        restored = copy.deepcopy(result)
        restored.pop("split_provenance")
        restored["split"] = record["split"]
        if restored != record:
            raise ValueError("partition unexpectedly changed source content")
        require_trajectory_record(result)
        groups[split].append(result)
        split_keys[split].add(pair)
    if split_keys["train"] & split_keys["eval"]:
        raise ValueError("train/eval identity overlap")
    if not ids:
        raise ValueError("source contains no trajectory records")
    # Fail before creating output if either input changed during validation.
    if source.read_bytes() != source_bytes or eval_keys_manifest.read_bytes() != keys_bytes:
        raise ValueError("partition input mutated during validation")
    output_root.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for split, rows in groups.items():
        payload = b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode() for row in rows)
        _write(output_root / f"{split}.jsonl", payload)
        outputs[split] = {
            "file": f"{split}.jsonl", "sha256": _sha256(payload),
            "records": len(rows), "unique_pairs": len(split_keys[split]),
            "transitions": sum(len(row["action_indices"]) for row in rows),
        }
    manifest = {
        "format": "original_vagen_eval_pair_split_v1", "source_path": str(source.resolve()),
        "source_sha256": input_hash, "source_records": len(ids),
        "eval_keys_manifest_sha256": _sha256(keys_bytes),
        "original_vagen_eval_source_sha256": test_hash,
        "eval_key_count": len(eval_keys), "outputs": outputs,
        "missing_eval_keys": [{"eval_set": category, "seed": seed} for category, seed in sorted(eval_keys - split_keys["eval"])],
        "train_eval_pair_overlap": 0,
        "identity_semantics": "exact_original_eval_set_and_seed; task_modulo_overlap_not_excluded",
    }
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    _write(output_root / "manifest.json", manifest_bytes)
    _write(output_root / "COMMITTED", (_sha256(manifest_bytes) + "\n").encode())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--eval-keys-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(split_by_eval_keys(args.source, args.eval_keys_manifest, args.output_root)))


if __name__ == "__main__":
    main()
