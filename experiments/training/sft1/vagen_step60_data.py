#!/usr/bin/env python3
"""Prepare and validate source-faithful VAGEN step60 rollout data.

The module keeps source-row partitioning deterministic and non-overwriting and
owns the hash-bound complete-shard consumption gate. A later work item adds the
SFT1/SFT2 conversion view without weakening these source contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SOURCE_TRAIN_SHA256 = (
    "3c8161bd45adc4cde5d67157cf4db225753ed3925cb9a52e3a57d1dd11dbe9d6"
)
UNAVAILABLE_SOURCE_COMMIT = "fee3ffac036a599b0ae979a6dd1ce2b21f7dec49"
SOURCE_ROW_COUNT = 20_000
ROWS_PER_CATEGORY = 10_000
ROWS_PER_CATEGORY_PER_BATCH = 1_000
BATCH_COUNT = 10
CATEGORIES = ("base", "common_sense")
SOURCE_ENV_CONFIG = {
    "render_mode": "vision",
    "prompt_format": "grounding_worldmodeling",
    "use_state_reward": False,
    "max_actions_per_step": 1,
    "format_reward": 0.02,
    "invalid_action_penalty": -0.2,
    "success_threshold": 1.5,
}
PARTITION_FORMAT = "vagen_step60_partition_v1"
CONVERSION_FORMAT = "vagen_step60_dual_view_conversion_v3"
REJECTION_FORMAT = "vagen_step60_rejections_v3"
SOURCE_AUDIT_CONTRACT_VERSION = "vagen_step60_reconstruction_audit_v3"
RECONSTRUCTION_FORMATS = {
    "runtime_contract": "vagen_step60_reconstruction_runtime_contract_v3",
    "raw_row": "vagen_step60_source_trajectory_v3",
    "shard_manifest": "vagen_step60_complete_shard_v3",
    "complete_marker": "vagen_step60_complete_shard_v3",
    "conversion_manifest": CONVERSION_FORMAT,
    "rejection_envelope": REJECTION_FORMAT,
    "source_audit": SOURCE_AUDIT_CONTRACT_VERSION,
    "partition_manifest": PARTITION_FORMAT,
    "hf_merge_manifest": "nimloth_vagen_step60_hf_export_v1",
}


def validate_reconstruction_format(surface: str, value: str) -> None:
    expected = RECONSTRUCTION_FORMATS.get(surface)
    if expected is None or value != expected:
        raise ValueError(
            f"unsupported reconstruction format for {surface}: {value!r}"
        )
SOURCE_ACTION_NAMES = (
    "moveahead",
    "moveback",
    "moveright",
    "moveleft",
    "rotateright",
    "rotateleft",
    "lookup",
    "lookdown",
)
_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SOURCE_RESPONSE_RE = re.compile(
    r"<think><observation>(?P<observation>.*?)</observation>"
    r"<reasoning>(?P<reasoning>.*?)</reasoning>"
    r"<prediction>(?P<prediction>.*?)</prediction></think>"
    r"<answer>(?P<action>[a-z_]+)</answer>",
    re.DOTALL,
)
_SOURCE_ENVELOPE_RE = re.compile(
    r"<think><observation>(.*?)</observation>"
    r"<reasoning>(.*?)</reasoning>"
    r"<prediction>(.*?)</prediction></think>"
    r"<answer>(.*?)</answer>",
    re.DOTALL,
)
_SOURCE_ANSWER_RE = re.compile(r"<answer>\s*([^<]+?)\s*</answer>", re.DOTALL)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


PUBLISHING_SENTINEL = ".NIMLOTH_PUBLISHING.json"
READINESS_MARKERS = frozenset(
    {"partition_manifest.json", "COMPLETE", "conversion_manifest.json"}
)
PUBLICATION_FAILURE_MARKERS = frozenset(
    {"FAILED", "FAILED.json", "FAILED_PUBLISH.json", "FAILED_VALIDATION.json"}
)
READINESS_STAGING_NAME = ".NIMLOTH_READINESS.tmp"


def lexical_absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_published_directory(path: Path, *, readiness_marker: str) -> Path:
    """Require one fully published reserved directory and its final marker."""

    if readiness_marker not in READINESS_MARKERS:
        raise ValueError(f"unsupported readiness marker: {readiness_marker!r}")
    path = lexical_absolute_path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"published artifact is not a real directory: {path}")
    if (path / PUBLISHING_SENTINEL).exists() or (
        path / PUBLISHING_SENTINEL
    ).is_symlink():
        raise ValueError(f"published artifact still has publishing sentinel: {path}")
    if (path / READINESS_STAGING_NAME).exists() or (
        path / READINESS_STAGING_NAME
    ).is_symlink():
        raise ValueError(f"published artifact has staged readiness: {path}")
    if any(
        (path / name).exists() or (path / name).is_symlink()
        for name in PUBLICATION_FAILURE_MARKERS
    ):
        raise ValueError(f"published artifact has a failure marker: {path}")
    for marker_name in READINESS_MARKERS:
        occurrences = [entry for entry in path.rglob(marker_name)]
        expected = path / readiness_marker
        if marker_name == readiness_marker:
            if occurrences != [expected]:
                raise ValueError(f"readiness marker ownership mismatch: {marker_name}")
            if expected.is_symlink() or not expected.is_file():
                raise ValueError(f"readiness marker is not a regular file: {expected}")
        elif occurrences:
            raise ValueError(f"unexpected readiness marker: {marker_name}")
    return path


def publish_reserved_directory(
    source: Path,
    target: Path,
    *,
    readiness_marker: str,
) -> None:
    """Reserve the final path atomically and publish its readiness marker last."""

    if readiness_marker not in READINESS_MARKERS:
        raise ValueError(f"unsupported readiness marker: {readiness_marker!r}")
    source = lexical_absolute_path(source)
    target = lexical_absolute_path(target)
    if source == target:
        raise ValueError("staging and final publication paths must differ")
    if source.is_symlink() or not source.is_dir():
        raise ValueError("publication staging path must be a real directory")
    if source.parent.resolve() != target.parent.resolve():
        raise ValueError("publication staging and target must be direct siblings")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"publication target already exists: {target}")
    if any(
        (source / name).exists() or (source / name).is_symlink()
        for name in (PUBLISHING_SENTINEL, READINESS_STAGING_NAME)
    ):
        raise ValueError("publication staging contains a reserved internal path")

    expected_marker = source / readiness_marker
    for marker_name in READINESS_MARKERS:
        occurrences = [entry for entry in source.rglob(marker_name)]
        if marker_name == readiness_marker:
            if occurrences != [expected_marker]:
                raise ValueError(f"readiness marker ownership mismatch: {marker_name}")
            if expected_marker.is_symlink() or not expected_marker.is_file():
                raise ValueError("readiness marker must be a root regular file")
        elif occurrences:
            raise ValueError(f"unexpected readiness marker: {marker_name}")

    os.mkdir(target)
    sentinel = target / PUBLISHING_SENTINEL
    sentinel_payload = {
        "format": "nimloth_reserved_directory_publication_v1",
        "readiness_marker": readiness_marker,
        "staging_name": source.name,
    }
    with sentinel.open("x", encoding="utf-8") as handle:
        json.dump(sentinel_payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(target)
    _fsync_directory(target.parent)

    payload_entries = sorted(
        (entry for entry in source.iterdir() if entry.name != readiness_marker),
        key=lambda entry: entry.name,
    )
    for entry in payload_entries:
        os.rename(entry, target / entry.name)
    _fsync_directory(target)
    staged_readiness = target / READINESS_STAGING_NAME
    os.rename(expected_marker, staged_readiness)
    source.rmdir()
    _fsync_directory(target)
    _fsync_directory(target.parent)
    sentinel.unlink()
    _fsync_directory(target)
    _fsync_directory(target.parent)
    # This rename is the commit point and intentionally the final operation.
    os.rename(staged_readiness, target / readiness_marker)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_identity(
    row: Mapping[str, Any],
    *,
    source_index: int,
    expected_split: str | None = None,
) -> tuple[str, int]:
    """Read the verified parquet schema without guessing field aliases."""

    if row.get("data_source") != "navigation":
        raise ValueError(
            f"source row {source_index} has data_source {row.get('data_source')!r}"
        )
    extra_info = row.get("extra_info")
    if not isinstance(extra_info, Mapping):
        raise TypeError(f"source row {source_index} has no mapping extra_info")
    if extra_info.get("env_name") != "navigation":
        raise ValueError(
            f"source row {source_index} has env_name {extra_info.get('env_name')!r}"
        )
    split = extra_info.get("split")
    if expected_split is not None and split != expected_split:
        raise ValueError(
            f"source row {source_index} has split {split!r}, "
            f"expected {expected_split!r}"
        )
    seed = extra_info.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"source row {source_index} has non-integer seed {seed!r}")
    env_config = extra_info.get("env_config")
    if not isinstance(env_config, Mapping):
        raise TypeError(f"source row {source_index} has no mapping env_config")
    eval_set = env_config.get("eval_set")
    if eval_set not in CATEGORIES:
        raise ValueError(
            f"source row {source_index} has unsupported eval_set {eval_set!r}"
        )
    for field, expected in SOURCE_ENV_CONFIG.items():
        actual = env_config.get(field)
        if actual != expected:
            raise ValueError(
                f"source row {source_index} env_config.{field} drift: "
                f"{actual!r} != {expected!r}"
            )
    return str(eval_set), seed


def parse_source_response(response: str) -> dict[str, Any]:
    """Parse only a complete real source response; never repair malformed CoT."""

    match = _SOURCE_RESPONSE_RE.fullmatch(response)
    if match is None:
        return {
            "format_valid": False,
            "thought": None,
            "action_name": None,
            "action_index": None,
        }
    observation = match.group("observation")
    reasoning = match.group("reasoning")
    prediction = match.group("prediction")
    thought = (
        f"<observation>{observation}</observation>"
        f"<reasoning>{reasoning}</reasoning>"
        f"<prediction>{prediction}</prediction>"
    )
    action = match.group("action")
    if not all(value.strip() for value in (observation, reasoning, prediction)) or (
        action not in SOURCE_ACTION_NAMES
    ):
        return {
            "format_valid": False,
            "thought": thought,
            "action_name": action,
            "action_index": None,
        }
    return {
        "format_valid": True,
        "thought": thought,
        "action_name": action,
        "action_index": SOURCE_ACTION_NAMES.index(action),
    }


def build_partition_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Build the reviewed ten-batch manifest from the pinned source order."""

    if not source_path:
        raise ValueError("source_path must be non-empty")
    normalized_sha256 = source_sha256.lower()
    if not _HEX_SHA256_RE.fullmatch(normalized_sha256):
        raise ValueError("source_sha256 must be a lowercase SHA256 digest")
    if normalized_sha256 != SOURCE_TRAIN_SHA256:
        raise ValueError(
            "source parquet SHA256 drift: "
            f"{normalized_sha256} != {SOURCE_TRAIN_SHA256}"
        )
    if len(rows) != SOURCE_ROW_COUNT:
        raise ValueError(
            f"source row count drift: {len(rows)} != {SOURCE_ROW_COUNT}"
        )

    category_ordinals = Counter[str]()
    seeds_by_category: dict[str, list[int]] = {
        category: [] for category in CATEGORIES
    }
    seen_keys: set[tuple[str, int]] = set()
    entries: list[dict[str, Any]] = []

    for source_index, row in enumerate(rows):
        eval_set, seed = _row_identity(
            row,
            source_index=source_index,
            expected_split="train",
        )
        expected_eval_set = CATEGORIES[source_index // ROWS_PER_CATEGORY]
        if eval_set != expected_eval_set:
            raise ValueError(
                f"source row order drift at {source_index}: "
                f"{eval_set!r} != {expected_eval_set!r}"
            )
        key = (eval_set, seed)
        if key in seen_keys:
            raise ValueError(f"duplicate source key at row {source_index}: {key!r}")
        seen_keys.add(key)

        category_ordinal = category_ordinals[eval_set]
        category_ordinals[eval_set] += 1
        seeds_by_category[eval_set].append(seed)
        batch = category_ordinal // ROWS_PER_CATEGORY_PER_BATCH + 1
        # Only batch1 is approved for conversion/launch in this task.
        dataset_split = (
            "heldout" if category_ordinal % 10 == 9 else "train"
        ) if batch == 1 else "unassigned"
        entries.append(
            {
                "source_index": source_index,
                "eval_set": eval_set,
                "seed": seed,
                "source_key": f"{eval_set}:{seed}",
                "category_ordinal": category_ordinal,
                "batch": batch,
                "dataset_split": dataset_split,
            }
        )

    expected_category_counts = Counter(
        {category: ROWS_PER_CATEGORY for category in CATEGORIES}
    )
    if category_ordinals != expected_category_counts:
        raise ValueError(
            f"source category counts drift: {dict(category_ordinals)!r}"
        )
    if seeds_by_category[CATEGORIES[0]] != seeds_by_category[CATEGORIES[1]]:
        raise ValueError("source categories no longer share the same ordered seeds")

    batches: list[dict[str, Any]] = []
    for batch in range(1, BATCH_COUNT + 1):
        selected = [entry for entry in entries if entry["batch"] == batch]
        category_counts = Counter(entry["eval_set"] for entry in selected)
        expected_batch_counts = Counter(
            {
                category: ROWS_PER_CATEGORY_PER_BATCH
                for category in CATEGORIES
            }
        )
        if len(selected) != 2 * ROWS_PER_CATEGORY_PER_BATCH:
            raise ValueError(f"batch {batch} has {len(selected)} rows")
        if category_counts != expected_batch_counts:
            raise ValueError(
                f"batch {batch} category imbalance: {dict(category_counts)!r}"
            )
        batches.append(
            {
                "batch": batch,
                "count": len(selected),
                "category_counts": dict(category_counts),
                "source_indices": [entry["source_index"] for entry in selected],
                "row_manifest_sha256": _canonical_sha256(selected),
            }
        )

    batch1 = [entry for entry in entries if entry["batch"] == 1]
    train = [entry for entry in batch1 if entry["dataset_split"] == "train"]
    heldout = [
        entry for entry in batch1 if entry["dataset_split"] == "heldout"
    ]
    train_indices = {entry["source_index"] for entry in train}
    heldout_indices = {entry["source_index"] for entry in heldout}
    train_keys = {(entry["eval_set"], entry["seed"]) for entry in train}
    heldout_keys = {
        (entry["eval_set"], entry["seed"]) for entry in heldout
    }
    train_seeds = {entry["seed"] for entry in train}
    heldout_seeds = {entry["seed"] for entry in heldout}
    if len(train) != 1_800 or len(heldout) != 200:
        raise ValueError(
            f"batch1 split drift: train={len(train)}, heldout={len(heldout)}"
        )
    overlaps = {
        "source_index": len(train_indices & heldout_indices),
        "eval_set_seed": len(train_keys & heldout_keys),
        "seed": len(train_seeds & heldout_seeds),
    }
    if any(overlaps.values()):
        raise ValueError(f"batch1 train/heldout overlap: {overlaps!r}")

    manifest = {
        "format": PARTITION_FORMAT,
        "source": {
            "path": source_path,
            "sha256": normalized_sha256,
            "row_count": len(rows),
        },
        "partition_rule": {
            "batch": "category_ordinal // 1000 + 1",
            "batch1_heldout": "category_ordinal % 10 == 9",
            "future_batch_split": "unassigned",
        },
        "rows": entries,
        "batches": batches,
        "checks": {
            "batch_count": len(batches),
            "unique_source_indices": len(
                {entry["source_index"] for entry in entries}
            ),
            "source_index_union_is_exact": (
                {entry["source_index"] for entry in entries}
                == set(range(SOURCE_ROW_COUNT))
            ),
            "batch1_train_count": len(train),
            "batch1_heldout_count": len(heldout),
            "batch1_overlap": overlaps,
            "categories_share_ordered_seeds": True,
        },
    }
    validate_partition_manifest(manifest, require_published=False)
    return manifest


def partition_manifest_payload_sha256(manifest: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_payload_sha256"
    }
    return _canonical_sha256(payload)


def validate_partition_manifest(
    manifest: Mapping[str, Any],
    *,
    require_published: bool,
) -> dict[str, Any]:
    """Recompute every partition/split invariant before a consumer trusts it."""

    if manifest.get("format") != PARTITION_FORMAT:
        raise ValueError(f"unsupported partition format: {manifest.get('format')!r}")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise TypeError("partition manifest source must be a mapping")
    if source.get("sha256") != SOURCE_TRAIN_SHA256:
        raise ValueError("partition manifest source SHA256 drift")
    if int(source.get("row_count", -1)) != SOURCE_ROW_COUNT:
        raise ValueError("partition manifest source row count drift")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != SOURCE_ROW_COUNT:
        raise ValueError("partition manifest rows do not cover 20,000 source rows")

    seeds_by_category: dict[str, list[int]] = {category: [] for category in CATEGORIES}
    seen_keys: set[tuple[str, int]] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"partition row {position} is not a mapping")
        source_index = int(row.get("source_index", -1))
        if source_index != position:
            raise ValueError(
                f"partition source index order drift: {source_index} != {position}"
            )
        expected_eval_set = CATEGORIES[position // ROWS_PER_CATEGORY]
        eval_set = str(row.get("eval_set"))
        if eval_set != expected_eval_set:
            raise ValueError(f"partition row {position} eval_set drift")
        ordinal = position % ROWS_PER_CATEGORY
        if int(row.get("category_ordinal", -1)) != ordinal:
            raise ValueError(f"partition row {position} category ordinal drift")
        expected_batch = ordinal // ROWS_PER_CATEGORY_PER_BATCH + 1
        if int(row.get("batch", -1)) != expected_batch:
            raise ValueError(f"partition row {position} batch drift")
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"partition row {position} seed must be int")
        key = (eval_set, seed)
        if key in seen_keys:
            raise ValueError(f"partition duplicate source key: {key!r}")
        seen_keys.add(key)
        seeds_by_category[eval_set].append(seed)
        if row.get("source_key") != f"{eval_set}:{seed}":
            raise ValueError(f"partition row {position} source key drift")
        expected_split = (
            "heldout" if ordinal % 10 == 9 else "train"
        ) if expected_batch == 1 else "unassigned"
        if row.get("dataset_split") != expected_split:
            raise ValueError(f"partition row {position} dataset split drift")
    if seeds_by_category[CATEGORIES[0]] != seeds_by_category[CATEGORIES[1]]:
        raise ValueError("partition categories do not share ordered seeds")

    batches = manifest.get("batches")
    if not isinstance(batches, list) or len(batches) != BATCH_COUNT:
        raise ValueError("partition manifest must contain ten batch summaries")
    summaries = {int(batch.get("batch", -1)): batch for batch in batches}
    if set(summaries) != set(range(1, BATCH_COUNT + 1)):
        raise ValueError("partition batch summary identities drift")
    for batch_number in range(1, BATCH_COUNT + 1):
        summary = summaries[batch_number]
        selected = [row for row in rows if int(row["batch"]) == batch_number]
        expected_indices = [int(row["source_index"]) for row in selected]
        if int(summary.get("count", -1)) != 2_000:
            raise ValueError(f"partition batch {batch_number} count drift")
        if summary.get("category_counts") != {
            "base": 1_000,
            "common_sense": 1_000,
        }:
            raise ValueError(f"partition batch {batch_number} category drift")
        if summary.get("source_indices") != expected_indices:
            raise ValueError(f"partition batch {batch_number} source indices drift")
        if summary.get("row_manifest_sha256") != _canonical_sha256(selected):
            raise ValueError(f"partition batch {batch_number} row hash mismatch")
        if require_published:
            parquet_name = summary.get("parquet")
            parquet_sha256 = summary.get("parquet_sha256")
            if not isinstance(parquet_name, str) or not parquet_name:
                raise ValueError(f"partition batch {batch_number} has no parquet path")
            if not isinstance(parquet_sha256, str) or not _HEX_SHA256_RE.fullmatch(
                parquet_sha256
            ):
                raise ValueError(f"partition batch {batch_number} has no parquet hash")
            if int(summary.get("parquet_size_bytes", -1)) <= 0:
                raise ValueError(f"partition batch {batch_number} has no parquet size")

    batch1 = [row for row in rows if int(row["batch"]) == 1]
    train = [row for row in batch1 if row["dataset_split"] == "train"]
    heldout = [row for row in batch1 if row["dataset_split"] == "heldout"]
    train_seeds = {int(row["seed"]) for row in train}
    heldout_seeds = {int(row["seed"]) for row in heldout}
    recomputed_checks = {
        "batch_count": BATCH_COUNT,
        "unique_source_indices": SOURCE_ROW_COUNT,
        "source_index_union_is_exact": True,
        "batch1_train_count": len(train),
        "batch1_heldout_count": len(heldout),
        "batch1_overlap": {
            "source_index": 0,
            "eval_set_seed": 0,
            "seed": len(train_seeds & heldout_seeds),
        },
        "categories_share_ordered_seeds": True,
    }
    if manifest.get("checks") != recomputed_checks:
        raise ValueError("partition manifest checks do not match recomputed evidence")
    if require_published:
        payload_hash = manifest.get("manifest_payload_sha256")
        if payload_hash != partition_manifest_payload_sha256(manifest):
            raise ValueError("partition manifest payload hash mismatch")
    return dict(manifest)


def load_published_partition_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load one marker-complete partition and rehash every sibling parquet."""

    manifest_path = lexical_absolute_path(manifest_path)
    published = validate_published_directory(
        manifest_path.parent,
        readiness_marker="partition_manifest.json",
    )
    if manifest_path != published / "partition_manifest.json":
        raise ValueError("partition manifest must be the root readiness marker")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validated = validate_partition_manifest(manifest, require_published=True)
    for batch in validated["batches"]:
        relative = Path(str(batch["parquet"]))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("partition parquet must be a root sibling file")
        parquet = published / relative
        if parquet.is_symlink() or not parquet.is_file():
            raise ValueError(f"partition parquet is not a regular file: {relative}")
        if parquet.stat().st_size != int(batch["parquet_size_bytes"]):
            raise ValueError(f"partition parquet size mismatch: {relative}")
        if _file_sha256(parquet) != batch["parquet_sha256"]:
            raise ValueError(f"partition parquet hash mismatch: {relative}")
    return validated


def _action_envelope(action_text: str, *, latent_token_count: int) -> str:
    from nimloth.latent import LatentActionTokens, latent_state_tokens

    tokens = LatentActionTokens()
    latent_block = "".join(latent_state_tokens(latent_token_count, tokens))
    normalized = action_text.strip().lower()
    action_token = (
        tokens.action_tokens[SOURCE_ACTION_NAMES.index(normalized)]
        if normalized in SOURCE_ACTION_NAMES
        else "<|action_(idx)|>"
    )
    return f"{latent_block}{tokens.action_start}{action_token}{tokens.action_end}"


def convert_source_prompt(text: str, *, latent_token_count: int) -> str:
    """Rewrite only source action-format instructions into the K16 contract."""

    from nimloth.latent import LatentActionTokens

    if not isinstance(text, str) or not text.strip():
        raise TypeError("source prompt text must be non-empty")
    tokens = LatentActionTokens()
    has_source_format = any(
        marker in text
        for marker in ("<answer>", "</answer>", "inside <answer>", "after </answer>")
    )
    if not has_source_format:
        return text
    converted = _SOURCE_ANSWER_RE.sub(
        lambda match: _action_envelope(
            match.group(1),
            latent_token_count=latent_token_count,
        ),
        text,
    )
    converted = converted.replace(
        "inside <answer>",
        f"between {tokens.action_start} and {tokens.action_end}",
    ).replace(
        "after </answer>",
        f"after {tokens.action_end}",
    )
    converted = converted.replace("<answer>", tokens.action_start).replace(
        "</answer>", tokens.action_end
    )
    if "<answer" in converted or "</answer" in converted:
        raise ValueError("converted prompt still contains the source answer envelope")
    if tokens.action_start not in converted:
        raise ValueError("source format text has no convertible answer envelope")
    mapping = ", ".join(
        f"{index}={name}" for index, name in enumerate(SOURCE_ACTION_NAMES)
    )
    if "Nimloth action indices:" not in converted:
        converted += f"\nNimloth action indices: {mapping}."
    return converted


def convert_source_assistant_response(
    response: str,
    *,
    latent_token_count: int,
) -> tuple[str, str, int]:
    """Preserve real source CoT and map its executed action to K16."""

    from nimloth.agent import NimlothPromptTemplate

    parsed = parse_source_response(response)
    if not parsed["format_valid"]:
        raise ValueError("source assistant response is not strict-valid")
    thought = str(parsed["thought"])
    action_name = str(parsed["action_name"])
    action_index = int(parsed["action_index"])
    template = NimlothPromptTemplate(
        latent_token_count=latent_token_count,
        action_count=len(SOURCE_ACTION_NAMES),
    )
    return (
        template.assistant_response(action_index, thought=thought),
        action_name,
        action_index,
    )


def _source_response_error_kind(response: str) -> str:
    if parse_source_response(response)["format_valid"]:
        return "ok"
    match = _SOURCE_ENVELOPE_RE.fullmatch(response)
    if match is None:
        return "missing_or_malformed_tags"
    answer = match.group(4)
    raw_actions = answer.split(",")
    if any(not item.strip() for item in raw_actions):
        return "missing_or_malformed_tags"
    actions = [item.strip() for item in raw_actions]
    if len(actions) > 1 and all(action in SOURCE_ACTION_NAMES for action in actions):
        return "too_many_actions"
    return "invalid_action_name"


def _generation_audit_reason(audit: Mapping[str, Any]) -> str | None:
    finish_reason = audit.get("finish_reason")
    stop_reason = audit.get("stop_reason")
    token_ids = audit.get("token_ids")
    eos_token_id = audit.get("eos_token_id")
    if finish_reason == "length":
        return "generation_length_truncated"
    if finish_reason != "stop":
        return "generation_finish_reason_invalid"
    if stop_reason is not None:
        return "generation_custom_stop"
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids)
        or isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
    ):
        return "generation_token_evidence_missing"
    if token_ids[-1] != eos_token_id:
        return "generation_eos_token_missing"
    return None


def _windowed_audit_messages(
    messages: list[dict[str, str]],
    *,
    window_size: int = 5,
) -> tuple[list[dict[str, str]], int]:
    if len(messages) < 2 or messages[0].get("role") != "system":
        raise ValueError("raw chat has no system/current observation")
    turns = (len(messages) - 2) // 2
    first_turn = max(0, turns - window_size)
    start = 1 + 2 * first_turn
    return [dict(messages[0]), *map(dict, messages[start:])], first_turn


def _validate_policy_request_audit(
    request: Mapping[str, Any],
    *,
    expected_response: str,
    expected_messages: list[dict[str, str]],
) -> None:
    if request.get("response") != expected_response:
        raise ValueError("policy request response does not match raw chat")
    if request.get("response_sha256") != hashlib.sha256(
        expected_response.encode("utf-8")
    ).hexdigest():
        raise ValueError("policy request response hash mismatch")
    expected_window, expected_first = _windowed_audit_messages(expected_messages)
    if request.get("message_window") != expected_window or request.get(
        "first_observation_index"
    ) != expected_first:
        raise ValueError("policy request message window mismatch")
    rendered_prompt = request.get("rendered_prompt")
    if not isinstance(rendered_prompt, str) or request.get(
        "rendered_prompt_sha256"
    ) != hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest():
        raise ValueError("policy request rendered prompt hash mismatch")


def validate_raw_reconstruction_semantics(source: Mapping[str, Any]) -> None:
    responses = source.get("assistant_responses")
    turns = source.get("turns")
    requests = source.get("policy_requests")
    terminal = source.get("terminal_generation")
    if not isinstance(responses, list) or not isinstance(turns, list):
        raise TypeError("raw reconstruction responses/turns must be lists")
    if not isinstance(requests, list) or not isinstance(terminal, Mapping):
        raise TypeError("raw reconstruction generation audit is incomplete")
    if len(responses) != len(turns):
        raise ValueError("raw reconstruction turns do not align with responses")
    system_prompt = source.get("system_prompt")
    observations = source.get("observation_texts")
    messages = source.get("messages")
    if not isinstance(system_prompt, str) or not isinstance(observations, list) or (
        not isinstance(messages, list)
    ):
        raise TypeError("raw source chat components are incomplete")
    if len(observations) != len(responses) + 1:
        raise ValueError("raw observations do not match ordinary responses")
    if any(not isinstance(row, Mapping) for row in requests):
        raise TypeError("raw policy request audit must contain mappings")
    ordinary_requests = [row for row in requests if row.get("kind") == "ordinary"]
    terminal_requests = [row for row in requests if row.get("kind") == "terminal"]
    if len(ordinary_requests) != len(responses) or len(terminal_requests) != 1:
        raise ValueError("raw reconstruction policy request counts drift")
    ordinary_format_valid = True
    for index, (response, turn, request) in enumerate(
        zip(responses, turns, ordinary_requests, strict=True)
    ):
        _validate_policy_request_audit(
            request,
            expected_response=str(response),
            expected_messages=messages[: 2 + 2 * index],
        )
        if turn.get("response") != response:
            raise ValueError("raw turn response does not match assistant responses")
        parsed = parse_source_response(str(response))
        ordinary_format_valid = ordinary_format_valid and bool(
            parsed["format_valid"]
        )
        if _generation_audit_reason(request) is not None:
            raise ValueError("completed raw row contains invalid ordinary boundary")
        extracted = turn.get("environment_extracted_actions")
        expected_extracted = (
            [parsed["action_name"]] if parsed["format_valid"] else []
        )
        if extracted != expected_extracted:
            raise ValueError("raw ordinary response/action extraction mismatch")
        if turn.get("parsed_response") != parsed:
            raise ValueError("raw ordinary parsed response mismatch")
        info = turn.get("info")
        if not isinstance(info, Mapping):
            raise TypeError("raw ordinary turn info must be a mapping")
        error_kind = _source_response_error_kind(str(response))
        if error_kind == "ok":
            expected_reward = 10.02 if bool(info.get("task_success")) else 0.02
        elif error_kind == "too_many_actions":
            expected_reward = 0.0
        else:
            expected_reward = -0.2
        actual_reward = float(turn.get("reward", float("nan")))
        if not math.isclose(
            actual_reward,
            expected_reward,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError("raw turn reward does not match parser/success class")
    flattened_actions = [
        action
        for turn in turns
        for action in turn.get("environment_extracted_actions", [])
    ]
    if source.get("executed_action_names") != flattened_actions:
        raise ValueError("raw executed action list does not match turns")
    terminal_response = terminal.get("assistant_response")
    _validate_policy_request_audit(
        terminal_requests[0],
        expected_response=str(terminal_response),
        expected_messages=messages[:-1],
    )
    terminal_parsed = parse_source_response(str(terminal_response))
    terminal_reason = _generation_audit_reason(terminal_requests[0])
    for field_name in ("finish_reason", "stop_reason", "token_ids", "eos_token_id"):
        if terminal.get(field_name) != terminal_requests[0].get(field_name):
            raise ValueError("raw terminal generation/request evidence mismatch")
    if terminal.get("generation_exclusion_reason") != terminal_reason:
        raise ValueError("raw terminal generation exclusion reason mismatch")
    if terminal.get("parsed") != terminal_parsed:
        raise ValueError("raw terminal parsed response mismatch")
    if terminal.get("executed") is not False or terminal.get(
        "environment_step_after_generation"
    ) is not False:
        raise ValueError("raw terminal draft action execution semantics drift")
    expected_eligible = bool(
        ordinary_format_valid
        and terminal_parsed["format_valid"]
        and terminal_reason is None
    )
    expected_reasons = []
    if not ordinary_format_valid or not terminal_parsed["format_valid"]:
        expected_reasons.append("source_response_format_invalid")
    if terminal_reason is not None:
        expected_reasons.append(terminal_reason)
    expected_messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(observations[0])},
    ]
    for index, response in enumerate(responses):
        expected_messages.extend(
            [
                {"role": "assistant", "content": str(response)},
                {"role": "user", "content": str(observations[index + 1])},
            ]
        )
    expected_messages.append(
        {"role": "assistant", "content": str(terminal_response)}
    )
    if messages != expected_messages:
        raise ValueError("raw full chat does not match observations/responses")
    if source.get("conversion_eligible") is not expected_eligible:
        raise ValueError("raw conversion eligibility does not match semantics")
    if source.get("exclusion_reasons") != sorted(set(expected_reasons)):
        raise ValueError("raw exclusion reasons do not match semantics")


def build_source_audit(source: Mapping[str, Any]) -> dict[str, Any]:
    raw_hash = source.get("raw_record_sha256")
    raw_payload = {
        key: value for key, value in source.items() if key != "raw_record_sha256"
    }
    if raw_hash != _canonical_sha256(raw_payload):
        raise ValueError("source raw record hash mismatch before conversion")
    terminal = source.get("terminal_generation")
    if not isinstance(terminal, Mapping):
        raise TypeError("source record has no terminal generation audit")
    terminal_response = terminal.get("assistant_response")
    if not isinstance(terminal_response, str) or not terminal_response:
        raise ValueError("source terminal generation has no full response")
    messages = source.get("messages")
    if not isinstance(messages, list):
        raise TypeError("source record messages must be a list")
    audit_payload = {
        "contract_version": SOURCE_AUDIT_CONTRACT_VERSION,
        "record_format": source.get("record_format"),
        "record_id": source.get("id"),
        "source_identity": {
            "source_index": source.get("source_index"),
            "source_key": source.get("source_key"),
            "eval_set": source.get("eval_set"),
            "seed": source.get("seed"),
            "batch": source.get("batch"),
            "split": source.get("split"),
        },
        "unavailable_source_commit": source.get("unavailable_source_commit"),
        "reconstruction_identity": source.get("reconstruction_identity"),
        "source_runtime_contract": source.get("source_runtime_contract"),
        "policy_artifact": source.get("policy_artifact"),
        "policy_runtime_contract": source.get("policy_runtime_contract"),
        "system_prompt": source.get("system_prompt"),
        "messages": messages,
        "observation_texts": source.get("observation_texts", []),
        "assistant_responses": source.get("assistant_responses", []),
        "executed_action_names": source.get("executed_action_names", []),
        "turns": source.get("turns", []),
        "image_artifacts": source.get("image_artifacts", []),
        "rewards": source.get("rewards", []),
        "environment_reward_events": source.get("environment_reward_events", []),
        "reward": source.get("reward"),
        "reward_provenance": source.get("reward_provenance"),
        "success": source.get("success"),
        "environment_done": source.get("environment_done"),
        "terminal_generation": dict(terminal),
        "terminal_assistant_response": terminal_response,
        "policy_requests": source.get("policy_requests", []),
    }
    return {
        **audit_payload,
        "raw_record_sha256": raw_hash,
        "audit_payload_sha256": _canonical_sha256(audit_payload),
        "source_sha256": raw_hash,
    }


def _resolved_image_paths(
    values: Any,
    *,
    source_root: Path | None,
) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError("source record image_paths must be a non-empty list")
    paths: list[str] = []
    for value in values:
        path = Path(str(value))
        if source_root is not None and not path.is_absolute():
            path = (source_root / path).resolve()
        if source_root is not None and not path.is_file():
            raise ValueError(f"source record image does not exist: {path}")
        paths.append(str(path))
    return paths


def _validate_source_reward_evidence(
    source: Mapping[str, Any],
) -> tuple[str, list[float], float]:
    runtime_contract = source.get("source_runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        raise TypeError("source record has no runtime reward contract")
    reward_provenance = source.get("reward_provenance")
    if reward_provenance != runtime_contract.get("reward_provenance"):
        raise ValueError("source row/runtime reward provenance mismatch")
    if reward_provenance != "step_rewards":
        raise ValueError("reconstruction reward provenance must be step_rewards")

    turns = source.get("turns")
    events = source.get("environment_reward_events")
    if not isinstance(turns, list) or not isinstance(events, list):
        raise TypeError("source reward turns/events must be lists")
    turn_rewards: list[float] = []
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise TypeError("source reward turn must be a mapping")
        value = float(turn.get("reward", float("nan")))
        if not math.isfinite(value):
            raise ValueError("source turn rewards contain non-finite values")
        turn_rewards.append(value)
    event_rewards = [float(value) for value in events]
    if not all(math.isfinite(value) for value in event_rewards):
        raise ValueError("source environment reward events contain non-finite values")
    if len(turn_rewards) != len(event_rewards) or any(
        not math.isclose(turn, event, rel_tol=1e-6, abs_tol=1e-7)
        for turn, event in zip(turn_rewards, event_rewards, strict=True)
    ):
        raise ValueError("source turn rewards and environment events disagree")

    raw_aggregate_reward = float(source.get("reward", float("nan")))
    if not math.isfinite(raw_aggregate_reward):
        raise ValueError("source aggregate reward is non-finite")
    source_rewards = [float(value) for value in source.get("rewards", [])]
    if not all(math.isfinite(value) for value in source_rewards):
        raise ValueError("source rewards contain non-finite values")
    if len(source_rewards) != len(event_rewards) or any(
        not math.isclose(step, event, rel_tol=1e-6, abs_tol=1e-7)
        for step, event in zip(source_rewards, event_rewards, strict=True)
    ):
        raise ValueError("source step rewards and environment events disagree")
    step_aggregate = sum(source_rewards)
    if not math.isfinite(step_aggregate) or not math.isclose(
        raw_aggregate_reward,
        step_aggregate,
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise ValueError("source aggregate reward does not equal step rewards")
    return str(reward_provenance), source_rewards, raw_aggregate_reward


def convert_source_record(
    source: Mapping[str, Any],
    *,
    latent_token_count: int,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Build linked SFT1 and SFT2 training views from one strict raw record."""

    from nimloth.agent import NimlothPromptTemplate
    from nimloth.rollout.record_format import (
        STEP_REWARD_PROVENANCE,
        TRAJECTORY_RECORD_FORMAT,
    )

    if latent_token_count != 16:
        raise ValueError("step60 SFT1/SFT2 conversion requires K16")
    if source.get("record_format") != RECONSTRUCTION_FORMATS["raw_row"]:
        raise ValueError("unsupported step60 source trajectory format")
    runtime_contract = source.get("source_runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        raise TypeError("source record has no reconstruction runtime contract")
    validate_reconstruction_format(
        "runtime_contract",
        str(runtime_contract.get("format", "")),
    )
    if source.get("unavailable_source_commit") != UNAVAILABLE_SOURCE_COMMIT:
        raise ValueError("source unavailable commit provenance mismatch")
    if runtime_contract.get("reconstruction_identity") != source.get(
        "reconstruction_identity"
    ):
        raise ValueError("source reconstruction identity mismatch")
    validate_raw_reconstruction_semantics(source)
    if source.get("conversion_eligible") is False:
        raise ValueError(
            "source record is conversion-ineligible: "
            f"{source.get('exclusion_reasons', [])!r}"
        )
    system_prompt = source.get("system_prompt")
    observations = source.get("observation_texts")
    responses = source.get("assistant_responses")
    executed_actions = source.get("executed_action_names")
    if not isinstance(system_prompt, str) or not system_prompt:
        raise ValueError("source record has no system prompt")
    if not isinstance(observations, list) or not isinstance(responses, list):
        raise TypeError("source observations/responses must be lists")
    if not isinstance(executed_actions, list):
        raise TypeError("source executed_action_names must be a list")
    if len(observations) != len(responses) + 1:
        raise ValueError("source observations must equal ordinary responses + 1")
    if len(executed_actions) != len(responses):
        raise ValueError("source executed actions do not align with responses")
    if not responses:
        raise ValueError("source trajectory has no executed actions")

    converted_system = convert_source_prompt(
        system_prompt,
        latent_token_count=latent_token_count,
    )
    converted_observations = [
        convert_source_prompt(str(text), latent_token_count=latent_token_count)
        for text in observations
    ]
    converted_responses: list[str] = []
    action_names: list[str] = []
    action_indices: list[int] = []
    for expected_action, response in zip(
        executed_actions,
        responses,
        strict=True,
    ):
        converted, action_name, action_index = convert_source_assistant_response(
            str(response),
            latent_token_count=latent_token_count,
        )
        if action_name != str(expected_action):
            raise ValueError(
                "source response/environment action mismatch: "
                f"{action_name!r} != {expected_action!r}"
            )
        converted_responses.append(converted)
        action_names.append(action_name)
        action_indices.append(action_index)

    terminal = source.get("terminal_generation")
    assert isinstance(terminal, Mapping)
    terminal_response = str(terminal["assistant_response"])
    terminal_parse = (
        dict(terminal["parsed"])
        if isinstance(terminal.get("parsed"), Mapping)
        else parse_source_response(terminal_response)
    )
    if not terminal_parse.get("format_valid"):
        raise ValueError("source terminal response is not strict-valid")
    terminal_index = int(terminal_parse["action_index"])
    terminal_name = str(terminal_parse["action_name"])
    if terminal.get("draft_action_name") is not None and (
        str(terminal["draft_action_name"]) != terminal_name
    ):
        raise ValueError("source terminal draft-action audit mismatch")
    if terminal.get("executed") is not False:
        raise ValueError("source terminal draft action was not marked unexecuted")
    if terminal.get("environment_step_after_generation", False) is not False:
        raise ValueError("source terminal generation has a following environment step")
    template = NimlothPromptTemplate(
        latent_token_count=latent_token_count,
        action_count=len(SOURCE_ACTION_NAMES),
    )
    terminal_prefix = template.assistant_prefix(
        thought=str(terminal_parse["thought"])
    )
    image_paths = _resolved_image_paths(
        source.get("image_paths"),
        source_root=source_root,
    )
    if len(image_paths) != len(observations):
        raise ValueError("source image/observation counts do not align")

    audit = build_source_audit(source)
    provenance = {
        "format": CONVERSION_FORMAT,
        "source_sha256": audit["source_sha256"],
        "latent_token_count": latent_token_count,
        "rewrite_scope": "action-format only; source task text, observations and CoT preserved",
        "source_action_envelope": "<answer>one_action</answer>",
        "training_action_envelope": "K16 Nimloth latent/action tokens",
    }
    sft1_messages: list[dict[str, str]] = [
        {"role": "system", "content": converted_system}
    ]
    for observation, response in zip(
        converted_observations[:-1],
        converted_responses,
        strict=True,
    ):
        sft1_messages.extend(
            [
                {"role": "user", "content": observation},
                {"role": "assistant", "content": response},
            ]
        )
    record_id = str(source["id"])
    split = str(source["split"])
    source_key = source.get("source_key")
    if not isinstance(source_key, str) or not source_key:
        raise ValueError("source record has no source_key")
    batch = source.get("batch")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch != 1:
        raise ValueError("source record is not explicitly bound to batch1")
    source_identity = {
        "source_index": int(source["source_index"]),
        "source_key": source_key,
        "eval_set": str(source["eval_set"]),
        "seed": int(source["seed"]),
        "batch": batch,
    }
    sft1 = {
        "id": record_id,
        **source_identity,
        "source_identity": {**source_identity, "split": split},
        "split": split,
        "success": bool(source.get("success", False)),
        "reward": float(source.get("reward", 0.0)),
        "system_prompt": converted_system,
        "messages": sft1_messages,
        "image_paths": image_paths[:-1],
        "actions": action_names,
        "action_indices": action_indices,
        "source_audit": audit,
        "conversion_provenance": provenance,
    }

    (
        resolved_reward_provenance,
        source_rewards,
        raw_aggregate_reward,
    ) = _validate_source_reward_evidence(source)
    if resolved_reward_provenance != STEP_REWARD_PROVENANCE:
        raise ValueError("reconstruction reward provenance must be step_rewards")
    if len(source_rewards) != len(action_indices):
        raise ValueError("source step rewards do not align with actions")
    rewards = source_rewards
    terminated = bool(source.get("terminated", False))
    truncated = bool(source.get("truncated", False))
    if terminated == truncated:
        raise ValueError("source step-reward status must be terminal xor truncated")
    aggregate_reward = sum(rewards)
    if not math.isclose(
        raw_aggregate_reward,
        aggregate_reward,
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise ValueError("source aggregate reward does not equal step rewards")
    aggregate_reward = raw_aggregate_reward

    instruction_match = re.search(
        r"Human Instruction:\s*(.+?)(?:\n|$)",
        str(observations[0]),
    )
    if instruction_match is None:
        raise ValueError("source initial observation has no instruction")
    terminal_generation_audit = {
        "source_response": terminal_response,
        "draft_action_name": terminal_name,
        "draft_action_index": terminal_index,
        "format_valid": True,
        "executed": False,
        "environment_step_after_generation": False,
    }
    sft2 = {
        "record_format": TRAJECTORY_RECORD_FORMAT,
        "id": record_id,
        "source_identity": {**source_identity, "split": split},
        "split": split,
        "success": bool(source.get("success", False)),
        "reward": aggregate_reward,
        "reward_provenance": resolved_reward_provenance,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "image_paths": image_paths,
        "action_indices": action_indices,
        "action_names": action_names,
        "action_log_probs": [],
        "instruction": instruction_match.group(1).strip(),
        "system_prompt": converted_system,
        "observation_texts": converted_observations,
        "policy_messages": [],
        "assistant_responses": converted_responses,
        "terminal_assistant_prefix": terminal_prefix,
        "state_anchor_steps": [],
        "state_latent_hiddens": [],
        "world_model_states": [],
        "policy_credit_assignment": "none",
        "policy_step_indices": [],
        "policy_token_ids": [],
        "policy_token_log_probs": [],
        "policy_reference_token_log_probs": [],
        "policy_loss_masks": [],
        "policy_token_roles": [],
        "policy_action_token_ids": [],
        "policy_reasoning_texts": [],
        "policy_finish_reasons": [],
        "policy_reasoning_truncated": [],
        "planner_policy_traces": [],
        "source_audit": audit,
        "terminal_generation_audit": terminal_generation_audit,
        "conversion_provenance": provenance,
        "prompt_template": template.spec.to_record(),
        "sampling_temperature": 0.7,
        "sampling_top_p": 0.95,
        "action_space_id": "navigation",
        "action_space_version": 1,
    }
    sft1["conversion_provenance"] = {
        **provenance,
        "converted_sha256": _canonical_sha256(
            {key: value for key, value in sft1.items() if key != "conversion_provenance"}
        ),
    }
    sft2["conversion_provenance"] = {
        **provenance,
        "converted_sha256": _canonical_sha256(
            {key: value for key, value in sft2.items() if key != "conversion_provenance"}
        ),
    }
    return {"source_audit": audit, "sft1": sft1, "sft2": sft2}


def measure_identity_overlap(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure overlap by the verified navigation row identities."""

    reference_keys = {
        _row_identity(row, source_index=index)
        for index, row in enumerate(reference_rows)
    }
    candidate_keys = {
        _row_identity(row, source_index=index)
        for index, row in enumerate(candidate_rows)
    }
    reference_seeds = {seed for _eval_set, seed in reference_keys}
    candidate_seeds = {seed for _eval_set, seed in candidate_keys}
    overlapping_keys = sorted(reference_keys & candidate_keys)
    overlapping_seeds = sorted(reference_seeds & candidate_seeds)
    return {
        "reference_rows": len(reference_rows),
        "candidate_rows": len(candidate_rows),
        "reference_unique_eval_set_seed": len(reference_keys),
        "candidate_unique_eval_set_seed": len(candidate_keys),
        "eval_set_seed_overlap_count": len(overlapping_keys),
        "seed_overlap_count": len(overlapping_seeds),
        "overlapping_eval_set_seed": [
            {"eval_set": eval_set, "seed": seed}
            for eval_set, seed in overlapping_keys
        ],
        "overlapping_seeds": overlapping_seeds,
    }


def require_nonoverlapping_heldout(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_name: str,
) -> dict[str, Any]:
    """Reject a candidate called held-out when its task keys overlap."""

    if not candidate_name:
        raise ValueError("candidate_name must be non-empty")
    evidence = measure_identity_overlap(reference_rows, candidate_rows)
    if evidence["eval_set_seed_overlap_count"]:
        raise ValueError(
            f"{candidate_name} is not held-out: "
            f"{evidence['eval_set_seed_overlap_count']} eval_set/seed keys overlap"
        )
    return evidence


def _validate_shard_runtime_policy_contract(manifest: Mapping[str, Any]) -> None:
    from experiments.training.sft1 import vagen_step60_collect as collect_contract

    runtime_contract = manifest["source_runtime_contract"]
    identity = manifest["reconstruction_identity"]
    expected_identity = {
        **identity,
        "base_commit": collect_contract.RECONSTRUCTION_BASE_COMMIT,
        "runtime_parent": collect_contract.RECONSTRUCTION_BASE_COMMIT,
        "runtime_head": collect_contract.APPROVED_RECONSTRUCTION_HEAD,
        "runtime_tree": collect_contract.APPROVED_RECONSTRUCTION_TREE,
        "diff_sha256": collect_contract.APPROVED_RECONSTRUCTION_DIFF_SHA256,
        "commit_count": 1,
        "parent_count": 1,
    }
    collect_contract.validate_reconstruction_git_identity(
        dict(identity),
        expected=expected_identity,
    )
    collect_contract.validate_source_runtime_contract(
        dict(runtime_contract),
        expected_reconstruction_identity=expected_identity,
    )
    policy_artifact = manifest.get("policy_artifact")
    policy_runtime = manifest.get("policy_runtime_contract")
    if not isinstance(policy_artifact, Mapping) or not isinstance(
        policy_runtime, Mapping
    ):
        raise TypeError("shard policy provenance must be mappings")
    if policy_runtime.get("backend") != "vllm":
        raise ValueError("shard policy backend mismatch")
    for key, expected in collect_contract.SOURCE_SAMPLING_CONTRACT.items():
        if policy_runtime.get(key) != expected:
            raise ValueError(f"shard policy sampling/package drift: {key}")
    expected_packages = collect_contract.EXECUTABLE_GENERATION_PACKAGES
    if policy_runtime.get("package_versions") != expected_packages:
        raise ValueError("shard policy package versions mismatch")
    if policy_runtime.get("source_generation_package_evidence") != (
        collect_contract.SOURCE_GENERATION_PACKAGE_EVIDENCE
    ) or runtime_contract.get("source_generation_package_evidence") != (
        collect_contract.SOURCE_GENERATION_PACKAGE_EVIDENCE
    ):
        raise ValueError("shard source package evidence mismatch")
    if policy_runtime.get("executable_generation_packages") != expected_packages or (
        runtime_contract.get("executable_generation_packages")
        != expected_packages
    ):
        raise ValueError("shard executable package identity mismatch")
    if policy_runtime.get("model_config_artifacts") != policy_artifact.get(
        "model_config_artifacts"
    ):
        raise ValueError("shard policy model/tokenizer identity mismatch")
    for key in (
        "merge_manifest_file_sha256",
        "merge_manifest_payload_sha256",
        "artifact_manifest_sha256",
    ):
        value = policy_artifact.get(key)
        if not isinstance(value, str) or _HEX_SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"shard policy artifact identity invalid: {key}")
    eos_token_id = policy_runtime.get("tokenizer_eos_token_id")
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise TypeError("shard policy tokenizer EOS identity is invalid")


def validate_complete_shard(
    shard_dir: Path,
    *,
    expected_source_indices: set[int],
) -> dict[str, Any]:
    """Accept only a hash-bound shard whose COMPLETE marker was published last."""

    shard_dir = validate_published_directory(
        shard_dir,
        readiness_marker="COMPLETE",
    )
    complete_path = shard_dir / "COMPLETE"
    manifest_path = shard_dir / "shard_manifest.json"
    raw_path = shard_dir / "raw.jsonl"
    for control_name in ("IN_PROGRESS.json", "records", "attempts"):
        control_path = shard_dir / control_name
        if control_path.is_symlink():
            raise ValueError(f"complete shard control path is a symlink: {control_name}")
    if (shard_dir / "IN_PROGRESS.json").exists() and not (
        shard_dir / "IN_PROGRESS.json"
    ).is_file():
        raise ValueError("complete shard in-progress metadata is invalid")
    for directory_name in ("records", "attempts"):
        directory = shard_dir / directory_name
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"complete shard control directory is invalid: {directory_name}")
    if complete_path.is_symlink() or manifest_path.is_symlink() or raw_path.is_symlink():
        raise ValueError("complete shard root artifacts must not be symlinks")
    if not complete_path.is_file():
        raise ValueError(f"shard has no COMPLETE marker: {shard_dir}")
    if not manifest_path.is_file() or not raw_path.is_file():
        raise ValueError("complete shard is missing its manifest or raw JSONL")
    marker = json.loads(complete_path.read_text(encoding="utf-8"))
    if marker.get("format") != RECONSTRUCTION_FORMATS["complete_marker"]:
        raise ValueError("unsupported COMPLETE marker reconstruction format")
    manifest_bytes_sha256 = _file_sha256(manifest_path)
    if marker.get("manifest_sha256") != manifest_bytes_sha256:
        raise ValueError("COMPLETE marker does not match shard manifest bytes")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != RECONSTRUCTION_FORMATS["shard_manifest"]:
        raise ValueError(f"unsupported shard manifest format: {manifest.get('format')!r}")
    if manifest.get("status") != "complete":
        raise ValueError("shard manifest status is not complete")
    runtime_contract = manifest.get("source_runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        raise TypeError("shard manifest has no reconstruction runtime contract")
    validate_reconstruction_format(
        "runtime_contract",
        str(runtime_contract.get("format", "")),
    )
    if runtime_contract.get("reward_provenance") != "step_rewards":
        raise ValueError("shard reconstruction reward provenance must be step_rewards")
    if manifest.get("unavailable_source_commit") != UNAVAILABLE_SOURCE_COMMIT:
        raise ValueError("shard unavailable source provenance mismatch")
    reconstruction_identity = manifest.get("reconstruction_identity")
    if not isinstance(reconstruction_identity, Mapping) or (
        runtime_contract.get("reconstruction_identity")
        != reconstruction_identity
    ):
        raise ValueError("shard reconstruction identity mismatch")
    _validate_shard_runtime_policy_contract(manifest)
    source_indices = [int(value) for value in manifest.get("source_indices", [])]
    if len(source_indices) != len(set(source_indices)):
        raise ValueError("shard manifest contains duplicate source indices")
    if set(source_indices) != expected_source_indices:
        raise ValueError(
            "shard source indices do not match the requested shard: "
            f"actual={sorted(source_indices)}, "
            f"expected={sorted(expected_source_indices)}"
        )
    raw_contract = manifest.get("raw_jsonl")
    if not isinstance(raw_contract, Mapping):
        raise TypeError("shard manifest has no raw_jsonl contract")
    if raw_contract.get("path") != raw_path.name:
        raise ValueError("shard manifest raw_jsonl path drift")
    if raw_contract.get("sha256") != _file_sha256(raw_path):
        raise ValueError("shard raw JSONL hash mismatch")

    records: list[dict[str, Any]] = []
    with raw_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"raw JSONL contains blank line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"raw JSONL line {line_number} is not an object")
            if row.get("record_format") != RECONSTRUCTION_FORMATS["raw_row"]:
                raise ValueError(f"raw JSONL line {line_number} format drift")
            raw_hash = row.get("raw_record_sha256")
            raw_payload = {
                key: value
                for key, value in row.items()
                if key != "raw_record_sha256"
            }
            if raw_hash != _canonical_sha256(raw_payload):
                raise ValueError(f"raw JSONL line {line_number} record hash mismatch")
            records.append(row)
    if len(records) != int(raw_contract.get("count", -1)):
        raise ValueError("shard raw JSONL count does not match manifest")
    record_indices = [int(row["source_index"]) for row in records]
    if record_indices != source_indices:
        raise ValueError("raw record order/identity does not match shard manifest")
    source_keys = [str(row.get("source_key", "")) for row in records]
    if source_keys != manifest.get("source_keys") or any(not key for key in source_keys):
        raise ValueError("raw source keys do not match shard manifest")
    for row in records:
        if int(row.get("batch", -1)) != 1:
            raise ValueError("raw record is not bound to batch1")
        runtime_contract = row.get("source_runtime_contract")
        if not isinstance(runtime_contract, Mapping) or row.get(
            "reward_provenance"
        ) != runtime_contract.get("reward_provenance"):
            raise ValueError("raw record reward provenance does not match runtime")
        _validate_source_reward_evidence(row)
        validate_raw_reconstruction_semantics(row)
        for field_name in (
            "unavailable_source_commit",
            "reconstruction_identity",
            "source_runtime_contract",
            "policy_artifact",
            "policy_runtime_contract",
            "format_failure_policy",
        ):
            if row.get(field_name) != manifest.get(field_name):
                raise ValueError(
                    f"raw record {field_name} does not match shard manifest"
                )

    declared_images = manifest.get("images")
    if not isinstance(declared_images, list):
        raise TypeError("shard manifest has no image artifact list")
    referenced_images = [
        str(path)
        for row in records
        for path in row.get("image_paths", [])
    ]
    record_image_artifacts = [
        artifact
        for row in records
        for artifact in row.get("image_artifacts", [])
    ]
    if record_image_artifacts != declared_images:
        raise ValueError("raw image artifact evidence does not match shard manifest")
    declared_paths: list[str] = []
    for artifact in declared_images:
        if not isinstance(artifact, Mapping):
            raise TypeError("shard image artifact entry is not a mapping")
        relative = Path(str(artifact.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"invalid shard-relative image path: {relative}")
        path = shard_dir
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError(f"shard image path has symlinked ancestor: {relative}")
        if not path.is_file():
            raise ValueError(f"shard image is missing: {relative}")
        if int(artifact.get("size_bytes", -1)) != path.stat().st_size:
            raise ValueError(f"shard image size mismatch: {relative}")
        if artifact.get("sha256") != _file_sha256(path):
            raise ValueError(f"shard image hash mismatch: {relative}")
        declared_paths.append(str(relative))
    if declared_paths != referenced_images:
        raise ValueError("raw image references do not match image artifact manifest")
    expected_counts = {
        "records": len(records),
        "eligible": sum(bool(row.get("conversion_eligible")) for row in records),
        "excluded": sum(not bool(row.get("conversion_eligible")) for row in records),
        "transitions": sum(len(row.get("turns", [])) for row in records),
        "images": len(declared_images),
        "terminal_generations": sum(
            isinstance(row.get("terminal_generation"), Mapping) for row in records
        ),
        "terminal_environment_steps": sum(
            bool(row.get("terminal_generation", {}).get("environment_step_after_generation"))
            for row in records
            if isinstance(row.get("terminal_generation"), Mapping)
        ),
    }
    if manifest.get("counts") != expected_counts:
        raise ValueError("shard manifest counts do not match raw records")
    return manifest


def partition_source_parquet(
    source_path: Path,
    output_dir: Path,
    *,
    expected_sha256: str = SOURCE_TRAIN_SHA256,
) -> dict[str, Any]:
    """Write all ten pinned parquet batches and one atomic manifest."""

    output_dir = lexical_absolute_path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"partition output already exists: {output_dir}")
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source parquet does not exist: {source_path}")
    actual_sha256 = _file_sha256(source_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"source parquet SHA256 drift: {actual_sha256} != {expected_sha256}"
        )

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - remote/runtime dependency
        raise RuntimeError("partitioning source parquet requires pyarrow") from error

    table = pq.read_table(source_path)
    rows = table.to_pylist()
    manifest = build_partition_manifest(
        rows,
        source_path=str(source_path),
        source_sha256=actual_sha256,
    )
    manifest["source"]["size_bytes"] = source_path.stat().st_size

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        for batch in manifest["batches"]:
            batch_number = int(batch["batch"])
            indices = batch["source_indices"]
            batch_path = temporary / f"batch_{batch_number:02d}.parquet"
            pq.write_table(table.take(pa.array(indices)), batch_path)
            batch["parquet"] = batch_path.name
            batch["parquet_sha256"] = _file_sha256(batch_path)
            batch["parquet_size_bytes"] = batch_path.stat().st_size
        manifest["manifest_payload_sha256"] = partition_manifest_payload_sha256(
            manifest
        )
        validate_partition_manifest(manifest, require_published=True)
        manifest_path = temporary / "partition_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        publish_reserved_directory(
            temporary,
            output_dir,
            readiness_marker="partition_manifest.json",
        )
        manifest = load_published_partition_manifest(
            output_dir / "partition_manifest.json"
        )
    except Exception as error:
        failure = (
            temporary / "FAILED_PUBLISH.json"
            if temporary.exists()
            else output_dir / "FAILED_VALIDATION.json"
        )
        if failure.parent.exists() and not failure.exists():
            failure.write_text(
                json.dumps(
                    {"error_type": type(error).__name__, "error": str(error)},
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Partition the pinned VAGEN step60 source train parquet"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = partition_source_parquet(args.source, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "source_sha256": manifest["source"]["sha256"],
                "batches": len(manifest["batches"]),
                "batch1_train": manifest["checks"]["batch1_train_count"],
                "batch1_heldout": manifest["checks"]["batch1_heldout_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
