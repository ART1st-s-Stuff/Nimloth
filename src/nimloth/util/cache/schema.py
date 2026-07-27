"""Qwen transition 预处理缓存的 schema 和身份标识工具。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from nimloth.rollout.transitions import TransitionSample

CE_MASK_VERSION = "last_assistant_span_v1"
TRANSITION_EXPANSION_VERSION = "wm_expand_v3_terminal_cot"
DEFAULT_MIN_PIXELS = 3136
COMPACT_CACHE_FORMAT = "dedup_sharded_v2"


def safe_cache_name(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace(" ", "_")


def transition_sample_id(sample: TransitionSample) -> str:
    return f"{sample.record_id}:{sample.step_index}"


def cache_fingerprint(
    jsonl_path: Path,
    *,
    max_length: int,
    max_pixels: int,
    min_pixels: int,
    vocab_size: int,
    value_gamma: float = 1.0,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
    cache_format: str = COMPACT_CACHE_FORMAT,
    image_dtype: str = "float32",
    processor_source: str = "",
    transition_expansion_version: str = TRANSITION_EXPANSION_VERSION,
) -> str:
    stat = jsonl_path.stat()
    payload = "|".join(
        [
            str(jsonl_path.resolve()),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            str(max_length),
            str(max_pixels),
            str(min_pixels),
            str(vocab_size),
            str(value_gamma),
            str(latent_token_count),
            "inject" if mask_latent_query_labels else "generate",
            str(mask_latent_query_labels),
            cache_format,
            image_dtype,
            processor_source,
            CE_MASK_VERSION,
            transition_expansion_version,
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
