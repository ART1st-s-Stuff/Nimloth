"""Qwen transition 预处理缓存的公开 API。"""

from nimloth.util.cache.build import (
    build_compact_transition_preprocess_cache,
    build_transition_preprocess_cache,
)
from nimloth.util.cache.dataset import (
    CachedTransitionDataset,
    CompactCachedTransitionCollator,
)
from nimloth.util.cache.encoding import (
    encode_qwen_item_from_image_grids,
    encode_transition_item,
)
from nimloth.util.cache.schema import (
    CE_MASK_VERSION,
    COMPACT_CACHE_FORMAT,
    COMPACT_CACHE_FORMAT_V1,
    DEFAULT_MIN_PIXELS,
    LEGACY_CACHE_FORMAT,
    LEGACY_TRANSITION_EXPANSION_VERSION,
    TRANSITION_EXPANSION_VERSION,
    SUPPORTED_COMPACT_CACHE_FORMATS,
    cache_fingerprint,
    safe_cache_name,
    transition_sample_id,
)

__all__ = [
    "CE_MASK_VERSION",
    "COMPACT_CACHE_FORMAT",
    "COMPACT_CACHE_FORMAT_V1",
    "CachedTransitionDataset",
    "CompactCachedTransitionCollator",
    "DEFAULT_MIN_PIXELS",
    "LEGACY_CACHE_FORMAT",
    "LEGACY_TRANSITION_EXPANSION_VERSION",
    "TRANSITION_EXPANSION_VERSION",
    "SUPPORTED_COMPACT_CACHE_FORMATS",
    "build_compact_transition_preprocess_cache",
    "build_transition_preprocess_cache",
    "cache_fingerprint",
    "encode_qwen_item_from_image_grids",
    "encode_transition_item",
    "safe_cache_name",
    "transition_sample_id",
]
