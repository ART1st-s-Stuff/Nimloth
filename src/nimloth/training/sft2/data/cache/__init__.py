"""Public SFT2 transition-cache API."""

from nimloth.training.sft2.data.cache.build import (
    build_compact_transition_preprocess_cache,
    build_transition_preprocess_cache,
)
from nimloth.training.sft2.data.cache.dataset import (
    CachedTransitionDataset,
    CompactCachedTransitionCollator,
)
from nimloth.training.sft2.data.cache.encoding import (
    encode_qwen_item_from_image_grids,
    encode_transition_item,
)
from nimloth.training.sft2.data.cache.schema import (
    CE_MASK_VERSION,
    COMPACT_CACHE_FORMAT,
    DEFAULT_MIN_PIXELS,
    LEGACY_CACHE_FORMAT,
    TRANSITION_EXPANSION_VERSION,
    cache_fingerprint,
    safe_cache_name,
    transition_sample_id,
)

__all__ = [
    "CE_MASK_VERSION",
    "COMPACT_CACHE_FORMAT",
    "CachedTransitionDataset",
    "CompactCachedTransitionCollator",
    "DEFAULT_MIN_PIXELS",
    "LEGACY_CACHE_FORMAT",
    "TRANSITION_EXPANSION_VERSION",
    "build_compact_transition_preprocess_cache",
    "build_transition_preprocess_cache",
    "cache_fingerprint",
    "encode_qwen_item_from_image_grids",
    "encode_transition_item",
    "safe_cache_name",
    "transition_sample_id",
]
