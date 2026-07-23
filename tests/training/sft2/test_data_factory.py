import json
from types import SimpleNamespace

import pytest

from nimloth.training.sft2.data.factory import _verify_cache_manifest
from nimloth.util.cache import (
    COMPACT_CACHE_FORMAT_V1,
    LEGACY_TRANSITION_EXPANSION_VERSION,
    cache_fingerprint,
)


class _Tokenizer:
    def __len__(self) -> int:
        return 32


def _cache_fixture(tmp_path, *, cached_count: int):
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text('{"id":"record"}\n', encoding="utf-8")
    model_path = tmp_path / "model"
    model_path.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    config = SimpleNamespace(
        require_prebuilt_cache=True,
        preprocess_cache_format="compact",
        max_length=128,
        max_pixels=100352,
        value_gamma=1.0,
        latent_token_count=16,
        mask_latent_query_labels=True,
        preprocess_cache_image_dtype="bfloat16",
        model=model_path,
    )
    fingerprint = cache_fingerprint(
        jsonl_path,
        max_length=config.max_length,
        max_pixels=config.max_pixels,
        min_pixels=3136,
        vocab_size=32,
        value_gamma=config.value_gamma,
        latent_token_count=config.latent_token_count,
        mask_latent_query_labels=config.mask_latent_query_labels,
        cache_format=COMPACT_CACHE_FORMAT_V1,
        image_dtype=config.preprocess_cache_image_dtype,
        processor_source=str(model_path.resolve()),
        transition_expansion_version=LEGACY_TRANSITION_EXPANSION_VERSION,
    )
    (cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": COMPACT_CACHE_FORMAT_V1,
                "base_fingerprint": fingerprint,
                "count": cached_count,
            }
        ),
        encoding="utf-8",
    )
    return cache_dir, jsonl_path, config, SimpleNamespace(tokenizer=_Tokenizer())


def test_full_prebuilt_cache_can_serve_an_explicit_prefix_subset(tmp_path) -> None:
    cache_dir, jsonl_path, config, processor = _cache_fixture(
        tmp_path,
        cached_count=100,
    )

    _verify_cache_manifest(
        cache_dir=cache_dir,
        jsonl_path=jsonl_path,
        expected_count=8,
        allow_prefix_subset=True,
        config=config,
        processor=processor,
    )


@pytest.mark.parametrize(
    ("expected_count", "allow_prefix_subset"),
    [(101, True), (8, False)],
)
def test_prebuilt_cache_rejects_non_prefix_count_mismatches(
    tmp_path,
    expected_count: int,
    allow_prefix_subset: bool,
) -> None:
    cache_dir, jsonl_path, config, processor = _cache_fixture(
        tmp_path,
        cached_count=100,
    )

    with pytest.raises(ValueError, match="fingerprint/count mismatch"):
        _verify_cache_manifest(
            cache_dir=cache_dir,
            jsonl_path=jsonl_path,
            expected_count=expected_count,
            allow_prefix_subset=allow_prefix_subset,
            config=config,
            processor=processor,
        )
