"""Build one Stage3 split's production transition cache on CPU, before model loading."""
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoProcessor

from nimloth.latent import add_special_tokens
from nimloth.util.cache import build_compact_transition_preprocess_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'processor'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--latent-token-count', type=int, required=True)
    parser.add_argument('--max-length', type=int, required=True)
    parser.add_argument('--max-pixels', type=int, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--gamma', type=float, default=1.0)
    args = parser.parse_args(argv)
    if min(args.latent_token_count, args.max_length, args.max_pixels, args.workers) < 1:
        parser.error('dimensions and worker count must be positive')
    if not 0 <= args.gamma <= 1:
        parser.error('gamma must be in [0, 1]')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('cache output must be new')
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    add_special_tokens(processor.tokenizer, latent_token_count=args.latent_token_count)
    args.output.mkdir(parents=True, exist_ok=False)
    build_compact_transition_preprocess_cache(
        jsonl_path=args.source, cache_dir=args.output, model_path=args.processor,
        processor=processor, max_length=args.max_length, max_pixels=args.max_pixels,
        preprocess_workers=args.workers, value_gamma=args.gamma,
        latent_token_count=args.latent_token_count, mask_latent_query_labels=True,
        success_only=False, force=False, image_dtype='bfloat16',
        image_shard_size=128, transition_shard_size=256,
    )


if __name__ == '__main__':
    main()
