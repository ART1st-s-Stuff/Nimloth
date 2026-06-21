#!/usr/bin/env python3
from pathlib import Path

from transformers import AutoProcessor

from nimloth.latent import add_special_tokens
from nimloth.training.common.qwen_batch import encode_qwen_item
from nimloth.wm.collate import prefix_messages_with_images
from nimloth.wm.dataset import expand_record_transitions, load_jsonl_records
from experiments.training.sft2.probe_kv_incremental import _vision_delta


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 602112
    add_special_tokens(processor.tokenizer)
    record = load_jsonl_records(args.train_jsonl, max_records=1)[0]
    trans = expand_record_transitions(record)
    enc_prev = encode_qwen_item(prefix_messages_with_images(trans[0]), processor, 12000, include_labels=False)
    enc_cur = encode_qwen_item(prefix_messages_with_images(trans[1]), processor, 12000, include_labels=False)
    print("prev", enc_prev["input_ids"].shape, enc_prev.get("pixel_values").shape, enc_prev.get("image_grid_thw"))
    print("cur", enc_cur["input_ids"].shape, enc_cur.get("pixel_values").shape, enc_cur.get("image_grid_thw"))
    dpv, dgrid = _vision_delta(enc_prev, enc_cur)
    print("delta", None if dpv is None else dpv.shape, None if dgrid is None else dgrid, None if dgrid is None else dgrid.shape)


if __name__ == "__main__":
    main()
