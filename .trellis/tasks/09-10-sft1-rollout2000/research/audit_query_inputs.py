"""Audit full-history Stage 2 inputs with the real processor/collator on CPU."""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import torch
from transformers import AutoProcessor

from nimloth.backbone.dino_grid import (
    DINOV2_LARGE_IDENTITY,
    CachedDINOGridTargets,
    file_sha256,
)
from nimloth.latent import add_special_tokens
from nimloth.training.sft.stage1.data import NimlothVLSFTDataset
from nimloth.training.sft.stage2.data import AnswerPrefixDataset, QueryAlignmentCollator

DATASETS = {}
COLLATOR = None


def audit_one(item):
    split, row, sample_index = item
    encoded = COLLATOR([DATASETS[split][sample_index]])
    length = int(encoded["attention_mask"][0].sum().item())
    if length >= 20000:
        raise ValueError(f"input at truncation boundary: {split}/{row}: {length}")
    if encoded["query_positions"].shape != (1, 16):
        raise ValueError("query positions must contain exactly sixteen slots")
    if encoded["dino_target"].shape != (1, 16, 1024):
        raise ValueError("real DINO target shape mismatch")
    if any(
        not torch.isfinite(value).all()
        for value in encoded.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    ):
        raise ValueError("nonfinite collator tensor")
    return {
        "split": split,
        "row": row,
        "sample_index": sample_index,
        "length": length,
        "supervised_tokens": int((encoded["labels"] != -100).sum()),
    }


def main():
    global COLLATOR
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (
        "model",
        "train-jsonl",
        "val-jsonl",
        "dino-cache-root",
        "output-json",
    ):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between one and eight")
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    started = time.time()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 100352
    added = add_special_tokens(processor.tokenizer, latent_token_count=16)
    targets = CachedDINOGridTargets.from_cache_root(
        args.dino_cache_root, identity=DINOV2_LARGE_IDENTITY, grid_size=4
    )
    COLLATOR = QueryAlignmentCollator(
        processor=processor,
        max_length=20000,
        query_count=16,
        targets=targets,
        last_answer_only=True,
    )
    jobs, summary = [], {}
    for split, source, records, answers in (
        ("train", args.train_jsonl, 1709, 20212),
        ("val", args.val_jsonl, 193, 2152),
    ):
        dataset = NimlothVLSFTDataset(source, processor=processor)
        prefixes = AnswerPrefixDataset(dataset)
        if len(dataset) != records or len(prefixes) != answers:
            raise ValueError(f"unexpected {split} record/answer counts")
        DATASETS[split] = prefixes
        last_indices = {}
        for index, (row, _) in enumerate(prefixes.index):
            last_indices[row] = index
        jobs.extend((split, row, index) for row, index in last_indices.items())
        summary[split] = {
            "records": records,
            "answers": answers,
            "jsonl": str(source.resolve()),
            "sha256": file_sha256(source),
            "checked": 0,
            "max_length": 0,
            "max_sample_index": None,
        }
    with mp.get_context("fork").Pool(args.workers) as pool:
        for result in pool.imap_unordered(audit_one, jobs, chunksize=1):
            split_summary = summary[result["split"]]
            split_summary["checked"] += 1
            if result["length"] > split_summary["max_length"]:
                split_summary.update(
                    max_length=result["length"],
                    max_sample_index=result["sample_index"],
                    max_row=result["row"],
                )
            done = sum(value["checked"] for value in summary.values())
            if done % 25 == 0 or done == len(jobs):
                print(
                    json.dumps(
                        {
                            "checked": done,
                            "total": len(jobs),
                            "max_lengths": {
                                key: value["max_length"]
                                for key, value in summary.items()
                            },
                        }
                    ),
                    flush=True,
                )
    for value in summary.values():
        if (
            value["checked"] != value["records"]
            or file_sha256(value["jsonl"]) != value["sha256"]
        ):
            raise ValueError("incomplete audit or changed source JSONL")
    report = {
        "status": "passed",
        "scope": "all trajectories final-answer full-history prefixes",
        "model": str(args.model.resolve()),
        "query_count": 16,
        "max_length": 20000,
        "min_pixels": 3136,
        "max_pixels": 100352,
        "added_query_tokens": added,
        "dino_cache_fingerprint": targets.cache_fingerprint,
        "splits": summary,
        "elapsed_seconds": time.time() - started,
        "train_max_sample_index": summary["train"]["max_sample_index"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
