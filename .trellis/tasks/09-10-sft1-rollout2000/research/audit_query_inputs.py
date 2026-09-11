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
from nimloth.training.sft.stage2.config import QueryAlignmentConfig
from nimloth.training.sft.stage2.data import QueryAlignmentCollator

DATASETS = {}
COLLATOR = None
QUERY_COUNT = None


def audit_one(item):
    split, row = item
    encoded = COLLATOR([DATASETS[split][row]])
    length = int(encoded["attention_mask"][0].sum().item())
    if length >= 20000:
        raise ValueError(f"input at truncation boundary: {split}/{row}: {length}")
    answers = int(encoded["query_positions"].shape[0])
    if encoded["query_positions"].shape != (answers, QUERY_COUNT):
        raise ValueError(f"every query position row must contain {QUERY_COUNT} slots")
    if encoded["dino_target"].shape != (answers, QUERY_COUNT, 1024):
        raise ValueError("real DINO target shape mismatch")
    if encoded["query_batch_indices"].tolist() != [0] * answers:
        raise ValueError("query positions do not belong to the full trajectory")
    if set(encoded["answer_indices"][encoded["answer_indices"] >= 0].tolist()) != set(range(answers)):
        raise ValueError("answer token ownership is incomplete")
    if any(
        not torch.isfinite(value).all()
        for value in encoded.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    ):
        raise ValueError("nonfinite collator tensor")
    return {
        "split": split,
        "row": row,
        "sample_index": row,
        "length": length,
        "answers": answers,
        "supervised_tokens": int((encoded["labels"] != -100).sum()),
    }


def main():
    global COLLATOR, QUERY_COUNT
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
    parser.add_argument("--grid-size", type=int, default=4)
    args = parser.parse_args()
    objective = QueryAlignmentConfig(grid_size=args.grid_size)
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
    QUERY_COUNT = objective.grid_tokens
    added = add_special_tokens(processor.tokenizer, latent_token_count=QUERY_COUNT)
    targets = CachedDINOGridTargets.from_cache_root(
        args.dino_cache_root,
        identity=DINOV2_LARGE_IDENTITY,
        grid_size=objective.grid_size,
    )
    COLLATOR = QueryAlignmentCollator(
        processor=processor,
        max_length=20000,
        query_count=QUERY_COUNT,
        targets=targets,
    )
    jobs, summary = [], {}
    for split, source, records, answers in (
        ("train", args.train_jsonl, 1709, 20212),
        ("val", args.val_jsonl, 193, 2152),
    ):
        dataset = NimlothVLSFTDataset(source, processor=processor)
        if len(dataset) != records:
            raise ValueError(f"unexpected {split} record/answer counts")
        DATASETS[split] = dataset
        jobs.extend((split, row) for row in range(len(dataset)))
        summary[split] = {
            "records": records,
            "answers": answers,
            "jsonl": str(source.resolve()),
            "sha256": file_sha256(source),
            "checked": 0,
            "checked_answers": 0,
            "max_length": 0,
            "max_sample_index": None,
        }
    with mp.get_context("fork").Pool(args.workers) as pool:
        for result in pool.imap_unordered(audit_one, jobs, chunksize=1):
            split_summary = summary[result["split"]]
            split_summary["checked"] += 1
            split_summary["checked_answers"] += result["answers"]
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
            or value["checked_answers"] != value["answers"]
            or file_sha256(value["jsonl"]) != value["sha256"]
        ):
            raise ValueError("incomplete audit or changed source JSONL")
    report = {
        "status": "passed",
        "scope": "all complete trajectories and all answer-aligned query states",
        "model": str(args.model.resolve()),
        "query_count": QUERY_COUNT,
        "grid_size": objective.grid_size,
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
