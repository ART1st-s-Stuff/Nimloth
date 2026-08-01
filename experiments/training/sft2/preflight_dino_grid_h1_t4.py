#!/usr/bin/env python3
"""Fail-closed preflight for the formal H=1/T=4 DINO-grid SFT2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
import wandb
import yaml
from transformers import AutoProcessor

from nimloth.backbone import CachedDINOGridTargets, DINOV2_LARGE_IDENTITY
from nimloth.latent import add_special_tokens
from nimloth.rollout.transitions import TransitionJsonlDataset
from nimloth.training.sft2.algorithm import SFT2_VALUE_OBJECTIVE
from nimloth.training.sft2.data.samplers import FutureRolloutBatchSampler
from nimloth.util.cache import (
    COMPACT_CACHE_FORMAT,
    DEFAULT_MIN_PIXELS,
    TRANSITION_EXPANSION_VERSION,
    CachedTransitionDataset,
    CompactCachedTransitionCollator,
    cache_fingerprint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--val-sha256", required=True)
    parser.add_argument("--preprocess-cache", type=Path, required=True)
    parser.add_argument("--dino-cache", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--partition", default="normal")
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--min-free-gib", type=float, default=80.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def existing_parent(path: Path) -> Path:
    current = path.resolve()
    while not current.exists():
        current = current.parent
    return current


def validate_config(
    config_path: Path,
    *,
    grad_accum: int,
    epochs: int,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train = config["train"]
    loss = config["loss"]
    tuning = config["tuning"]
    latent = config["latent"]
    assert config["objective"]["name"] == "dino_grid"
    assert train["epochs"] == epochs
    assert train["batch_size"] == 1
    assert train["history_size"] == 1
    assert train["prediction_horizon"] == 4
    assert train["batch_mode"] == "trajectory_online_cache"
    assert train["require_prebuilt_cache"] is True
    assert train["train_wm_predictor"] is True
    assert train["checkpoint_interval_minutes"] == 20.0
    assert tuning["llm_tune"] == "freeze"
    assert tuning["vision_tune"] == "full"
    assert latent["query_tune"] == "freeze"
    assert loss["value_gamma"] == 1.0
    assert loss["lambda_value"] == 1.0
    assert "rank" not in json.dumps(loss).lower()
    return {
        "config_grad_accum": int(train["grad_accum"]),
        "launch_grad_accum": grad_accum,
        "epochs": epochs,
        "batch_size_per_rank": 1,
        "history_size": 1,
        "prediction_horizon": 4,
        "value_objective": SFT2_VALUE_OBJECTIVE,
        "checkpoint_interval_minutes": float(
            train["checkpoint_interval_minutes"]
        ),
    }


def validate_split(
    *,
    jsonl: Path,
    cache_dir: Path,
    processor: Any,
    model: Path,
) -> tuple[dict[str, Any], list[Any]]:
    samples = TransitionJsonlDataset(jsonl, value_gamma=1.0).samples
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == COMPACT_CACHE_FORMAT
    assert manifest["count"] == len(samples)
    assert manifest["value_gamma"] == 1.0
    assert manifest["latent_token_count"] == 16
    assert manifest["latent_query_mode"] == "inject"
    assert manifest["image_dtype"] == "bfloat16"
    assert manifest["transition_expansion_version"] == TRANSITION_EXPANSION_VERSION
    assert manifest["base_fingerprint"] == cache_fingerprint(
        jsonl,
        max_length=12000,
        max_pixels=100352,
        min_pixels=DEFAULT_MIN_PIXELS,
        vocab_size=len(processor.tokenizer),
        value_gamma=1.0,
        latent_token_count=16,
        mask_latent_query_labels=True,
        cache_format=COMPACT_CACHE_FORMAT,
        image_dtype="bfloat16",
        processor_source=str(model.resolve()),
        transition_expansion_version=TRANSITION_EXPANSION_VERSION,
    )
    assert not (cache_dir / "build_state.json").exists()
    image_paths = sorted((cache_dir / "images").glob("shard_*.pt"))
    transition_paths = sorted((cache_dir / "transitions").glob("shard_*.pt"))
    assert len(image_paths) == manifest["image_shards"]
    assert len(transition_paths) == manifest["transition_shards"]
    assert all(path.stat().st_size > 0 for path in image_paths + transition_paths)
    assert manifest["transition_shards"] == math.ceil(len(samples) / 256)

    dataset = CachedTransitionDataset(cache_dir, samples, max_open_shards=2)
    collator = CompactCachedTransitionCollator(cache_dir, max_open_shards=2)
    sampler = FutureRolloutBatchSampler(
        samples,
        prediction_horizon=4,
        batch_size=1,
        num_replicas=1,
        rank=0,
        shuffle=False,
        pad_to_equal_batches=False,
    )
    selected_ordinals = {0, sampler.window_count // 2, sampler.window_count - 1}
    selected: list[dict[str, Any]] = []
    loaded_windows = 0
    for ordinal, indices in enumerate(sampler):
        rows = [dataset[index] for index in indices]
        steps = [int(row["step_index"]) for row in rows]
        actions = [int(row["action_index"]) for row in rows]
        assert len(rows) == 4
        assert len({row["record_id"] for row in rows}) == 1
        assert all(right == left + 1 for left, right in zip(steps, steps[1:]))
        assert actions == [
            samples[index.sample_index].action_index for index in indices
        ]
        assert all(
            math.isfinite(float(row["action_value_target"])) for row in rows
        )
        if ordinal in selected_ordinals:
            batch = collator(rows)
            pixel_rows = [batch["current_enc_rows"][0], *batch["next_enc_rows"]]
            assert all(
                row is not None and "pixel_values" in row for row in pixel_rows
            )
            assert all(
                torch.isfinite(row["pixel_values"]).all()
                for row in pixel_rows
                if row is not None
            )
            selected.append(
                {
                    "ordinal": ordinal,
                    "record_id": rows[0]["record_id"],
                    "steps": steps,
                    "actions": actions,
                    "pixel_dtype": str(
                        batch["current_enc_rows"][0]["pixel_values"].dtype
                    ),
                }
            )
        loaded_windows += 1
    assert loaded_windows == sampler.window_count
    return (
        {
            "transitions": len(samples),
            "h1_t4_windows": sampler.window_count,
            "loaded_windows": loaded_windows,
            "unique_images": manifest["unique_images"],
            "image_shards": manifest["image_shards"],
            "transition_shards": manifest["transition_shards"],
            "fingerprint": manifest["fingerprint"],
            "base_fingerprint": manifest["base_fingerprint"],
            "selected_materializations": selected,
        },
        samples,
    )


def distributed_schedule(
    samples: list[Any],
    *,
    world_size: int,
    grad_accum: int,
    epochs: int,
) -> dict[str, Any]:
    samplers = [
        FutureRolloutBatchSampler(
            samples,
            prediction_horizon=4,
            batch_size=1,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=0,
            pad_to_equal_batches=True,
        )
        for rank in range(world_size)
    ]
    assert len({len(sampler) for sampler in samplers}) == 1
    epoch_sigreg: dict[str, Any] = {}
    for epoch in range(1, epochs + 1):
        for sampler in samplers:
            sampler.set_epoch(epoch)
        global_valid = [
            sum(rank_counts)
            for rank_counts in zip(
                *(sampler.current_steps_per_batch for sampler in samplers),
                strict=True,
            )
        ]
        assert sum(global_valid) == samplers[0].window_count
        assert min(global_valid) > 1
        epoch_sigreg[str(epoch)] = {
            "microbatches": len(global_valid),
            "valid_min": min(global_valid),
            "valid_max": max(global_valid),
            "valid_sum": sum(global_valid),
            "padding_slots": len(global_valid) * world_size - sum(global_valid),
        }
    microbatches = len(samplers[0])
    optimizer_steps = math.ceil(microbatches / grad_accum)
    return {
        "world_size": world_size,
        "microbatches_per_epoch": microbatches,
        "optimizer_steps_per_epoch": optimizer_steps,
        "total_optimizer_steps": optimizer_steps * epochs,
        "global_sigreg": epoch_sigreg,
    }


def main() -> None:
    args = parse_args()
    assert args.partition in {"normal", "preempt"}
    assert args.nodes > 0
    assert args.gpus_per_node > 0
    assert args.world_size == args.nodes * args.gpus_per_node
    assert args.grad_accum == 4
    assert len(args.wandb_run_id) == 8
    assert git_output(args.repo, "rev-parse", "HEAD") == args.expected_commit
    assert not git_output(
        args.repo,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    assert args.model.joinpath("config.json").is_file()
    assert sha256(args.train_jsonl) == args.train_sha256
    assert sha256(args.val_jsonl) == args.val_sha256
    assert args.preprocess_cache.joinpath("cache_done.flag").is_file()
    assert not args.run_output.exists() or not any(args.run_output.iterdir())

    free_bytes = shutil.disk_usage(existing_parent(args.run_output)).free
    assert free_bytes >= args.min_free_gib * 1024**3, (
        f"insufficient free space: {free_bytes / 1024**3:.1f} GiB "
        f"< {args.min_free_gib:.1f} GiB"
    )
    config = validate_config(
        args.config,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    assert add_special_tokens(processor.tokenizer, latent_token_count=16) == 0
    processor.image_processor.min_pixels = DEFAULT_MIN_PIXELS
    processor.image_processor.max_pixels = 100352
    train_result, train_samples = validate_split(
        jsonl=args.train_jsonl,
        cache_dir=args.preprocess_cache / "train",
        processor=processor,
        model=args.model,
    )
    val_result, val_samples = validate_split(
        jsonl=args.val_jsonl,
        cache_dir=args.preprocess_cache / "val",
        processor=processor,
        model=args.model,
    )

    dino = CachedDINOGridTargets.from_cache_root(
        args.dino_cache,
        identity=DINOV2_LARGE_IDENTITY,
        grid_size=4,
    )
    dino_coverage = {}
    for split, samples in (("train", train_samples), ("val", val_samples)):
        required = {
            str(Path(sample.next_image_path).resolve()) for sample in samples
        }
        missing = required - dino.path_to_feature.keys()
        assert not missing, f"{split} DINO cache misses {len(missing)} images"
        dino_coverage[split] = {
            "required_images": len(required),
            "missing": 0,
        }

    api = wandb.Api()
    project_path = f"{args.wandb_entity}/{args.wandb_project}"
    assert not list(
        api.runs(project_path, filters={"display_name": args.wandb_run_name})
    )
    assert not list(api.runs(project_path, filters={"name": args.wandb_run_id}))

    result = {
        "status": "passed",
        "commit": args.expected_commit,
        "config": config,
        "objective": {
            "decision_state_executed_action_mc": True,
            "rank_loss": False,
            "recorded_action_sequence": True,
            "value_gamma": 1.0,
        },
        "inputs": {
            "config": str(args.config.resolve()),
            "model_init": str(args.model.resolve()),
            "train_jsonl": str(args.train_jsonl.resolve()),
            "val_jsonl": str(args.val_jsonl.resolve()),
            "preprocess_cache": str(args.preprocess_cache.resolve()),
            "preprocess_cache_access": "read_only_reuse",
            "dino_cache": str(args.dino_cache.resolve()),
        },
        "modules": {
            "trainable": [
                "qwen_vision",
                "state_projector",
                "wm_predictor",
                "value_head",
            ],
            "frozen": ["qwen_llm", "dino_teacher_cache", "latent_query"],
        },
        "data": {
            "train": train_result,
            "val": val_result,
            "train_sha256": args.train_sha256,
            "val_sha256": args.val_sha256,
        },
        "dino": {
            "cache_fingerprint": dino.cache_fingerprint,
            "coverage": dino_coverage,
        },
        "training": {
            **distributed_schedule(
                train_samples,
                world_size=args.world_size,
                grad_accum=args.grad_accum,
                epochs=args.epochs,
            ),
            "fresh_optimizer": True,
            "resume": False,
            "checkpoint_interval_minutes": config[
                "checkpoint_interval_minutes"
            ],
        },
        "topology": {
            "partition": args.partition,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "gpu_type": "H800",
            "local_ranks": args.gpus_per_node,
            "world_size": args.world_size,
            "grad_accum": args.grad_accum,
        },
        "output": {
            "run_output": str(args.run_output),
            "free_gib": free_bytes / 1024**3,
        },
        "wandb": {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "run_name": args.wandb_run_name,
            "requested_run_id": args.wandb_run_id,
            "fresh": True,
        },
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.result_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
