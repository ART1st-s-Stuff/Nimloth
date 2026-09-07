#!/usr/bin/env python3
"""Fail-closed checks for the stage1 -> stage2 -> direct-eval experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_DATA = {
    "train": {"records": 613, "assistant_turns": 7309},
    "val": {"records": 355, "assistant_turns": 6054},
}
EXPECTED_HELD_OUT = {
    "base": "6b575621a6b15e90e1040dd86d661a5e1ee70134f42fd7f3d61706347449c55a",
    "common_sense": "3e7d2cb4246b6e2edaeaabd318dba93e4dbbff114c8368ed0c862e64f417afcf",
}
EXPECTED_DINO_FINGERPRINT = "b50d261e2b533f3e"
TRAIN_WORLD_SIZE = 4


def _run_identity(commit: str, world_size: int) -> dict[str, Any]:
    return {
        "schema": "sft_stage1_stage2_preemptible_run_v1",
        "commit": commit,
        "world_size": world_size,
        "stages": ["format", "query", "direct_eval"],
    }


def init_run_identity(args: argparse.Namespace) -> int:
    root = Path(args.run_root)
    identity = root / "run_identity.json"
    if identity.exists():
        raise FileExistsError(f"run identity already exists: {identity}")
    _atomic_json(identity, _run_identity(args.commit, args.world_size))
    return 0


def validate_run_identity(args: argparse.Namespace) -> int:
    actual = json.loads(
        (Path(args.run_root) / "run_identity.json").read_text(encoding="utf-8")
    )
    expected = _run_identity(args.commit, args.world_size)
    if actual != expected:
        raise ValueError("existing run identity does not match commit/world/stages")
    return 0


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _message_image_count(content: Any) -> int:
    if isinstance(content, str):
        return content.count("<image>")
    if isinstance(content, list):
        return sum(
            int(isinstance(part, dict) and part.get("type") == "image")
            for part in content
        )
    raise ValueError("message content must be text or structured content")


def inspect_jsonl(path: Path, split: str) -> tuple[dict[str, Any], set[str]]:
    expected = EXPECTED_DATA[split]
    record_ids: set[str] = set()
    image_paths: set[str] = set()
    assistant_turns = 0
    image_references = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{path}:{line_number}: missing record id")
            if record_id in record_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate record id {record_id}"
                )
            record_ids.add(record_id)
            messages = record.get("messages")
            paths = record.get("image_paths")
            if not isinstance(messages, list) or not isinstance(paths, list):
                raise TypeError(f"{path}:{line_number}: invalid messages/image_paths")
            record_assistants = sum(
                int(isinstance(message, dict) and message.get("role") == "assistant")
                for message in messages
            )
            record_images = sum(
                _message_image_count(message.get("content"))
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            )
            if record_assistants < 1 or record_images != len(paths):
                raise ValueError(
                    f"{path}:{line_number}: answer/image alignment mismatch "
                    f"assistants={record_assistants} placeholders={record_images} paths={len(paths)}"
                )
            assistant_turns += record_assistants
            image_references += len(paths)
            for raw_path in paths:
                resolved = Path(raw_path).expanduser().resolve()
                if not resolved.is_file():
                    raise FileNotFoundError(f"recorded image is missing: {resolved}")
                image_paths.add(str(resolved))
    if len(record_ids) != expected["records"]:
        raise ValueError(
            f"{split} record count mismatch: {len(record_ids)} != {expected['records']}"
        )
    if assistant_turns != expected["assistant_turns"]:
        raise ValueError(
            f"{split} prefix count mismatch: {assistant_turns} != "
            f"{expected['assistant_turns']}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "records": len(record_ids),
        "assistant_turns": assistant_turns,
        "image_references": image_references,
        "unique_images": len(image_paths),
    }, image_paths


def preflight_inputs(args: argparse.Namespace) -> int:
    from nimloth.backbone.dino_grid import (
        DINOV2_LARGE_IDENTITY,
        CachedDINOGridTargets,
    )

    output_usage = shutil.disk_usage(args.output_root)
    required_free_bytes = 100 * 1024**3
    if output_usage.free < required_free_bytes:
        raise OSError(
            f"output filesystem has less than 100 GiB free: {output_usage.free} bytes"
        )
    source = Path(args.source_checkpoint)
    for name in ("config.json", "tokenizer_config.json", "preprocessor_config.json"):
        if not (source / name).is_file():
            raise FileNotFoundError(f"source checkpoint missing {name}: {source}")
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            f"source checkpoint has an invalid safetensors index: {index_path}"
        )
    shard_names = sorted(set(weight_map.values()))
    if any(not (source / name).is_file() for name in shard_names):
        raise FileNotFoundError("source checkpoint is missing an indexed model shard")
    train, train_images = inspect_jsonl(Path(args.train_jsonl), "train")
    val, val_images = inspect_jsonl(Path(args.val_jsonl), "val")
    targets = CachedDINOGridTargets.from_cache_root(
        args.dino_cache_root,
        identity=DINOV2_LARGE_IDENTITY,
        grid_size=4,
    )
    required_images = train_images | val_images
    missing = sorted(required_images - set(targets.path_to_feature))
    if missing:
        raise ValueError(
            f"DINO cache does not cover {len(missing)} required images; first={missing[0]}"
        )
    if targets.cache_fingerprint != EXPECTED_DINO_FINGERPRINT:
        raise ValueError(
            "DINO cache fingerprint mismatch: "
            f"{targets.cache_fingerprint} != {EXPECTED_DINO_FINGERPRINT}"
        )
    # The cache loader validates every manifest and shard shape. Inspect every
    # feature actually consumed by these JSONLs as well; NaN/Inf must fail here,
    # before any GPU training starts.
    import torch

    ordered_images = sorted(required_images)
    for start in range(0, len(ordered_images), 256):
        features = targets.load(
            ordered_images[start : start + 256], device=torch.device("cpu")
        )
        if not torch.isfinite(features).all():
            raise ValueError("DINO cache contains non-finite required features")
    assets: dict[str, Any] = {}
    for name, expected_hash in EXPECTED_HELD_OUT.items():
        path = Path(args.asset_root) / f"{name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual_hash = _sha256(path)
        if actual_hash != expected_hash or len(payload.get("tasks", [])) != 60:
            raise ValueError(f"held-out asset identity mismatch: {path}")
        assets[name] = {"path": str(path.resolve()), "sha256": actual_hash, "tasks": 60}
    payload = {
        "schema": "sft_stage1_stage2_eval_input_preflight_v1",
        "output_filesystem": {
            "root": str(Path(args.output_root).resolve()),
            "free_bytes": output_usage.free,
            "required_free_bytes": required_free_bytes,
        },
        "source_checkpoint": str(source.resolve()),
        "source_config_sha256": _sha256(source / "config.json"),
        "source_weight_index_sha256": _sha256(index_path),
        "source_weight_index_keys": len(weight_map),
        "source_weight_shards": {
            name: {
                "bytes": (source / name).stat().st_size,
                "sha256": _sha256(source / name),
            }
            for name in shard_names
        },
        "train": train,
        "val": val,
        "dino_cache_root": str(Path(args.dino_cache_root).resolve()),
        "dino_cache_fingerprint": targets.cache_fingerprint,
        "dino_required_unique_images": len(required_images),
        "dino_missing_images": 0,
        "dino_required_features_finite": True,
        "held_out_assets": assets,
    }
    _atomic_json(Path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def validate_merged(args: argparse.Namespace) -> int:
    import torch

    adapter = Path(args.adapter_dir)
    merged = Path(args.merged_dir)
    state_path = adapter / "training_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"adapter training state missing: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("training_stage") != args.stage or int(state.get("epoch", -1)) != 1:
        raise ValueError("adapter is not the completed requested one-epoch stage")
    if not state.get("lora"):
        raise ValueError("experiment requires a LoRA adapter checkpoint")
    config_path = merged / "config.json"
    if not config_path.is_file() or not any(merged.glob("*.safetensors")):
        raise FileNotFoundError(f"merged checkpoint is incomplete: {merged}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_count = 1 if args.stage == "format" else 16
    expected_mode = "generate" if args.stage == "format" else "inject"
    if (
        config.get("nimloth_training_stage") != args.stage
        or int(config.get("nimloth_latent_token_count", -1)) != expected_count
        or config.get("nimloth_latent_query_mode") != expected_mode
    ):
        raise ValueError("merged checkpoint stage/query protocol mismatch")
    required_query = ("slot_projector.pt", "grid_state_config.json")
    if args.stage == "query" and any(
        not (merged / name).is_file() for name in required_query
    ):
        raise FileNotFoundError(
            "query merged checkpoint is missing projector artifacts"
        )
    payload = {
        "stage": args.stage,
        "adapter_dir": str(adapter.resolve()),
        "merged_dir": str(merged.resolve()),
        "epoch": 1,
        "global_step": int(state.get("step", -1)),
        "latent_token_count": expected_count,
        "latent_query_mode": expected_mode,
        "config_sha256": _sha256(config_path),
        "query_artifacts": {
            name: _sha256(merged / name)
            for name in required_query
            if (merged / name).is_file()
        },
    }
    _atomic_json(Path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def validate_stage_checkpoint(args: argparse.Namespace) -> int:
    import torch

    checkpoint = Path(args.checkpoint)
    if not (checkpoint / "COMMITTED").is_file():
        raise FileNotFoundError("stage checkpoint has no durable completion marker")
    state = torch.load(
        checkpoint / "training_state.pt", map_location="cpu", weights_only=False
    )
    if (
        state.get("training_stage") != args.stage
        or int(state.get("epoch", -1)) != 1
        or int(state.get("world_size", -1)) != TRAIN_WORLD_SIZE
        or state.get("identity", {}).get("world_size") != TRAIN_WORLD_SIZE
    ):
        raise ValueError("stage checkpoint identity/epoch/world mismatch")
    return 0


def _argv_value(argv: list[str], flag: str) -> str:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"evaluation contract missing {flag}") from error


def _load_validated_trajectories(path: Path):
    """Use the production schema loader and rollout validator for final evidence."""
    from nimloth.rollout.storage import load_trajectories
    from nimloth.training.sft.evaluation.rollout import validate_trajectories

    trajectories = load_trajectories(path)
    validate_trajectories(trajectories, expected_count=120)
    return trajectories


def _validate_eval_arm(path: Path, arm: str) -> dict[str, Any]:
    summary = json.loads((path / "rollout_summary.json").read_text(encoding="utf-8"))
    contract = json.loads(
        (path / "evaluation_contract.json").read_text(encoding="utf-8")
    )
    if summary.get("status") != "ALL_OK" or summary.get("num_trajectories") != 120:
        raise ValueError(f"{arm} evaluation did not complete 120 episodes")
    if summary.get("eval_sets") != ["base", "common_sense"]:
        raise ValueError(f"{arm} evaluation used the wrong held-out datasets")
    argv = contract.get("rollout_argv", [])
    required = {
        "--backend": "vllm",
        "--num-episodes": "120",
        "--split": "test",
        "--seed-offset": "1",
        "--max-steps": "20",
        "--temperature": "0.0",
        "--top-p": "1.0",
        "--max-response-tokens": "512",
        "--tensor-parallel-size": "1",
    }
    if contract.get("evaluation") != "sft_eval_direct_v1":
        raise ValueError(f"{arm} is not a direct evaluation")
    for flag, value in required.items():
        if _argv_value(argv, flag) != value:
            raise ValueError(f"{arm} evaluation contract mismatch for {flag}")
    if "--seed-per-eval-set" not in argv or "--planner-enabled" in argv:
        raise ValueError(f"{arm} evaluation seed/planner contract mismatch")
    trajectories = _load_validated_trajectories(path / "trajectories.jsonl")
    if len({row.record_id for row in trajectories}) != 120:
        raise ValueError(f"{arm} trajectory persistence is incomplete or duplicated")
    metrics = summary.get("metrics", {})
    if set(metrics.get("by_eval_set", {})) != {"base", "common_sense"}:
        raise ValueError(f"{arm} metrics are missing a held-out group")
    return {
        "checkpoint": _argv_value(argv, "--model"),
        "policy_fingerprint": contract.get("policy_fingerprint"),
        "num_trajectories": 120,
        "num_transitions": summary.get("num_transitions"),
        "metrics": metrics,
        "summary": str((path / "rollout_summary.json").resolve()),
    }


def finalize(args: argparse.Namespace) -> int:
    stage1 = _validate_eval_arm(Path(args.stage1_eval), "stage1")
    stage2 = _validate_eval_arm(Path(args.stage2_eval), "stage2")
    payload = {
        "schema": "sft_stage1_stage2_direct_eval_v1",
        "status": "passed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "training": {
            "stage1": json.loads(Path(args.stage1_merge).read_text(encoding="utf-8")),
            "stage2": json.loads(Path(args.stage2_merge).read_text(encoding="utf-8")),
        },
        "evaluation": {"stage1": stage1, "stage2": stage2},
        "limitations": [
            "single one-epoch experiment from the historical step79 initialization",
            "results assess this code path and are not approved as a later-training seed",
        ],
    }
    _atomic_json(Path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def record_exit(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if output.exists():
        if args.exit_code == 0:
            return 0
        payload = json.loads(output.read_text(encoding="utf-8"))
        if args.exit_code == 75:
            payload.update(
                {
                    "status": "preempted",
                    "exit_code": args.exit_code,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            _atomic_json(output, payload)
            return 0
        previous_status = payload.get("status")
        cleanup_failed = args.exit_code in {91, 92} or previous_status == "passed"
        payload.update(
            {
                "status": "cleanup_failed" if cleanup_failed else "failed",
                "pipeline_status_before_cleanup": previous_status,
                "exit_code": args.exit_code,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _atomic_json(output, payload)
        return 0
    status = "preempted" if args.exit_code == 75 else "failed"
    _atomic_json(
        output,
        {
            "schema": "sft_stage1_stage2_direct_eval_v1",
            "status": status,
            "exit_code": args.exit_code,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return 0


def validate_rank_rows(rows: list[dict[str, Any]], visible_count: int) -> None:
    if len(rows) != TRAIN_WORLD_SIZE or visible_count != TRAIN_WORLD_SIZE:
        raise RuntimeError(
            f"expected world{TRAIN_WORLD_SIZE}/{TRAIN_WORLD_SIZE} visible GPUs, "
            f"got {len(rows)}/{visible_count}"
        )
    expected = list(range(TRAIN_WORLD_SIZE))
    if sorted(item["rank"] for item in rows) != expected:
        raise RuntimeError("global ranks are incomplete or duplicated")
    if sorted(item["local_rank"] for item in rows) != expected:
        raise RuntimeError(
            f"local ranks do not map one-to-one onto {TRAIN_WORLD_SIZE} GPUs"
        )
    if len({item["host"] for item in rows}) != 1:
        raise RuntimeError("rank preflight escaped the one-node allocation")


def rank_map(args: argparse.Namespace) -> int:
    import socket

    import torch
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != TRAIN_WORLD_SIZE or torch.cuda.device_count() != TRAIN_WORLD_SIZE:
        raise RuntimeError(f"expected world{TRAIN_WORLD_SIZE}/four visible GPUs")
    torch.cuda.set_device(local_rank)
    row = {
        "rank": rank,
        "local_rank": local_rank,
        "device": torch.cuda.current_device(),
        "name": torch.cuda.get_device_name(local_rank),
        "host": socket.gethostname(),
    }
    rows: list[Any] = [None] * world
    dist.all_gather_object(rows, row)
    if rank == 0:
        validate_rank_rows(rows, torch.cuda.device_count())
        _atomic_json(Path(args.output), {"world_size": world, "ranks": rows})
    dist.barrier()
    dist.destroy_process_group()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, function in (
        ("init-run-identity", init_run_identity),
        ("validate-run-identity", validate_run_identity),
    ):
        identity = subparsers.add_parser(name)
        identity.add_argument("--run-root", required=True)
        identity.add_argument("--commit", required=True)
        identity.add_argument("--world-size", type=int, required=True)
        identity.set_defaults(func=function)
    preflight = subparsers.add_parser("preflight-inputs")
    preflight.add_argument("--source-checkpoint", required=True)
    preflight.add_argument("--train-jsonl", required=True)
    preflight.add_argument("--val-jsonl", required=True)
    preflight.add_argument("--dino-cache-root", required=True)
    preflight.add_argument("--asset-root", required=True)
    preflight.add_argument("--output-root", required=True)
    preflight.add_argument("--output", required=True)
    preflight.set_defaults(func=preflight_inputs)
    merged = subparsers.add_parser("validate-merged")
    merged.add_argument("--stage", choices=("format", "query"), required=True)
    merged.add_argument("--adapter-dir", required=True)
    merged.add_argument("--merged-dir", required=True)
    merged.add_argument("--output", required=True)
    merged.set_defaults(func=validate_merged)
    stage_checkpoint = subparsers.add_parser("validate-stage-checkpoint")
    stage_checkpoint.add_argument("--stage", choices=("format", "query"), required=True)
    stage_checkpoint.add_argument("--checkpoint", required=True)
    stage_checkpoint.set_defaults(func=validate_stage_checkpoint)
    final = subparsers.add_parser("finalize")
    final.add_argument("--stage1-eval", required=True)
    final.add_argument("--stage2-eval", required=True)
    final.add_argument("--stage1-merge", required=True)
    final.add_argument("--stage2-merge", required=True)
    final.add_argument("--output", required=True)
    final.set_defaults(func=finalize)
    exit_parser = subparsers.add_parser("record-exit")
    exit_parser.add_argument("--exit-code", type=int, required=True)
    exit_parser.add_argument("--output", required=True)
    exit_parser.set_defaults(func=record_exit)
    ranks = subparsers.add_parser("rank-map")
    ranks.add_argument("--output", required=True)
    ranks.set_defaults(func=rank_map)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
