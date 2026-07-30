"""Cache DINO-grid states from a frozen SFT2 checkpoint.

The cache is built from the exact compact Qwen preprocessing cache used by
SFT2.  Each distributed rank owns a contiguous sample range and writes small
atomic shards, so a preempted encoding pass can resume without recomputing
completed states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.backbone import backbone_hidden_size
from nimloth.backbone.qwen25vl.input import Qwen25VLInputBuilder
from nimloth.backbone.qwen25vl.latent import extract_qwen_latents
from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.rollout.transitions import TransitionJsonlDataset, TransitionSample
from nimloth.training.sft2.data.factory import _verify_cache_manifest
from nimloth.util.cache import CachedTransitionDataset, CompactCachedTransitionCollator
from nimloth.util.distributed import cleanup_dist, setup_dist
from nimloth.wm.grid import GridPredictorConfig, SharedSlotProjector


CACHE_VERSION = "id56_dino_grid_state_cache_v1"
GRID_SHAPE = (16, 1024)


def contiguous_rank_bounds(total: int, rank: int, world_size: int) -> tuple[int, int]:
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return total * rank // world_size, total * (rank + 1) // world_size


def rank_shard_specs(
    total: int,
    *,
    rank: int,
    world_size: int,
    shard_size: int,
) -> list[tuple[int, int, int]]:
    """Return ``(local_shard_index, global_start, global_end)`` specs."""

    if shard_size < 1:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    start, end = contiguous_rank_bounds(total, rank, world_size)
    return [
        (local, offset, min(offset + shard_size, end))
        for local, offset in enumerate(range(start, end, shard_size))
    ]


def _path_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    required = [
        checkpoint / "config.json",
        checkpoint / "model.safetensors.index.json",
        checkpoint / "state_proj.pt",
        checkpoint / "wm_predictor" / "config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete SFT2 checkpoint: {missing}")
    model_index = json.loads(required[1].read_text(encoding="utf-8"))
    shard_names = sorted(set(model_index.get("weight_map", {}).values()))
    shard_paths = [checkpoint / name for name in shard_names]
    absent_shards = [str(path) for path in shard_paths if not path.is_file()]
    if absent_shards:
        raise FileNotFoundError(f"missing SFT2 model shards: {absent_shards}")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "files": [_path_identity(path) for path in [*required, *shard_paths]],
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _contract_fingerprint(contract: dict[str, Any]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _load_frozen_modules(args: argparse.Namespace, device: torch.device):
    processor = AutoProcessor.from_pretrained(
        args.sft2_checkpoint,
        trust_remote_code=True,
    )
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = int(args.max_pixels)
    added = add_special_tokens(
        processor.tokenizer,
        latent_token_count=args.latent_token_count,
    )
    if added:
        raise ValueError(
            f"SFT2 checkpoint tokenizer is missing {added} required latent/action tokens"
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.sft2_checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    if model.get_input_embeddings().num_embeddings != len(processor.tokenizer):
        raise ValueError(
            "SFT2 checkpoint model/tokenizer vocabulary mismatch: "
            f"model={model.get_input_embeddings().num_embeddings}, "
            f"tokenizer={len(processor.tokenizer)}"
        )
    model.to(device).eval().requires_grad_(False)

    raw_grid = json.loads(
        (args.sft2_checkpoint / "wm_predictor" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    grid_config = GridPredictorConfig(**raw_grid)
    state_dict = torch.load(
        args.sft2_checkpoint / "state_proj.pt",
        map_location="cpu",
        weights_only=True,
    )
    first = state_dict.get("net.0.weight")
    if first is None or first.ndim != 2:
        raise ValueError("ID56 state projector lacks net.0.weight")
    projector = SharedSlotProjector(
        input_dim=backbone_hidden_size(model.config),
        output_dim=grid_config.emb_dim,
        hidden_dim=int(first.shape[0]),
        grid_tokens=grid_config.grid_tokens,
    ).to(device=device, dtype=first.dtype)
    projector.load_state_dict(state_dict, strict=True)
    projector.eval().requires_grad_(False)
    if (grid_config.grid_tokens, grid_config.emb_dim) != GRID_SHAPE:
        raise ValueError(
            "ID56 grid shape mismatch: "
            f"{(grid_config.grid_tokens, grid_config.emb_dim)} != {GRID_SHAPE}"
        )
    token_ids = special_token_ids(
        processor.tokenizer,
        latent_token_count=args.latent_token_count,
    )
    return processor, token_ids, model, projector


def _preprocess_config(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        require_prebuilt_cache=True,
        preprocess_cache_processor_source=args.preprocess_cache_processor_source,
        model=args.sft2_checkpoint,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        value_gamma=1.0,
        latent_token_count=args.latent_token_count,
        mask_latent_query_labels=True,
        preprocess_cache_image_dtype=args.preprocess_cache_image_dtype,
    )


def _validate_preprocess_cache(
    *,
    args: argparse.Namespace,
    processor,
    samples_by_split: dict[str, list[TransitionSample]],
) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    config = _preprocess_config(args)
    jsonl_by_split = {"train": args.train_jsonl, "val": args.val_jsonl}
    for split, samples in samples_by_split.items():
        cache_dir = args.preprocess_cache_dir / split
        _verify_cache_manifest(
            cache_dir=cache_dir,
            jsonl_path=jsonl_by_split[split],
            expected_count=len(samples),
            allow_prefix_subset=False,
            config=config,
            processor=processor,
        )
        manifests[split] = json.loads(
            (cache_dir / "manifest.json").read_text(encoding="utf-8")
        )
    return manifests


def _cache_row(sample: TransitionSample) -> dict[str, Any]:
    return {
        "id": f"{sample.record_id}:{sample.step_index}",
        "record_id": sample.record_id,
        "step_index": int(sample.step_index),
        "action_index": int(sample.action_index),
        "success": bool(sample.success),
        "current_image_path": str(sample.current_image_path),
        "next_image_path": str(sample.next_image_path),
    }


def _shard_path(
    split_dir: Path,
    *,
    rank: int,
    local_shard_index: int,
) -> Path:
    return split_dir / f"shard_r{rank:03d}_{local_shard_index:06d}.pt"


def _valid_shard(
    path: Path,
    *,
    contract_fingerprint: str,
    start: int,
    end: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        states = payload["state_emb"]
        rows = payload["rows"]
        return bool(
            payload.get("contract_fingerprint") == contract_fingerprint
            and int(payload.get("start", -1)) == start
            and int(payload.get("end", -1)) == end
            and tuple(states.shape) == (end - start, *GRID_SHAPE)
            and states.dtype == torch.float16
            and torch.isfinite(states).all()
            and len(rows) == end - start
        )
    except Exception:
        return False


@torch.no_grad()
def _encode_split(
    *,
    args: argparse.Namespace,
    split: str,
    samples: list[TransitionSample],
    processor,
    token_id_map: dict[str, int],
    model: torch.nn.Module,
    projector: torch.nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
    contract_fingerprint: str,
) -> None:
    split_dir = args.output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    dataset = CachedTransitionDataset(
        args.preprocess_cache_dir / split,
        samples,
        max_open_shards=args.preprocess_cache_shard_lru,
    )
    collator = CompactCachedTransitionCollator(
        args.preprocess_cache_dir / split,
        max_open_shards=args.preprocess_cache_shard_lru,
    )
    input_builder = Qwen25VLInputBuilder(
        processor=processor,
        max_length=args.max_length,
        latent_token_count=args.latent_token_count,
        mask_latent_query_labels=True,
    )
    specs = rank_shard_specs(
        len(samples),
        rank=rank,
        world_size=world_size,
        shard_size=args.state_cache_shard_size,
    )
    for local_shard_index, start, end in specs:
        path = _shard_path(
            split_dir,
            rank=rank,
            local_shard_index=local_shard_index,
        )
        if _valid_shard(
            path,
            contract_fingerprint=contract_fingerprint,
            start=start,
            end=end,
        ):
            continue
        shard_states: list[torch.Tensor] = []
        for batch_start in range(start, end, args.encode_batch_size):
            batch_end = min(batch_start + args.encode_batch_size, end)
            raw = collator([dataset[index] for index in range(batch_start, batch_end)])
            encoding = input_builder.collate_encoded(
                raw["current_enc_rows"],
                include_labels=False,
            ).tensors
            hidden, _loss = extract_qwen_latents(
                model,
                dict(encoding),
                token_id_map,
                device,
                latent_token_count=args.latent_token_count,
            )
            state = projector(hidden).detach().float().cpu()
            if tuple(state.shape[1:]) != GRID_SHAPE:
                raise ValueError(
                    f"encoded ID56 grid shape mismatch: {tuple(state.shape)}"
                )
            if not torch.isfinite(state).all():
                raise ValueError(
                    f"non-finite ID56 grid state in {split} [{batch_start},{batch_end})"
                )
            shard_states.append(state.to(dtype=torch.float16))
        states = torch.cat(shard_states)
        _atomic_torch_save(
            {
                "contract_fingerprint": contract_fingerprint,
                "start": start,
                "end": end,
                "state_emb": states,
                "rows": [_cache_row(sample) for sample in samples[start:end]],
            },
            path,
        )
        print(
            json.dumps(
                {
                    "state_cache": "shard_done",
                    "split": split,
                    "rank": rank,
                    "start": start,
                    "end": end,
                    "path": str(path),
                }
            ),
            flush=True,
        )


def _finalize_split(
    *,
    args: argparse.Namespace,
    split: str,
    samples: list[TransitionSample],
    world_size: int,
    contract_fingerprint: str,
    preprocess_manifest: dict[str, Any],
) -> None:
    split_dir = args.output_dir / split
    shards: list[dict[str, Any]] = []
    total_bytes = 0
    expected_start = 0
    for rank in range(world_size):
        for local_shard_index, start, end in rank_shard_specs(
            len(samples),
            rank=rank,
            world_size=world_size,
            shard_size=args.state_cache_shard_size,
        ):
            if start != expected_start:
                raise ValueError(
                    f"state-cache range gap before {split} index {start}; "
                    f"expected {expected_start}"
                )
            path = _shard_path(
                split_dir,
                rank=rank,
                local_shard_index=local_shard_index,
            )
            if not _valid_shard(
                path,
                contract_fingerprint=contract_fingerprint,
                start=start,
                end=end,
            ):
                raise ValueError(f"invalid or missing completed state shard: {path}")
            shards.append(
                {
                    "file": path.name,
                    "count": end - start,
                    "start": start,
                    "end": end,
                    "rank": rank,
                }
            )
            total_bytes += path.stat().st_size
            expected_start = end
    if expected_start != len(samples):
        raise ValueError(
            f"state-cache coverage mismatch for {split}: {expected_start} != {len(samples)}"
        )
    fingerprint = hashlib.sha256(
        f"{contract_fingerprint}|{split}|{preprocess_manifest['fingerprint']}".encode()
    ).hexdigest()[:16]
    _atomic_json(
        split_dir / "manifest.json",
        {
            "version": CACHE_VERSION,
            "split": split,
            "representation": "dino_grid_state",
            "state_shape": list(GRID_SHAPE),
            "latent_token_count": GRID_SHAPE[0],
            "count": len(samples),
            "cond_dim": math.prod(GRID_SHAPE),
            "state_dtype": "float16",
            "compression": "none",
            "shard_size": args.state_cache_shard_size,
            "shards": shards,
            "fingerprint": fingerprint,
            "contract_fingerprint": contract_fingerprint,
            "preprocess_cache": str((args.preprocess_cache_dir / split).resolve()),
            "preprocess_fingerprint": preprocess_manifest["fingerprint"],
            "jsonl_path": str(
                (args.train_jsonl if split == "train" else args.val_jsonl).resolve()
            ),
            "source_checkpoint": str(args.sft2_checkpoint.resolve()),
            "backbone_weights": "online checkpoint weights; vision_ema.pt not applied",
            "cache_build_world_size": world_size,
            "total_bytes": total_bytes,
        },
    )


def run(args: argparse.Namespace) -> int:
    rank, world_size, _local_rank, device = setup_dist()
    try:
        if args.latent_token_count != GRID_SHAPE[0]:
            raise ValueError(
                f"ID56 DINO-grid cache requires {GRID_SHAPE[0]} latent tokens"
            )
        samples_by_split = {
            "train": TransitionJsonlDataset(
                args.train_jsonl,
                max_records=-1,
                success_only=False,
                value_gamma=1.0,
            ).samples,
            "val": TransitionJsonlDataset(
                args.val_jsonl,
                max_records=-1,
                success_only=False,
                value_gamma=1.0,
            ).samples,
        }
        processor, token_id_map, model, projector = _load_frozen_modules(args, device)
        preprocess_manifests = _validate_preprocess_cache(
            args=args,
            processor=processor,
            samples_by_split=samples_by_split,
        )
        contract = {
            "version": CACHE_VERSION,
            "git_commit": args.git_commit,
            "checkpoint": _checkpoint_identity(args.sft2_checkpoint),
            "train_jsonl": _path_identity(args.train_jsonl),
            "val_jsonl": _path_identity(args.val_jsonl),
            "preprocess_cache_dir": str(args.preprocess_cache_dir.resolve()),
            "preprocess_fingerprints": {
                split: manifest["fingerprint"]
                for split, manifest in preprocess_manifests.items()
            },
            "counts": {
                split: len(samples) for split, samples in samples_by_split.items()
            },
            "world_size": world_size,
            "max_length": args.max_length,
            "max_pixels": args.max_pixels,
            "latent_token_count": args.latent_token_count,
            "attn_implementation": args.attn_implementation,
            "encode_batch_size": args.encode_batch_size,
            "state_cache_shard_size": args.state_cache_shard_size,
            "state_shape": list(GRID_SHAPE),
            "dataset_split": "train JSONL for CFM optimization; disjoint val JSONL for model selection only",
            "trainable_modules": "none during cache build",
            "frozen_modules": "ID56 online Qwen backbone and shared DINO-grid projector",
        }
        fingerprint = _contract_fingerprint(contract)
        contract["fingerprint"] = fingerprint
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            contract_path = args.output_dir / "contract.json"
            if contract_path.is_file():
                existing = json.loads(contract_path.read_text(encoding="utf-8"))
                if existing != contract:
                    raise ValueError(
                        "state-cache resume contract differs from existing output"
                    )
            elif any(args.output_dir.iterdir()):
                raise FileExistsError(
                    f"state-cache output is nonempty without contract: {args.output_dir}"
                )
            else:
                _atomic_json(contract_path, contract)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        for split in ("train", "val"):
            _encode_split(
                args=args,
                split=split,
                samples=samples_by_split[split],
                processor=processor,
                token_id_map=token_id_map,
                model=model,
                projector=projector,
                device=device,
                rank=rank,
                world_size=world_size,
                contract_fingerprint=fingerprint,
            )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            if rank == 0:
                _finalize_split(
                    args=args,
                    split=split,
                    samples=samples_by_split[split],
                    world_size=world_size,
                    contract_fingerprint=fingerprint,
                    preprocess_manifest=preprocess_manifests[split],
                )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        if rank == 0:
            _atomic_json(
                args.output_dir / "summary.json",
                {
                    "status": "completed",
                    "contract_fingerprint": fingerprint,
                    "train_items": len(samples_by_split["train"]),
                    "val_items": len(samples_by_split["val"]),
                    "state_shape": list(GRID_SHAPE),
                },
            )
        return 0
    finally:
        cleanup_dist()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache ID56 actual DINO-grid states from compact SFT2 inputs"
    )
    parser.add_argument("--sft2-checkpoint", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--preprocess-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--preprocess-cache-processor-source",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--latent-token-count", type=int, default=16)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--state-cache-shard-size", type=int, default=128)
    parser.add_argument("--preprocess-cache-shard-lru", type=int, default=2)
    parser.add_argument("--preprocess-cache-image-dtype", default="bfloat16")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
