"""Build RCDM state caches from GT compressed Qwen vision tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from nimloth.rcdm.state_cache import RCDMStateCacheManifest, _save_payload
from nimloth.representation_ablation.compressor import AttentionTokenCompressor
from nimloth.representation_ablation.vision_token_cache import VisionTokenCache
from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch

StateDType = Literal["float16", "bfloat16", "float32"]
Compression = Literal["gzip", "none"]


def _torch_dtype(name: StateDType) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported state dtype: {name}")


def _shard_name(index: int, compression: Compression) -> str:
    return f"shard_{index:06d}{'.pt.gz' if compression == 'gzip' else '.pt'}"


@torch.no_grad()
def build_compressed_vision_rcdm_cache(
    *,
    jsonl_path: Path,
    vision_token_cache_dir: Path,
    compressor_checkpoint: Path,
    cache_dir: Path,
    split_name: str,
    device: torch.device,
    max_records: int = -1,
    success_only: bool = False,
    batch_size: int = 128,
    shard_size: int = 4096,
    compression: Compression = "gzip",
    state_dtype: StateDType = "float16",
    force: bool = False,
) -> RCDMStateCacheManifest:
    """Build an RCDM-compatible cache where condition = flatten(compressor(QwenVision(image)))."""

    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file() and not force:
        manifest = RCDMStateCacheManifest.load(cache_dir)
        print(json.dumps({"compressed_vision_rcdm_cache": "hit", "split": split_name, "dir": str(cache_dir), "count": manifest.count}))
        return manifest

    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("shard_*.pt*"):
        old.unlink()
    if manifest_path.exists():
        manifest_path.unlink()

    vision_cache = VisionTokenCache(vision_token_cache_dir, device=device)
    compressor = AttentionTokenCompressor.load_checkpoint(compressor_checkpoint, map_location=device).to(device)
    compressor.eval()
    target_dtype = _torch_dtype(state_dtype)

    ds = TransitionQwenDataset(jsonl_path, max_records=max_records, success_only=success_only)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_transition_batch)

    shard_rows: list[dict[str, Any]] = []
    shard_states: list[torch.Tensor] = []
    shards: list[dict[str, Any]] = []
    shard_index = 0
    count = 0
    cond_dim = compressor.num_tokens * compressor.emb_dim

    def flush() -> None:
        nonlocal shard_rows, shard_states, shard_index
        if not shard_rows:
            return
        filename = _shard_name(shard_index, compression)
        states = torch.stack(shard_states, dim=0).to(dtype=target_dtype)
        _save_payload({"state_emb": states, "rows": shard_rows}, cache_dir / filename, compression)
        shards.append({"file": filename, "count": len(shard_rows)})
        shard_rows = []
        shard_states = []
        shard_index += 1

    for items in loader:
        paths = [item["current_image_path"] for item in items]
        tokens = vision_cache.get_many(paths).to(device=device)
        states = compressor(tokens).flatten(1).detach().float().cpu()
        for item, state in zip(items, states, strict=True):
            shard_rows.append(
                {
                    "id": str(item.get("id", count)),
                    "record_id": str(item.get("record_id", "")),
                    "step_index": int(item.get("step_index", -1)),
                    "action_index": int(item.get("action_index", -1)),
                    "success": bool(item.get("success", False)),
                    "current_image_path": str(item["current_image_path"]),
                    "next_image_path": str(item["next_image_path"]),
                }
            )
            shard_states.append(state)
            count += 1
            if len(shard_rows) >= shard_size:
                flush()
    flush()

    manifest = RCDMStateCacheManifest(
        cache_dir=cache_dir,
        count=count,
        cond_dim=cond_dim,
        state_dtype=state_dtype,
        compression=compression,
        shard_size=shard_size,
        shards=shards,
        fingerprint="compressed_vision_gt",
    )
    manifest.write(
        {
            "source": "compressed_qwen_vision_tokens",
            "split": split_name,
            "jsonl_path": str(jsonl_path),
            "vision_token_cache_dir": str(vision_token_cache_dir),
            "compressor_checkpoint": str(compressor_checkpoint),
            "max_records": max_records,
            "success_only": success_only,
            "condition": "flatten(AttentionTokenCompressor(Qwen visual tokens of current_image_path))",
        }
    )
    print(json.dumps({"compressed_vision_rcdm_cache": "built", "split": split_name, "dir": str(cache_dir), "count": count, "cond_dim": cond_dim}))
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build RCDM cache from GT compressed Qwen vision tokens")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--vision-token-cache-dir", type=Path, required=True)
    ap.add_argument("--compressor-checkpoint", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--split-name", default="train")
    ap.add_argument("--max-records", type=int, default=-1)
    ap.add_argument("--success-only", action="store_true")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--compression", choices=("gzip", "none"), default="gzip")
    ap.add_argument("--state-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--force", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_compressed_vision_rcdm_cache(
        jsonl_path=args.jsonl,
        vision_token_cache_dir=args.vision_token_cache_dir,
        compressor_checkpoint=args.compressor_checkpoint,
        cache_dir=args.cache_dir,
        split_name=args.split_name,
        device=device,
        max_records=args.max_records,
        success_only=args.success_only,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        compression=args.compression,
        state_dtype=args.state_dtype,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
