"""Build an aligned positive-control cache from Qwen vision features.

The cache condition is the exact representation used by the previously useful
ViT-token CFM: old SFT2 Qwen visual features ``(81, 2048)`` passed through the
frozen rollout-4 attention compressor to ``(16, 512)``.  Rows are read from an
existing state cache so positive/query/projected representations align exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens
from nimloth.rcdm.state_cache import (
    RCDMStateCacheDataset,
    RCDMStateCacheManifest,
    contiguous_rank_bounds,
)
from nimloth.training.common.dist import cleanup_dist, setup_dist

POSITIVE_CACHE_VERSION = "qwen_compressed_vision_positive_v1"


@dataclass(frozen=True)
class AttentionTokenCompressorConfig:
    input_dim: int = 2048
    output_dim: int = 512
    input_tokens: int = 81
    num_output_tokens: int = 16
    depth: int = 2
    heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1


class _CompressorBlock(nn.Module):
    def __init__(self, config: AttentionTokenCompressorConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.output_dim)
        self.input_norm = nn.LayerNorm(config.output_dim)
        self.cross_attn = nn.MultiheadAttention(
            config.output_dim, config.heads, dropout=config.dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(config.output_dim)
        self.self_attn = nn.MultiheadAttention(
            config.output_dim, config.heads, dropout=config.dropout, batch_first=True
        )
        hidden = config.output_dim * config.mlp_ratio
        self.mlp_norm = nn.LayerNorm(config.output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.output_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.output_dim),
        )

    def forward(self, queries: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        normalized_inputs = self.input_norm(inputs)
        queries = queries + self.cross_attn(
            normalized_queries, normalized_inputs, normalized_inputs, need_weights=False
        )[0]
        normalized_queries = self.self_norm(queries)
        queries = queries + self.self_attn(
            normalized_queries, normalized_queries, normalized_queries, need_weights=False
        )[0]
        return queries + self.mlp(self.mlp_norm(queries))


class AttentionTokenCompressor(nn.Module):
    """Checkpoint-compatible copy of the proven 81x2048 -> 16x512 compressor."""

    def __init__(self, config: AttentionTokenCompressorConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.output_dim)
        self.input_pos = nn.Parameter(
            torch.randn(1, config.input_tokens, config.output_dim) * 0.02
        )
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_output_tokens, config.output_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [_CompressorBlock(config) for _ in range(config.depth)]
        )
        self.out_norm = nn.LayerNorm(config.output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        expected = (self.config.input_tokens, self.config.input_dim)
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"expected vision tokens (B, {expected[0]}, {expected[1]}), got {tuple(tokens.shape)}")
        inputs = self.input_proj(tokens.float()) + self.input_pos
        queries = self.query_tokens.expand(tokens.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, inputs)
        return self.out_norm(queries)

    @classmethod
    def load_checkpoint(
        cls, checkpoint: Path, *, map_location: str | torch.device = "cpu"
    ) -> "AttentionTokenCompressor":
        config = AttentionTokenCompressorConfig(
            **json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        )
        module = cls(config)
        module.load_state_dict(
            torch.load(
                checkpoint / "compressor.pt",
                map_location=map_location,
                weights_only=True,
            ),
            strict=True,
        )
        return module


class _SourceRows(Dataset):
    def __init__(self, source_cache_dir: Path) -> None:
        self.dataset = RCDMStateCacheDataset(source_cache_dir)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[index]
        row.pop("state_emb", None)
        return row


def _collate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def _load_rgb(path: str, *, input_image_size: int) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    if input_image_size > 0 and rgb.size != (input_image_size, input_image_size):
        resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
        rgb = rgb.resize((input_image_size, input_image_size), resample)
    return rgb


def _vision_batch(
    paths: list[str],
    processor,
    *,
    max_pixels: int,
    input_image_size: int,
) -> dict[str, torch.Tensor]:
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = max_pixels
    images = [[_load_rgb(path, input_image_size=input_image_size)] for path in paths]
    text = ["<|vision_start|><|image_pad|><|vision_end|>" for _ in paths]
    return processor(text=text, images=images, padding=True, return_tensors="pt")


@torch.no_grad()
def extract_qwen_vision_tokens(
    model,
    processor,
    paths: list[str],
    *,
    device: torch.device,
    max_pixels: int,
    expected_tokens: int,
    input_image_size: int,
) -> torch.Tensor:
    encoded = _vision_batch(
        paths,
        processor,
        max_pixels=max_pixels,
        input_image_size=input_image_size,
    )
    pixel_values = encoded["pixel_values"].to(device=device, dtype=model.visual.dtype)
    grid = encoded["image_grid_thw"].to(device=device)
    flat = model.visual(pixel_values, grid_thw=grid)
    merge_unit = int(getattr(model.visual, "spatial_merge_unit", 1))
    lengths = [math.prod(int(value) for value in row.tolist()) // merge_unit for row in grid.cpu()]
    if any(length != expected_tokens for length in lengths):
        raise ValueError(f"expected {expected_tokens} visual tokens, got {lengths}")
    if sum(lengths) != flat.shape[0]:
        raise ValueError(f"vision token split mismatch: lengths={lengths}, flat={tuple(flat.shape)}")
    return torch.stack(list(flat.split(lengths, dim=0)), dim=0)


def _path_signature(path: Path) -> str:
    if path.is_dir():
        parts = []
        for name in ("config.json", "adapter_config.json", "vision_full_state.pt", "compressor.pt"):
            child = path / name
            if child.is_file():
                stat = child.stat()
                parts.append(f"{child.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
        return "|".join(parts) or str(path.resolve())
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def positive_cache_fingerprint(
    *,
    source_fingerprint: str,
    qwen_checkpoint: Path,
    compressor_checkpoint: Path,
    max_pixels: int,
    max_items: int,
    input_image_size: int,
) -> str:
    payload = "|".join(
        [
            POSITIVE_CACHE_VERSION,
            source_fingerprint,
            _path_signature(qwen_checkpoint),
            _path_signature(compressor_checkpoint),
            str(max_pixels),
            str(max_items),
            str(input_image_size),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_visual_model(checkpoint: Path, device: torch.device, *, max_pixels: int):
    from nimloth.backbone.qwen_tuning import configure_qwen_tuning
    from nimloth.training.sft2.checkpoint import load_lora_adapter_state

    adapter_config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    base = Path(adapter_config["base_model_name_or_path"])
    processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = max_pixels
    add_special_tokens(processor.tokenizer)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    # The preserved checkpoint's vision_full_state was saved after PEFT had
    # wrapped matching vision projections.  Recreate that exact module topology
    # before loading rather than forcing PEFT keys into a plain visual encoder.
    tuning_args = SimpleNamespace(
        lora=False,
        llm_tune="lora",
        vision_tune="full" if (checkpoint / "vision_full_state.pt").is_file() else "freeze",
        lora_r=int(adapter_config.get("r", 64)),
        lora_alpha=int(adapter_config.get("lora_alpha", 128)),
        lora_dropout=float(adapter_config.get("lora_dropout", 0.0)),
        gradient_checkpointing=False,
    )
    model = configure_qwen_tuning(model, tuning_args)
    load_lora_adapter_state(model, checkpoint)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model


def build_positive_cache(args: argparse.Namespace) -> RCDMStateCacheManifest:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    device = torch.device("cuda", int(__import__("os").environ.get("LOCAL_RANK", "0"))) if torch.cuda.is_available() else torch.device("cpu")
    source_manifest = RCDMStateCacheManifest.load(args.source_cache_dir)
    count = source_manifest.count if args.max_items < 0 else min(args.max_items, source_manifest.count)
    fingerprint = positive_cache_fingerprint(
        source_fingerprint=source_manifest.fingerprint,
        qwen_checkpoint=args.qwen_checkpoint,
        compressor_checkpoint=args.compressor_checkpoint,
        max_pixels=args.max_pixels,
        max_items=args.max_items,
        input_image_size=args.input_image_size,
    )
    manifest_path = args.output_dir / "manifest.json"
    hit = False
    if rank == 0 and manifest_path.is_file():
        existing = RCDMStateCacheManifest.load(args.output_dir)
        hit = existing.fingerprint == fingerprint and all(
            (args.output_dir / str(shard["file"])).is_file() for shard in existing.shards
        )
    if world > 1:
        payload = [hit]
        dist.broadcast_object_list(payload, src=0)
        hit = bool(payload[0])
    if hit:
        return RCDMStateCacheManifest.load(args.output_dir)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for old in args.output_dir.glob("shard_*.pt"):
            old.unlink()
        manifest_path.unlink(missing_ok=True)
    if world > 1:
        dist.barrier()

    processor, model = _load_visual_model(args.qwen_checkpoint, device, max_pixels=args.max_pixels)
    compressor = AttentionTokenCompressor.load_checkpoint(
        args.compressor_checkpoint, map_location=device
    ).to(device).eval()
    for parameter in compressor.parameters():
        parameter.requires_grad_(False)

    dataset = _SourceRows(args.source_cache_dir)
    start, end = contiguous_rank_bounds(count, rank, world)
    loader = DataLoader(
        Subset(dataset, range(start, end)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_rows,
    )
    shard_rows: list[dict[str, Any]] = []
    shard_states: list[torch.Tensor] = []
    shards: list[dict[str, Any]] = []
    shard_index = 0

    def flush() -> None:
        nonlocal shard_rows, shard_states, shard_index
        if not shard_rows:
            return
        filename = f"shard_r{rank:03d}_{shard_index:06d}.pt"
        torch.save(
            {
                "state_emb": torch.stack(shard_states).to(torch.float16),
                "rows": shard_rows,
            },
            args.output_dir / filename,
        )
        shards.append({"file": filename, "count": len(shard_rows)})
        shard_rows = []
        shard_states = []
        shard_index += 1

    local_count = 0
    for rows in loader:
        paths = [str(row["current_image_path"]) for row in rows]
        visual = extract_qwen_vision_tokens(
            model,
            processor,
            paths,
            device=device,
            max_pixels=args.max_pixels,
            expected_tokens=compressor.config.input_tokens,
            input_image_size=args.input_image_size,
        )
        compressed = compressor(visual.float()).detach().cpu()
        for row, state in zip(rows, compressed, strict=True):
            shard_rows.append(dict(row))
            shard_states.append(state)
            local_count += 1
            if len(shard_rows) >= args.shard_size:
                flush()
    flush()
    local = {
        "rank": rank,
        "start": start,
        "end": end,
        "count": local_count,
        "shards": shards,
        "bytes": sum((args.output_dir / str(shard["file"])).stat().st_size for shard in shards),
    }
    if world > 1:
        gathered: list[dict[str, Any] | None] = [None] * world
        dist.all_gather_object(gathered, local)
        results = [item for item in gathered if item is not None]
    else:
        results = [local]
    results.sort(key=lambda item: int(item["rank"]))
    if rank == 0:
        merged_shards = [shard for item in results for shard in item["shards"]]
        merged_count = sum(int(item["count"]) for item in results)
        if merged_count != count:
            raise RuntimeError(f"positive cache count mismatch: {merged_count} != {count}")
        manifest = RCDMStateCacheManifest(
            cache_dir=args.output_dir,
            count=count,
            cond_dim=compressor.config.num_output_tokens * compressor.config.output_dim,
            state_dtype="float16",
            compression="none",
            shard_size=args.shard_size,
            shards=merged_shards,
            fingerprint=fingerprint,
        )
        manifest.write(
            {
                "source_cache_dir": str(args.source_cache_dir),
                "source_fingerprint": source_manifest.fingerprint,
                "qwen_checkpoint": str(args.qwen_checkpoint),
                "compressor_checkpoint": str(args.compressor_checkpoint),
                "representation": "qwen_compressed_vision_positive",
                "state_shape": [compressor.config.num_output_tokens, compressor.config.output_dim],
                "max_pixels": args.max_pixels,
                "max_items": args.max_items,
                "input_image_size": args.input_image_size,
                "cache_build_world_size": world,
                "rank_ranges": results,
                "total_bytes": sum(int(item["bytes"]) for item in results),
            }
        )
        print(json.dumps({"positive_cache": "done", "count": count, "world_size": world, "dir": str(args.output_dir)}), flush=True)
    if world > 1:
        dist.barrier()
    return RCDMStateCacheManifest.load(args.output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build aligned compressed-Qwen positive-control cache")
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--compressor-checkpoint", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=-1)
    parser.add_argument("--max-pixels", type=int, default=602112)
    parser.add_argument(
        "--input-image-size",
        type=int,
        default=255,
        help="Resize current 512px rollout images to the proven old 255px Qwen-feature input",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_dist()
    try:
        build_positive_cache(build_arg_parser().parse_args(argv))
        return 0
    finally:
        cleanup_dist()


if __name__ == "__main__":
    raise SystemExit(main())
