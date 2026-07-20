"""Project cached SFT1 query hidden states through its shared DINO-grid projector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nimloth.rcdm.state_cache import RCDMStateCacheManifest

VERSION = "project_dino_grid_cache_v1"


class SharedSlotProjector(nn.Module):
    """Checkpoint-compatible token-wise SFT1 shared slot projector."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, *, grid_tokens: int) -> None:
        super().__init__()
        self.grid_tokens = int(grid_tokens)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[1] != self.grid_tokens:
            raise ValueError(
                f"shared slot projector expected [B,{self.grid_tokens},D], got {tuple(hidden.shape)}"
            )
        return self.net(hidden.to(dtype=next(self.parameters()).dtype))


def load_grid_projector(checkpoint: Path) -> tuple[dict[str, Any], SharedSlotProjector]:
    config = json.loads((checkpoint / "grid_state_config.json").read_text(encoding="utf-8"))
    required = {
        "grid_size", "grid_tokens", "qwen_hidden_dim", "state_dim",
        "projector_hidden_dim", "shared_slot_projector", "ordering",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"grid projector config missing fields: {missing}")
    grid_tokens = int(config["grid_tokens"])
    if config["shared_slot_projector"] is not True:
        raise ValueError("grid config shared_slot_projector must be true")
    if int(config["grid_size"]) ** 2 != grid_tokens:
        raise ValueError("grid_size/grid_tokens mismatch")
    if config["ordering"] != "row_major":
        raise ValueError(f"grid config ordering must be row_major, got {config['ordering']!r}")
    dimensions = {
        "qwen_hidden_dim": int(config["qwen_hidden_dim"]),
        "state_dim": int(config["state_dim"]),
        "projector_hidden_dim": int(config["projector_hidden_dim"]),
    }
    state_dict = torch.load(checkpoint / "slot_projector.pt", map_location="cpu", weights_only=True)
    expected_shapes = {
        "net.0.weight": (dimensions["projector_hidden_dim"], dimensions["qwen_hidden_dim"]),
        "net.0.bias": (dimensions["projector_hidden_dim"],),
        "net.1.weight": (dimensions["projector_hidden_dim"],),
        "net.1.bias": (dimensions["projector_hidden_dim"],),
        "net.3.weight": (dimensions["state_dim"], dimensions["projector_hidden_dim"]),
        "net.3.bias": (dimensions["state_dim"],),
    }
    if set(state_dict) != set(expected_shapes):
        raise ValueError(f"slot projector keys mismatch: {sorted(state_dict)}")
    for key, shape in expected_shapes.items():
        if tuple(state_dict[key].shape) != shape:
            raise ValueError(f"slot projector shape mismatch for {key}: {tuple(state_dict[key].shape)} != {shape}")
    projector = SharedSlotProjector(
        dimensions["qwen_hidden_dim"], dimensions["state_dim"],
        dimensions["projector_hidden_dim"], grid_tokens=grid_tokens,
    )
    projector.load_state_dict(state_dict, strict=True)
    return config, projector


def _fingerprint(source: dict[str, Any], checkpoint: Path, config: dict[str, Any]) -> str:
    files = [checkpoint / "grid_state_config.json", checkpoint / "slot_projector.pt"]
    payload = {
        "version": VERSION,
        "source_fingerprint": source["fingerprint"],
        "checkpoint": str(checkpoint.resolve()),
        "files": [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in files],
        "config": config,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@torch.no_grad()
def project_split(*, source_dir: Path, output_dir: Path, projector: SharedSlotProjector,
                  checkpoint: Path, config: dict[str, Any], device: torch.device,
                  batch_size: int) -> dict[str, Any]:
    source = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    if source.get("representation") != "qwen_query_hidden":
        raise ValueError("DINO grid projection source must be qwen_query_hidden")
    expected_shape = [int(config["grid_tokens"]), int(config["qwen_hidden_dim"])]
    if [int(value) for value in source.get("state_shape", [])] != expected_shape:
        raise ValueError(f"query shape mismatch: expected={expected_shape}, actual={source.get('state_shape')}")
    source_model = source.get("model_path")
    if source_model is None or Path(source_model).resolve() != checkpoint.resolve():
        raise ValueError(
            "query cache/model lineage mismatch: "
            f"source model_path={source_model!r}, grid checkpoint={str(checkpoint)!r}"
        )
    fingerprint = _fingerprint(source, checkpoint, config)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and all(
            (output_dir / shard["file"]).is_file() for shard in existing["shards"]
        ):
            return existing
        raise ValueError(f"DINO grid cache exists with a different fingerprint: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    projector.to(
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).eval()
    shards, total, total_bytes = [], 0, 0
    for shard in source["shards"]:
        payload = torch.load(source_dir / shard["file"], map_location="cpu", weights_only=False)
        query = payload["state_emb"]
        if list(query.shape[1:]) != expected_shape:
            raise ValueError(f"query shard shape mismatch in {shard['file']}: {tuple(query.shape)}")
        outputs = []
        for start in range(0, query.shape[0], batch_size):
            outputs.append(projector(query[start:start + batch_size].to(device)).float().cpu().half())
        state = torch.cat(outputs)
        if not torch.isfinite(state).all():
            raise ValueError(f"non-finite DINO grid state in {shard['file']}")
        destination = output_dir / shard["file"]
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save({"state_emb": state, "rows": payload["rows"]}, temporary)
        temporary.replace(destination)
        count = int(state.shape[0])
        shards.append({"file": destination.name, "count": count})
        total += count
        total_bytes += destination.stat().st_size
    if total != int(source["count"]):
        raise ValueError(f"DINO grid row count mismatch: {total} != {source['count']}")
    manifest = RCDMStateCacheManifest(
        cache_dir=output_dir, count=total,
        cond_dim=int(config["grid_tokens"]) * int(config["state_dim"]),
        state_dtype="float16", compression="none", shard_size=int(source["shard_size"]),
        shards=shards, fingerprint=fingerprint,
    )
    manifest.write({
        "split": source["split"], "representation": "dino_grid_state",
        "state_shape": [int(config["grid_tokens"]), int(config["state_dim"])],
        "latent_token_count": int(config["grid_tokens"]),
        "source_query_cache": str(source_dir),
        "source_query_fingerprint": source["fingerprint"],
        "source_checkpoint": source["model_path"],
        "grid_checkpoint": str(checkpoint), "grid_state_config": config,
        "total_bytes": total_bytes,
    })
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, projector = load_grid_projector(args.sft1_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {split: project_split(
        source_dir=args.query_cache_dir / split, output_dir=args.output_dir / split,
        projector=projector, checkpoint=args.sft1_checkpoint, config=config,
        device=device, batch_size=args.batch_size,
    ) for split in ("train", "val")}
    (args.output_dir / "projection_summary.json").write_text(
        json.dumps({"status": "completed", "config": config, "splits": results}, indent=2) + "\n"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project query cache to DINO-supervised SFT1 grid state")
    parser.add_argument("--query-cache-dir", type=Path, required=True)
    parser.add_argument("--sft1-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
