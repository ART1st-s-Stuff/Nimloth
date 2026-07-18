"""Project a preprojection Query cache through an SFT2 StateProjector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from nimloth.rcdm.state_cache import RCDMStateCacheManifest
from nimloth.wm.state_proj import StateProjector

VERSION = "project_query_cache_v1"


def resolve_projector_config(
    training_state: dict[str, Any], state_dict: dict[str, torch.Tensor]
) -> dict[str, int]:
    latent_count = int(training_state["latent_token_count"])
    qwen_dim = int(training_state["qwen_hidden_dim"])
    input_dim = int(training_state["state_proj_input_dim"])
    hidden_dim = int(training_state["state_proj_hidden_dim"])
    output_dim = int(training_state["state_proj_output_dim"])
    if input_dim != latent_count * qwen_dim:
        raise ValueError(
            f"projector input mismatch: {input_dim} != {latent_count}*{qwen_dim}"
        )
    first = state_dict["net.net.0.weight"]
    last = state_dict["net.net.3.weight"]
    if tuple(first.shape) != (hidden_dim, input_dim):
        raise ValueError(f"first projector layer mismatch: {tuple(first.shape)}")
    if tuple(last.shape) != (output_dim, hidden_dim):
        raise ValueError(f"last projector layer mismatch: {tuple(last.shape)}")
    return {
        "latent_token_count": latent_count,
        "qwen_hidden_dim": qwen_dim,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
    }


def _fingerprint(source: dict[str, Any], checkpoint: Path, config: dict[str, int]) -> str:
    stat = checkpoint.stat()
    payload = {
        "version": VERSION,
        "source_fingerprint": source["fingerprint"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_bytes": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "config": config,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@torch.no_grad()
def project_split(
    *,
    source_dir: Path,
    output_dir: Path,
    projector: StateProjector,
    checkpoint: Path,
    config: dict[str, int],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    source = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    if source.get("representation") != "qwen_query_hidden":
        raise ValueError(f"source is not qwen_query_hidden: {source_dir}")
    expected_shape = [config["latent_token_count"], config["qwen_hidden_dim"]]
    if source.get("state_shape") != expected_shape:
        raise ValueError(
            f"query shape mismatch: expected={expected_shape}, actual={source.get('state_shape')}"
        )
    fingerprint = _fingerprint(source, checkpoint, config)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == fingerprint and all(
            (output_dir / shard["file"]).is_file() for shard in existing["shards"]
        ):
            return existing
        raise ValueError(f"projected cache exists with a different fingerprint: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    projector.eval()
    shards: list[dict[str, Any]] = []
    total = 0
    total_bytes = 0
    for shard in source["shards"]:
        source_path = source_dir / shard["file"]
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        query = payload["state_emb"]
        outputs: list[torch.Tensor] = []
        for start in range(0, query.shape[0], batch_size):
            states = query[start : start + batch_size].to(
                device=device, dtype=next(projector.parameters()).dtype
            )
            outputs.append(projector(states).float().cpu().to(torch.float16))
        projected = torch.cat(outputs)
        if not torch.isfinite(projected).all():
            raise ValueError(f"non-finite projected state in {source_path}")
        destination = output_dir / shard["file"]
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save({"state_emb": projected, "rows": payload["rows"]}, temporary)
        temporary.replace(destination)
        count = int(projected.shape[0])
        shards.append({"file": destination.name, "count": count})
        total += count
        total_bytes += destination.stat().st_size
    if total != int(source["count"]):
        raise ValueError(f"projected row count mismatch: {total} != {source['count']}")
    manifest = RCDMStateCacheManifest(
        cache_dir=output_dir,
        count=total,
        cond_dim=config["output_dim"],
        state_dtype="float16",
        compression="none",
        shard_size=int(source["shard_size"]),
        shards=shards,
        fingerprint=fingerprint,
    )
    extra = {
        "split": source["split"],
        "representation": "projected",
        "state_shape": [config["output_dim"]],
        "source_query_cache": str(source_dir),
        "source_query_fingerprint": source["fingerprint"],
        "state_proj_checkpoint": str(checkpoint),
        "projector_config": config,
        "total_bytes": total_bytes,
    }
    manifest.write(extra)
    result = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps({"project_query_cache": str(output_dir), "count": total, "fingerprint": fingerprint}), flush=True)
    return result


def run(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_state = torch.load(args.training_state, map_location="cpu", weights_only=False)
    state_dict = torch.load(args.state_proj_checkpoint, map_location="cpu", weights_only=True)
    config = resolve_projector_config(training_state, state_dict)
    projector = StateProjector(
        config["qwen_hidden_dim"],
        config["output_dim"],
        projector_hidden_dim=config["hidden_dim"],
        latent_token_count=config["latent_token_count"],
    )
    projector.load_state_dict(state_dict, strict=True)
    projector.to(device=device, dtype=torch.float32).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        split: project_split(
            source_dir=args.query_cache_dir / split,
            output_dir=args.output_dir / split,
            projector=projector,
            checkpoint=args.state_proj_checkpoint,
            config=config,
            device=device,
            batch_size=args.batch_size,
        )
        for split in ("train", "val")
    }
    (args.output_dir / "projection_summary.json").write_text(
        json.dumps({"status": "completed", "config": config, "splits": results}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-cache-dir", type=Path, required=True)
    parser.add_argument("--state-proj-checkpoint", type=Path, required=True)
    parser.add_argument("--training-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
