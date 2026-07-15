"""Build and verify the shared frozen 8×1024 State cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from nimloth.rcdm.state_cache import RCDMStateCacheDataset
from nimloth.training.wm_heads.config import load_config, output_dir
from nimloth.wm.frozen_query_state import StateViews
from nimloth.wm.frozen_state_cache import build_frozen_query_state_cache


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verify(source_dir: Path, target_dir: Path, expected_count: int) -> dict[str, Any]:
    source, target = RCDMStateCacheDataset(source_dir), RCDMStateCacheDataset(target_dir)
    manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
    if len(source) != len(target) or len(target) != expected_count:
        raise ValueError(f"cache count mismatch source={len(source)} target={len(target)} expected={expected_count}")
    if manifest["source_fingerprint"] != source.manifest.fingerprint:
        raise ValueError("derived cache source fingerprint mismatch")
    for index in range(len(target)):
        source_row, target_row = source[index], target[index]
        if str(source_row["id"]) != str(target_row["id"]):
            raise ValueError(f"derived cache ordering mismatch at row {index}")
        state = target_row["state_emb"]
        if tuple(state.shape) != (8, 1024) or not torch.isfinite(state).all():
            raise ValueError(f"invalid frozen State at row {index}: {tuple(state.shape)}")
        views = StateViews.from_tokens(state.unsqueeze(0).contiguous())
        if not torch.equal(views.vector.reshape_as(views.tokens), views.tokens):
            raise ValueError(f"flatten view mismatch at row {index}")
    return {"count": len(target), "fingerprint": manifest["fingerprint"], "source_fingerprint": manifest["source_fingerprint"], "state_shape": manifest["state_shape"], "finite": True, "ordered": True, "view_exact": True}


def run(config_path: Path, device: torch.device) -> dict[str, Any]:
    config, started = load_config(config_path), time.perf_counter()
    source_root = Path(config["inputs"]["query_cache"])
    cache_root = output_dir(config) / "state_cache"
    expected = {"train": int(config["state"]["train_count"]), "val": int(config["state"]["val_count"])}
    summaries = {}
    for split in ("train", "val"):
        build_frozen_query_state_cache(source_root / split, cache_root / split, Path(config["inputs"]["encoder_checkpoint"]), device=device)
        summaries[split] = _verify(source_root / split, cache_root / split, expected[split])
    summary = {"status": "completed", "duration_s": time.perf_counter() - started, "device": str(device), "loaded_modules": ["FrozenQueryStateEncoder"], "qwen_loaded": False, "splits": summaries}
    cache_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(cache_root / "cache_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.config, torch.device(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
