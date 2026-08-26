#!/usr/bin/env python3
"""Generate, finalize evidence for, or inspect the fresh SFT1-v2 teacher cache."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.distributed as dist

from nimloth.training.sft1.cache_runtime import (
    audit_fresh_cache_parity,
    generate_teacher_cache,
)
from nimloth.training.sft1.controller import assert_clean_resolved_source
from nimloth.training.sft1.data import sha256_file
from nimloth.training.sft1.driver import build_training_manifest
from nimloth.training.sft1.experiment_config import load_sft1_v2_config
from nimloth.training.sft1.manifest import load_sft1_v2_manifest
from nimloth.training.sft1.teacher_cache import inspect_teacher_cache


def _distributed_device(expected_world_size: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("fresh SFT1-v2 teacher generation requires CUDA")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != expected_world_size or not 0 <= rank < world_size:
        raise ValueError("cache torchrun world differs from resolved config")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def _atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"immutable cache evidence exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _finalize_evidence(config) -> dict[str, object]:
    output = Path(config.cache.output_dir)
    summary = inspect_teacher_cache(output)
    expected_manifest = build_training_manifest(config, summary)
    manifest_path = output / "training_manifest.json"
    if manifest_path.exists():
        if load_sft1_v2_manifest(manifest_path) != expected_manifest:
            raise ValueError("existing training manifest differs from fresh cache/config")
    else:
        _atomic_text(
            manifest_path,
            json.dumps(asdict(expected_manifest), indent=2, sort_keys=True) + "\n",
        )
    parity_path = output / "parity_report.json"
    if parity_path.exists():
        parity_sha = sha256_file(parity_path)
    else:
        parity_sha = audit_fresh_cache_parity(
            config,
            manifest_identity=expected_manifest.identity,
            output_path=parity_path,
        )
    readme = output / "README.md"
    text = (
        "# SFT1-v2 fresh detached teacher cache\n\n"
        "Generated from original early-4 observations, actual archived CoT, "
        "exact rendered instruction spans, frozen ID176, and pinned DINO. "
        "Contains no student hidden/state or pre-encoded student prompt.\n"
    )
    if readme.exists():
        if readme.read_text(encoding="utf-8") != text:
            raise ValueError("existing fresh cache README differs")
    else:
        _atomic_text(readme, text)
    return {
        "cache": asdict(summary),
        "training_manifest_identity": expected_manifest.identity,
        "training_manifest_sha256": sha256_file(manifest_path),
        "parity_report_sha256": parity_sha,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--inspect", type=Path)
    parser.add_argument("--finalize-evidence", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.inspect is not None:
        if args.config is not None or args.finalize_evidence:
            raise ValueError("--inspect is exclusive")
        print(json.dumps(
            asdict(inspect_teacher_cache(args.inspect)), indent=2, sort_keys=True
        ))
        return 0
    if args.config is None:
        raise ValueError("generation/evidence finalization requires --config")
    config = load_sft1_v2_config(args.config)
    assert_clean_resolved_source(config, args.repo_root)
    if args.finalize_evidence:
        print(json.dumps(_finalize_evidence(config), indent=2, sort_keys=True))
        return 0

    rank, world_size, device = _distributed_device(config.runtime.world_size)
    try:
        summary = generate_teacher_cache(
            config,
            repo_root=args.repo_root,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        torch.cuda.empty_cache()
        if rank == 0:
            if summary is None:
                raise RuntimeError("rank zero did not finalize teacher cache")
            print(json.dumps(_finalize_evidence(config), indent=2, sort_keys=True))
        dist.barrier()
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
