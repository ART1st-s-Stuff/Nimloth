"""Build a fingerprinted 8×1024 State cache without loading Qwen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import torch

from nimloth.rcdm.state_cache import RCDMStateCacheDataset, RCDMStateCacheManifest
from nimloth.wm.frozen_query_state import FrozenQueryStateEncoder

CACHE_VERSION = "frozen_query_state_v1"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _fingerprint(source: str, encoder_hash: str, step: int) -> str:
    payload = f"{CACHE_VERSION}|{source}|{encoder_hash}|{step}|float16"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _row_metadata(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "state_emb"}


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _batches(dataset: RCDMStateCacheDataset, size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(dataset), size):
        yield [dataset[index] for index in range(start, min(start + size, len(dataset)))]


@torch.inference_mode()
def _encoded_rows(
    dataset: RCDMStateCacheDataset,
    encoder: FrozenQueryStateEncoder,
    device: torch.device,
    batch_size: int,
) -> Iterator[tuple[dict[str, Any], torch.Tensor]]:
    for items in _batches(dataset, batch_size):
        query = torch.stack([item["state_emb"] for item in items]).to(device)
        states = encoder(query).to(dtype=torch.float16).cpu()
        yield from zip((_row_metadata(item) for item in items), states, strict=True)


def _write_shard(
    output_dir: Path,
    index: int,
    rows: list[dict[str, Any]],
    states: list[torch.Tensor],
) -> dict[str, Any]:
    filename = f"shard_{index:06d}.pt"
    payload = {"state_emb": torch.stack(states), "rows": rows}
    _atomic_torch_save(payload, output_dir / filename)
    return {"file": filename, "count": len(rows)}


def _write_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "manifest.json")


def _existing_manifest(output_dir: Path, fingerprint: str) -> RCDMStateCacheManifest | None:
    path = output_dir / "manifest.json"
    if not path.is_file():
        return None
    manifest = RCDMStateCacheManifest.load(output_dir)
    if manifest.fingerprint != fingerprint:
        raise ValueError("frozen State cache fingerprint mismatch; use a fresh output directory")
    return manifest


def build_frozen_query_state_cache(
    source_dir: Path,
    output_dir: Path,
    encoder_checkpoint: Path,
    *,
    shard_size: int = 512,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> RCDMStateCacheManifest:
    source = RCDMStateCacheDataset(source_dir)
    encoder_hash = _file_hash(encoder_checkpoint)
    encoder = FrozenQueryStateEncoder.from_probe_checkpoint(encoder_checkpoint)
    fingerprint = _fingerprint(source.manifest.fingerprint, encoder_hash, encoder.source_step)
    existing = _existing_manifest(output_dir, fingerprint)
    if existing is not None:
        return existing
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("frozen State cache output is non-empty and has no valid manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _build(source, output_dir, encoder_checkpoint, encoder_hash, encoder.to(target), target, fingerprint, shard_size, batch_size)


def _build(
    source: RCDMStateCacheDataset,
    output_dir: Path,
    checkpoint: Path,
    encoder_hash: str,
    encoder: FrozenQueryStateEncoder,
    device: torch.device,
    fingerprint: str,
    shard_size: int,
    batch_size: int,
) -> RCDMStateCacheManifest:
    rows: list[dict[str, Any]] = []
    states: list[torch.Tensor] = []
    shards: list[dict[str, Any]] = []
    for row, state in _encoded_rows(source, encoder, device, batch_size):
        rows.append(row)
        states.append(state)
        if len(rows) == shard_size:
            shards.append(_write_shard(output_dir, len(shards), rows, states))
            rows, states = [], []
    if rows:
        shards.append(_write_shard(output_dir, len(shards), rows, states))
    payload = _manifest_payload(source, checkpoint, encoder_hash, encoder, fingerprint, shards)
    _write_manifest(output_dir, payload)
    return RCDMStateCacheManifest.load(output_dir)


def _manifest_payload(
    source: RCDMStateCacheDataset,
    checkpoint: Path,
    encoder_hash: str,
    encoder: FrozenQueryStateEncoder,
    fingerprint: str,
    shards: list[dict[str, Any]],
) -> dict[str, Any]:
    shape = [8, encoder.output_dim]
    return {
        "version": CACHE_VERSION,
        "representation": "frozen_query_state",
        "count": len(source),
        "cond_dim": 8 * encoder.output_dim,
        "state_shape": shape,
        "state_dtype": "float16",
        "compression": "none",
        "shard_size": max(shard["count"] for shard in shards),
        "shards": shards,
        "fingerprint": fingerprint,
        "source_cache": str(source.cache_dir),
        "source_fingerprint": source.manifest.fingerprint,
        "encoder_checkpoint": str(checkpoint),
        "encoder_hash": encoder_hash,
        "encoder_step": encoder.source_step,
        "view_contract": f"tokens8x{encoder.output_dim}_flatten_exact",
    }
