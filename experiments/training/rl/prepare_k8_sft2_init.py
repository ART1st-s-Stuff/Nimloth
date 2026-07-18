#!/usr/bin/env python3
"""Snapshot a live SFT2 LoRA checkpoint into an immutable RL-init bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch


ROOT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "preprocessor_config.json",
    "README.md",
    "special_tokens_map.json",
    "state_proj.pt",
    "tokenizer_config.json",
    "tokenizer.json",
    "video_preprocessor_config.json",
    "vision_ema.pt",
    "vocab.json",
)
TREE_FILES = (
    "wm_predictor/config.json",
    "wm_predictor/predictor.pt",
    "value_head/config.json",
    "value_head/value_head.pt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _sanitized_training_state(state: dict[str, Any]) -> dict[str, Any]:
    """Keep protocol/shape provenance but deliberately omit SFT optimizer tensors."""

    keep = (
        "step",
        "epoch",
        "latent_token_count",
        "latent_query_mode",
        "mask_latent_query_labels",
        "query_tune",
        "qwen_hidden_dim",
        "state_proj_input_dim",
        "state_proj_hidden_dim",
        "state_proj_output_dim",
        "value_hidden_dim",
        "best_val_success_rate",
        "best_val_wm_mse",
        "best_val",
        "lora",
        "llm_tune",
        "vision_tune",
        "vision_ema",
        "epoch_complete",
        "micro_step_in_epoch",
        "training_invariants",
        "base_model_path",
    )
    return {key: state[key] for key in keep if key in state}


def snapshot_checkpoint(
    source: Path,
    output: Path,
    *,
    require_epoch_complete: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"snapshot output already exists: {output}")
    state_path = source / "training_state.pt"
    before_state = _load_state(state_path)
    before_step = int(before_state.get("step", -1))
    if before_step < 0:
        raise ValueError(f"source checkpoint has no valid step: {state_path}")
    if int(before_state.get("latent_token_count", -1)) != 8:
        raise ValueError("source checkpoint is not k=8")
    if before_state.get("latent_query_mode") != "inject":
        raise ValueError("source checkpoint is not inject mode")
    if require_epoch_complete and not bool(before_state.get("epoch_complete", False)):
        raise ValueError(
            "source checkpoint is not a complete epoch; refusing RL initialization"
        )

    relative_files = [Path(name) for name in (*ROOT_FILES, *TREE_FILES)]
    missing = [str(path) for path in relative_files if not (source / path).is_file()]
    if missing:
        raise FileNotFoundError(f"source checkpoint is incomplete: {missing}")

    output.mkdir(parents=True)
    hashes_before = {str(path): _sha256(source / path) for path in relative_files}
    for relative in relative_files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)

    sanitized = _sanitized_training_state(before_state)
    torch.save(sanitized, output / "training_state.pt")
    hashes_after_source = {str(path): _sha256(source / path) for path in relative_files}
    hashes_output = {str(path): _sha256(output / path) for path in relative_files}
    after_state = _load_state(state_path)
    after_step = int(after_state.get("step", -1))

    stable = (
        before_step == after_step
        and hashes_before == hashes_after_source
        and hashes_before == hashes_output
    )
    manifest = {
        "source": str(source),
        "source_step": before_step,
        "source_epoch": int(before_state.get("epoch", -1)),
        "source_epoch_complete": bool(before_state.get("epoch_complete", False)),
        "required_epoch_complete": bool(require_epoch_complete),
        "latent_token_count": 8,
        "latent_query_mode": "inject",
        "base_model_path": str(before_state.get("base_model_path", "")),
        "sft_optimizer_omitted": True,
        "files": {
            name: {"bytes": (output / name).stat().st_size, "sha256": digest}
            for name, digest in hashes_output.items()
        },
        "stable_during_copy": stable,
    }
    (output / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not stable:
        (output / "SNAPSHOT_FAILED").write_text(
            "source latest changed during copy; do not use\n", encoding="utf-8"
        )
        raise RuntimeError("source checkpoint changed during snapshot copy")
    (output / "SNAPSHOT_READY").write_text("ready\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-epoch-complete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(snapshot_checkpoint(
        args.source,
        args.output,
        require_epoch_complete=args.require_epoch_complete,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
