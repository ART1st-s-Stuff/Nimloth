"""Run full validation metrics and six-rollout matched-noise rendering."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from nimloth.eval.matched_wm_ablation import (
    evaluate_full_dynamics,
    load_frozen_state_adapter,
    load_state_records,
    load_turn_spec,
    prepare_turn_batch,
    render_turn_comparison,
    write_turn_artifacts,
)
from nimloth.training.reconstruction.state_to_vision_tokens import load_proven_cfm
from nimloth.training.wm_heads.config import load_config, output_dir
from nimloth.wm.matched_heads import MatchedWMHeads


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest_gate(path: Path, representation: str, shape: list[int]) -> None:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("representation") != representation:
        raise ValueError(f"representation mismatch for {path}: {manifest.get('representation')}")
    if [int(value) for value in manifest.get("state_shape", [])] != shape:
        raise ValueError(f"state shape mismatch for {path}: {manifest.get('state_shape')}")


def _wandb_upload(config: dict, root: Path, artifacts: dict, dynamics: dict, disabled: bool) -> str | None:
    if disabled:
        return None
    import wandb

    id_path = output_dir(config) / "train/wandb_run_id.txt"
    run_id = id_path.read_text().strip()
    run = wandb.init(project=config["wandb"]["project"], name=config["wandb"]["run_name"], id=run_id, resume="allow", dir=str(root))
    payload = {f"full_val/{branch}/{key}": value for branch, values in dynamics["one_step"].items() for key, value in values.items()}
    for horizon, values in dynamics["horizons"].items():
        for branch in ("vector", "token"):
            payload.update({f"horizon/{horizon}/{branch}/{key}": value for key, value in values[branch].items()})
    for index, path in enumerate(artifacts["contact_sheets"]):
        payload[f"turn_reconstruction/run_{index:02d}"] = wandb.Image(path)
    run.log(payload, step=int(config["training"]["steps"]) + 1)
    url = run.url
    run.finish()
    return url


def run(config_path: Path, checkpoint: Path, device: torch.device, no_wandb: bool) -> dict[str, Any]:
    config, started = load_config(config_path), time.perf_counter()
    root, state_cache = output_dir(config) / "eval", output_dir(config) / "state_cache/val"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"evaluation output is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    heads = MatchedWMHeads.load_checkpoint(checkpoint, device).to(device).eval()
    dynamics = evaluate_full_dynamics(heads, state_cache, device, batch_size=int(config["training"]["eval_batch_size"]))
    _atomic_json(root / "dynamics_metrics.json", dynamics)
    batch = _turn_batch(config, state_cache)
    adapter = load_frozen_state_adapter(Path(config["inputs"]["encoder_checkpoint"]), device)
    cfm = load_proven_cfm(Path(config["inputs"]["cfm_checkpoint"]), device)
    render = config["reconstruction"]
    images, noise = render_turn_comparison(batch, heads, adapter, cfm, device, steps=int(render["sample_steps"]), cfg_scale=float(render["cfg_scale"]), chunk_size=int(render["chunk_size"]), seed=int(config["seed"]))
    artifacts = write_turn_artifacts(batch, images, root / "turns", seed=int(config["seed"]), steps=int(render["sample_steps"]), cfg_scale=float(render["cfg_scale"]), noise_fingerprint=noise)
    summary = {"status": "completed", "elapsed_s": time.perf_counter() - started, "checkpoint": str(checkpoint), "qwen_loaded": False, "loaded_modules": ["MatchedWMHeads", "StateToVisionTokens", "TokenConditionedFlowUNet"], "dynamics": dynamics, "artifacts": artifacts}
    summary["wandb_url"] = _wandb_upload(config, root, artifacts, dynamics, no_wandb)
    _atomic_json(root / "evaluation_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def _turn_batch(config: dict, state_cache: Path):
    positive_cache = Path(config["inputs"]["positive_cache"]) / "val"
    _manifest_gate(state_cache, "frozen_query_state", [8, 1024])
    _manifest_gate(positive_cache, "qwen_compressed_vision_positive", [16, 512])
    state, positive = load_state_records(state_cache), load_state_records(positive_cache)
    spec = Path(config["inputs"]["turn_spec"])
    return prepare_turn_batch(load_turn_spec(spec), state, positive)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args.config, args.checkpoint, torch.device(args.device), args.no_wandb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
