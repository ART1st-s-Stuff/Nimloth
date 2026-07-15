"""Run full-val and turn reconstruction for SFT2 dynamics dimensions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from nimloth.eval.dynamics_dim_ablation import evaluate_dynamics_dims, render_dynamics_dim_comparison, write_dynamics_dim_artifacts
from nimloth.eval.matched_wm_cli import load_turn_batch
from nimloth.eval.matched_wm_render import load_frozen_state_adapter
from nimloth.training.reconstruction.state_to_vision_tokens import load_proven_cfm
from nimloth.training.wm_heads.config import load_config, output_dir
from nimloth.wm.dynamics_dim_heads import DynamicsDimWMHeads


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _wandb(config: dict, root: Path, artifacts: dict, dynamics: dict, disabled: bool) -> str | None:
    if disabled:
        return None
    import wandb

    run_id = (output_dir(config) / "train/wandb_run_id.txt").read_text().strip()
    run = wandb.init(project=config["wandb"]["project"], name=config["wandb"]["run_name"], id=run_id, resume="allow", dir=str(root))
    payload = {f"full_val/{branch}/{key}": value for branch, values in dynamics["one_step"].items() for key, value in values.items()}
    for horizon, values in dynamics["horizons"].items():
        for branch in ("full", "factorized"):
            payload.update({f"horizon/{horizon}/{branch}/{key}": value for key, value in values[branch].items()})
    for index, path in enumerate(artifacts["contact_sheets"]):
        payload[f"dynamics_dim_turns/run_{index:02d}"] = wandb.Image(path)
    step = int(config["training"]["epochs"]) * 10000 + 1
    run.log(payload, step=step)
    url = run.url
    run.finish()
    return url


def run(config_path: Path, checkpoint: Path, device: torch.device, no_wandb: bool) -> dict[str, Any]:
    config, started = load_config(config_path), time.perf_counter()
    root, cache = output_dir(config) / "eval", Path(config["inputs"]["state_cache"]) / "val"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"dynamics-dimension eval output is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    heads = DynamicsDimWMHeads.load_checkpoint(checkpoint, device).to(device).eval()
    dynamics = evaluate_dynamics_dims(heads, cache, device, batch_size=int(config["training"]["eval_batch_size"]))
    _atomic_json(root / "dynamics_metrics.json", dynamics)
    batch = load_turn_batch(config, cache)
    adapter = load_frozen_state_adapter(Path(config["inputs"]["adapter_checkpoint"]), device)
    cfm = load_proven_cfm(Path(config["inputs"]["cfm_checkpoint"]), device)
    render = config["reconstruction"]
    images, noise = render_dynamics_dim_comparison(batch, heads, adapter, cfm, device, steps=int(render["sample_steps"]), cfg_scale=float(render["cfg_scale"]), chunk_size=int(render["chunk_size"]), seed=int(config["seed"]))
    artifacts = write_dynamics_dim_artifacts(batch, images, root / "turns", seed=int(config["seed"]), steps=int(render["sample_steps"]), cfg_scale=float(render["cfg_scale"]), noise_fingerprint=noise)
    summary = {"status": "completed", "elapsed_s": time.perf_counter() - started, "checkpoint": str(checkpoint), "qwen_loaded": False, "loaded_modules": ["DynamicsDimWMHeads", "StateToVisionTokens", "TokenConditionedFlowUNet"], "dynamics": dynamics, "artifacts": artifacts}
    summary["wandb_url"] = _wandb(config, root, artifacts, dynamics, no_wandb)
    _atomic_json(root / "evaluation_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


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
