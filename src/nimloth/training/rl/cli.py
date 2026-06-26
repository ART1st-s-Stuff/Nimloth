"""CLI entry point for RL training.

Usage::

    python -m nimloth.training.rl.cli --config configs/training/rl/defaults.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def parse_rl_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online RL training (WM predictor + value head)")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Qwen model name or path")
    parser.add_argument("--wm-checkpoint", type=Path, default=None,
                        help="Warm-start WM predictor checkpoint dir")
    parser.add_argument("--state-proj-checkpoint", type=Path, default=None,
                        help="Warm-start StateProjector checkpoint (.pt file)")
    parser.add_argument("--value-head-checkpoint", type=Path, default=None,
                        help="Warm-start ValueHead checkpoint dir")
    parser.add_argument("--vagen-config", type=Path, default=None,
                        help="VAGEN YAML config for inline rollout (optional)")
    parser.add_argument("--vagen-checkpoint", type=Path, default=None,
                        help="VAGEN model checkpoint dir for inline rollout (optional)")
    parser.add_argument("--use-jsonl-rollout", action="store_true",
                        help="Read trajectories from pre-existing JSONL (external rollout)")
    return parser.parse_args(argv)


def load_rl_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file.  Returns the parsed dict."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    """Parse args, load config, build modules, and launch RL training."""
    from nimloth.training.common.config import setup_dist
    from nimloth.training.common.dist import is_main
    from nimloth.training.rl.rollout import JSONLRolloutCollector, VAGENRolloutCollector
    from nimloth.training.rl.trainer import train_rl
    from nimloth.wm.predictor import LatentWMPredictor
    from nimloth.wm.state_proj import StateProjector
    from nimloth.wm.value_head import ValueHead

    args = parse_rl_args(argv)
    config = load_rl_config(args.config)

    output_dir = args.output_dir or Path(config.get("training", {}).get("output_dir", "outputs/rl"))
    output_dir = Path(output_dir).resolve()

    if is_main():
        print(json.dumps({
            "config_summary": {
                "rl": config.get("rl", {}),
                "freeze": config.get("freeze", {}),
                "predictor": config.get("predictor", {}),
                "value_head": config.get("value_head", {}),
                "output_dir": str(output_dir),
            }
        }, indent=2, default=str))

    # --- WM modules -----------------------------------------------------------
    from nimloth.wm.lewm import LeWMConfig
    pred_cfg = config.get("predictor", {})
    wm_config = LeWMConfig(
        emb_dim=pred_cfg.get("emb_dim", 128),
        history_size=pred_cfg.get("history_size", 4),
    )
    wm_predictor = LatentWMPredictor.create(wm_config)
    state_proj = StateProjector(qwen_hidden_dim=2048, lewm_emb_dim=wm_config.emb_dim)
    value_head = ValueHead(emb_dim=wm_config.emb_dim)

    # --- warm-start from checkpoints ------------------------------------------
    if args.wm_checkpoint is not None:
        loaded = LatentWMPredictor.load_checkpoint(args.wm_checkpoint)
        wm_predictor.load_state_dict(loaded.state_dict())
        if is_main():
            print(json.dumps({"warm_start": "wm_predictor", "source": str(args.wm_checkpoint)}))
    if args.state_proj_checkpoint is not None:
        import torch
        state_proj.load_state_dict(torch.load(args.state_proj_checkpoint, map_location="cpu", weights_only=True))
        if is_main():
            print(json.dumps({"warm_start": "state_proj", "source": str(args.state_proj_checkpoint)}))
    if args.value_head_checkpoint is not None:
        loaded_vh = ValueHead.load_checkpoint(args.value_head_checkpoint, emb_dim=wm_config.emb_dim)
        value_head.load_state_dict(loaded_vh.state_dict())
        if is_main():
            print(json.dumps({"warm_start": "value_head", "source": str(args.value_head_checkpoint)}))

    # --- rollout collector ----------------------------------------------------
    if args.use_jsonl_rollout:
        collector = JSONLRolloutCollector()
        if is_main():
            print(json.dumps({"rollout_mode": "jsonl"}))
    elif args.vagen_config is not None and args.vagen_checkpoint is not None:
        collector = VAGENRolloutCollector(
            vagen_config_path=args.vagen_config,
            vagen_checkpoint_dir=args.vagen_checkpoint,
            output_root=output_dir / "rollouts",
        )
        if is_main():
            print(json.dumps({"rollout_mode": "vagen_inline"}))
    else:
        # Default: JSONL mode with trajectories pre-collected externally
        collector = JSONLRolloutCollector()
        if is_main():
            print(json.dumps({"rollout_mode": "jsonl", "info": "no --vagen-config, reading pre-collected JSONL"}))

    # --- Qwen model loading (deferred to trainer or explicit pre-load) --------
    # The trainer receives the Qwen model reference; for now we raise if the
    # user wants inline VAGEN rollout without a Qwen model.
    if not args.use_jsonl_rollout and args.vagen_config is None:
        if is_main():
            print(json.dumps({
                "error": "Inline VAGEN rollout requires --vagen-config and --vagen-checkpoint, "
                         "or use --use-jsonl-rollout for external rollout mode."
            }))
            return 1

    # For now, the Qwen model loading is handled externally (e.g., via VAGEN's
    # own model loading in the Slurm script).  The trainer's `train_rl` function
    # accepts the Qwen model as an argument.  In the current workflow, the
    # Slurm script loads Qwen and calls `train_rl` directly.
    if is_main():
        print(json.dumps({"status": "cli_ready", "note": "Qwen model loading delegated to Slurm script"}))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
