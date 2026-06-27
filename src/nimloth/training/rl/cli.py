"""CLI entry point for RL training.

Usage::

    python -m nimloth.training.rl.cli \
      --config configs/training/rl/defaults.yaml \
      --model Qwen/Qwen2.5-VL-3B-Instruct \
      --output-dir outputs/experiments/training/rl/<date>/<name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def build_rl_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Online RL training (WM predictor + value head)"
    )

    # ---- Required -----------------------------------------------------------
    ap.add_argument("--config", type=Path, required=True, help="YAML config file")
    ap.add_argument("--model", type=Path, required=True,
                    help="Init HF dir (SFT1 hf_merged, SFT2 best/, or HF model name)")
    ap.add_argument("--output-dir", type=Path, required=True)

    # ---- Tuning -------------------------------------------------------------
    ap.add_argument("--llm-tune", choices=("freeze", "lora", "full"), default="freeze")
    ap.add_argument("--vision-tune", choices=("freeze", "lora", "full"), default="freeze")
    ap.add_argument("--vision-ema", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--vision-ema-decay", type=float, default=0.999)
    ap.add_argument("--lora", action="store_true",
                    help="Shorthand: --llm-tune lora --vision-tune freeze")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    # ---- Model loading ------------------------------------------------------
    ap.add_argument("--attn-implementation", default="flash_attention_2")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--max-pixels", type=int, default=602112)

    # ---- WM warm-start ------------------------------------------------------
    ap.add_argument("--wm-checkpoint", type=Path, default=None,
                    help="Warm-start WM predictor checkpoint dir")
    ap.add_argument("--state-proj-checkpoint", type=Path, default=None,
                    help="Warm-start StateProjector checkpoint (.pt file)")
    ap.add_argument("--value-head-checkpoint", type=Path, default=None,
                    help="Warm-start ValueHead checkpoint dir")

    # ---- Rollout ------------------------------------------------------------
    ap.add_argument("--env-url", default=None,
                    help="VAGEN env server URL for online rollout collection")
    ap.add_argument("--vagen-config", type=Path, default=None,
                    help="VAGEN YAML config for inline rollout (optional)")
    ap.add_argument("--vagen-checkpoint", type=Path, default=None,
                    help="VAGEN model checkpoint dir for inline rollout (optional)")
    ap.add_argument("--use-jsonl-rollout", action="store_true",
                    help="Read trajectories from pre-existing JSONL (external rollout)")

    # ---- Training control ---------------------------------------------------
    ap.add_argument("--resume", action="store_true",
                    help="Resume from --output-dir/best/")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override seed from config")
    ap.add_argument("--rl-iterations", type=int, default=None,
                    help="Override rl.iterations from config")
    ap.add_argument("--rl-envs-per-iteration", type=int, default=None,
                    help="Override rl.envs_per_iteration from config")

    return ap


def parse_rl_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = build_rl_arg_parser()
    return ap.parse_args(argv)


def load_rl_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_config_overrides(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Apply CLI overrides on top of YAML config (in-place)."""
    rl_cfg = config.setdefault("rl", {})
    train_cfg = config.setdefault("training", {})
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.rl_iterations is not None:
        rl_cfg["iterations"] = args.rl_iterations
    if args.rl_envs_per_iteration is not None:
        rl_cfg["envs_per_iteration"] = args.rl_envs_per_iteration
    return config


def main(argv: list[str] | None = None) -> int:
    """Parse args, load config, build modules, and launch RL training."""
    import torch
    from nimloth.training.common.dist import is_main
    from nimloth.training.rl.rollout import JSONLRolloutCollector, VAGENRolloutCollector
    from nimloth.training.rl.trainer import train_rl
    from nimloth.wm.predictor import LatentWMPredictor
    from nimloth.wm.state_proj import StateProjector
    from nimloth.wm.value_head import ValueHead

    args = parse_rl_args(argv)
    config = load_rl_config(args.config)
    config = merge_config_overrides(args, config)

    output_dir = Path(args.output_dir).resolve()

    if is_main():
        print(json.dumps({
            "config_summary": {
                "llm_tune": args.llm_tune,
                "vision_tune": args.vision_tune,
                "lora": args.lora,
                "resume": args.resume,
                "rl": config.get("rl", {}),
                "freeze": config.get("freeze", {}),
                "predictor": config.get("predictor", {}),
                "value_head": config.get("value_head", {}),
                "output_dir": str(output_dir),
            }
        }, indent=2, default=str))

    # --- WM modules ----------------------------------------------------------
    from nimloth.wm.lewm import LeWMConfig

    if args.wm_checkpoint is not None:
        # Load from checkpoint — use its config to avoid shape mismatches
        wm_predictor = LatentWMPredictor.load_checkpoint(args.wm_checkpoint)
        if is_main():
            print(json.dumps({"warm_start": "wm_predictor", "source": str(args.wm_checkpoint),
                              "history_size": wm_predictor.config.history_size}))
    else:
        pred_cfg = config.get("predictor", {})
        wm_config = LeWMConfig(
            emb_dim=pred_cfg.get("emb_dim", 128),
            history_size=pred_cfg.get("history_size", 4),
        )
        wm_predictor = LatentWMPredictor.create(wm_config)

    emb_dim = wm_predictor.config.emb_dim
    state_proj = StateProjector(qwen_hidden_dim=2048, lewm_emb_dim=emb_dim)
    value_head = ValueHead(emb_dim=emb_dim)
    if args.state_proj_checkpoint is not None:
        state_proj.load_state_dict(
            torch.load(args.state_proj_checkpoint, map_location="cpu", weights_only=True)
        )
        if is_main():
            print(json.dumps({"warm_start": "state_proj", "source": str(args.state_proj_checkpoint)}))
    if args.value_head_checkpoint is not None:
        loaded_vh = ValueHead.load_checkpoint(args.value_head_checkpoint, emb_dim=emb_dim)
        value_head.load_state_dict(loaded_vh.state_dict())
        if is_main():
            print(json.dumps({"warm_start": "value_head", "source": str(args.value_head_checkpoint)}))

    # --- Rollout collector ---------------------------------------------------
    if args.env_url:
        from nimloth.training.rl.rollout import EnvRolloutCollector
        collector = EnvRolloutCollector(
            qwen_model=None,  # filled in by trainer after model loading
            processor=None,   # filled in by trainer
            env_url=args.env_url,
            device=None,      # filled in by trainer
        )
        if is_main():
            print(json.dumps({"rollout_mode": "env", "env_url": args.env_url}))
    elif args.use_jsonl_rollout or (args.vagen_config is None):
        collector = JSONLRolloutCollector()
        if is_main():
            print(json.dumps({"rollout_mode": "jsonl"}))
    else:
        collector = VAGENRolloutCollector(
            vagen_config_path=args.vagen_config,
            vagen_checkpoint_dir=args.vagen_checkpoint,
            output_root=output_dir / "rollouts",
        )
        if is_main():
            print(json.dumps({"rollout_mode": "vagen_inline"}))

    # --- Launch training -----------------------------------------------------
    if is_main():
        print(json.dumps({
            "status": "cli_ready",
            "note": "Qwen model loading handled inside train_rl() via configure_qwen_tuning",
        }))

    return train_rl(
        args=args,
        config=config,
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
        collector=collector,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
