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
    ap.add_argument("--jsonl-sources", type=Path, nargs="+", default=None,
                    help="JSONL 文件或目录列表，用于 JSONL rollout collector（与 --use-jsonl-rollout 配合）")

    # ---- Logging ------------------------------------------------------------
    ap.add_argument("--experiment-name", default=None,
                    help="Optional run name used for wandb if WANDB_RUN_NAME is not set")

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


def load_state_projector_for_rl(
    checkpoint: Path,
    *,
    qwen_hidden_dim: int,
    lewm_emb_dim: int,
    latent_token_count: int,
):
    """Rebuild StateProjector from its checkpoint widths and validate k."""

    import torch
    from nimloth.wm.state_proj import StateProjector

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    first_weight = state.get("net.net.0.weight")
    final_weight = state.get("net.net.3.weight")
    if first_weight is None or final_weight is None:
        raise ValueError("unrecognized StateProjector checkpoint layout")
    projector_hidden_dim, input_dim = map(int, first_weight.shape)
    output_dim, final_input_dim = map(int, final_weight.shape)
    expected_input_dim = int(qwen_hidden_dim) * int(latent_token_count)
    if input_dim != expected_input_dim:
        raise ValueError(
            "StateProjector input dim does not match policy protocol: "
            f"checkpoint={input_dim}, expected={expected_input_dim} "
            f"(k={latent_token_count}, hidden={qwen_hidden_dim})"
        )
    if final_input_dim != projector_hidden_dim or output_dim != int(lewm_emb_dim):
        raise ValueError(
            "StateProjector output layout does not match WM: "
            f"hidden={projector_hidden_dim}, final_input={final_input_dim}, "
            f"output={output_dim}, wm={lewm_emb_dim}"
        )
    module = StateProjector(
        qwen_hidden_dim=qwen_hidden_dim,
        lewm_emb_dim=lewm_emb_dim,
        projector_hidden_dim=projector_hidden_dim,
        latent_token_count=latent_token_count,
    )
    module.load_state_dict(state)
    return module


def main(argv: list[str] | None = None) -> int:
    """Parse args, load config, build modules, and launch RL training."""
    import torch
    from transformers import AutoConfig
    from nimloth.training.common.dist import is_main
    from nimloth.training.rl.rollout import (
        JSONLRolloutCollector,
        VAGENRolloutCollector,
        validate_rl_policy_protocol,
    )
    from nimloth.training.rl.trainer import train_rl
    from nimloth.wm.predictor import LatentWMPredictor
    from nimloth.wm.state_proj import StateProjector
    from nimloth.wm.value_head import ValueHead

    args = parse_rl_args(argv)
    config = load_rl_config(args.config)
    config = merge_config_overrides(args, config)

    output_dir = Path(args.output_dir).resolve()
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    latent_token_count = validate_rl_policy_protocol(model_config)
    text_config = getattr(model_config, "text_config", model_config)
    qwen_hidden_dim = int(getattr(text_config, "hidden_size"))

    if is_main():
        print(json.dumps({
            "config_summary": {
                "llm_tune": args.llm_tune,
                "vision_tune": args.vision_tune,
                "lora": args.lora,
                "resume": args.resume,
                "latent_token_count": latent_token_count,
                "latent_query_mode": "inject",
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
    if args.state_proj_checkpoint is not None:
        state_proj = load_state_projector_for_rl(
            args.state_proj_checkpoint,
            qwen_hidden_dim=qwen_hidden_dim,
            lewm_emb_dim=emb_dim,
            latent_token_count=latent_token_count,
        )
        if is_main():
            print(json.dumps({
                "warm_start": "state_proj",
                "source": str(args.state_proj_checkpoint),
                "latent_token_count": latent_token_count,
                "input_dim": state_proj.input_dim,
            }))
    else:
        state_proj = StateProjector(
            qwen_hidden_dim=qwen_hidden_dim,
            lewm_emb_dim=emb_dim,
            projector_hidden_dim=int(
                config.get("predictor", {}).get("projector_hidden_dim", 2048)
            ),
            latent_token_count=latent_token_count,
        )
    value_head = ValueHead(emb_dim=emb_dim)
    if args.value_head_checkpoint is not None:
        loaded_vh = ValueHead.load_checkpoint(args.value_head_checkpoint, emb_dim=emb_dim)
        value_head.load_state_dict(loaded_vh.state_dict())
        if is_main():
            print(json.dumps({"warm_start": "value_head", "source": str(args.value_head_checkpoint)}))

    # --- Rollout collectors --------------------------------------------------
    validation_collector = None
    if args.env_url:
        from nimloth.training.rl.rollout import EnvRolloutCollector
        rl_cfg = config.get("rl", {})
        rollout_cfg = config.get("rollout", {})
        eval_sets = tuple(rollout_cfg.get("eval_sets", ()))
        if not eval_sets:
            raise ValueError(
                "direct env training requires rollout.eval_sets with explicit *_train datasets"
            )
        collector = EnvRolloutCollector(
            qwen_model=None,  # filled in by trainer after model loading
            processor=None,   # filled in by trainer
            env_url=args.env_url,
            device=None,      # filled in by trainer
            temperature=float(rl_cfg.get("temperature", 1.0)),
            top_p=float(rl_cfg.get("top_p", 1.0)),
            eval_sets=eval_sets,
            split="train",
            seed_offset=int(rollout_cfg.get("seed_offset", 0)),
            history_window=int(rollout_cfg.get("history_window", 4)),
            env_timeout=int(rollout_cfg.get("env_timeout", 180)),
            latent_token_count=latent_token_count,
        )
        if is_main():
            print(json.dumps({"rollout_mode": "env", "env_url": args.env_url,
                              "eval_sets": eval_sets,
                              "temperature": rl_cfg.get("temperature", 1.0),
                              "top_p": rl_cfg.get("top_p", 1.0),
                              "seed_offset": rollout_cfg.get("seed_offset", 0),
                              "history_window": rollout_cfg.get("history_window", 4),
                              "env_timeout": rollout_cfg.get("env_timeout", 180)}))
        validation_cfg = config.get("validation", {})
        if bool(validation_cfg.get("enabled", False)):
            heldout_sets = tuple(validation_cfg.get("eval_sets", ()))
            if not heldout_sets:
                raise ValueError(
                    "validation.enabled requires validation.eval_sets"
                )
            validation_collector = EnvRolloutCollector(
                qwen_model=None,
                processor=None,
                env_url=args.env_url,
                device=None,
                temperature=float(validation_cfg.get("temperature", 0.0)),
                top_p=float(validation_cfg.get("top_p", 1.0)),
                eval_sets=heldout_sets,
                split="validation",
                seed_offset=int(validation_cfg.get("seed_offset", 1)),
                history_window=int(
                    validation_cfg.get(
                        "history_window", rollout_cfg.get("history_window", 4)
                    )
                ),
                env_timeout=int(
                    validation_cfg.get(
                        "env_timeout", rollout_cfg.get("env_timeout", 180)
                    )
                ),
                latent_token_count=latent_token_count,
            )
            if is_main():
                print(json.dumps({
                    "validation_collector": "heldout_env",
                    "eval_sets": heldout_sets,
                    "seed_offset": validation_collector._base_seed_offset,
                    "temperature": validation_collector._temperature,
                    "envs": validation_cfg.get("envs", 16),
                    "interval": validation_cfg.get("interval", 50),
                    "baseline": validation_cfg.get("baseline", False),
                }))
    elif args.use_jsonl_rollout or (args.vagen_config is None and not args.env_url):
        cfg_sources = (
            config.get("rollout", {}).get("jsonl_sources")
            or config.get("rl", {}).get("jsonl_sources")
            or []
        )
        jsonl_sources = args.jsonl_sources or [Path(p) for p in cfg_sources]
        if not jsonl_sources:
            raise ValueError(
                "JSONL rollout mode requires --jsonl-sources or rollout.jsonl_sources in config"
            )
        collector = JSONLRolloutCollector(sources=jsonl_sources)
        if is_main():
            print(json.dumps({"rollout_mode": "jsonl",
                              "num_sources": len(jsonl_sources),
                              "sources": [str(s) for s in jsonl_sources]}))
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
            "note": "Qwen loading and rank-0 W&B initialization are handled inside train_rl()",
        }))

    return train_rl(
        args=args,
        config=config,
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
        collector=collector,
        output_dir=output_dir,
        validation_collector=validation_collector,
    )


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
