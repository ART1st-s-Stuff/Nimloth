"""RL 训练命令行入口。

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

from nimloth.config.rl import load_rl_config, merge_rl_config_overrides


def build_rl_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Online RL training (WM predictor + value head)"
    )

    # 必填运行参数
    ap.add_argument("--config", type=Path, required=True, help="YAML config file")
    ap.add_argument("--model", type=Path, required=True,
                    help="Init HF dir (SFT1 hf_merged, SFT2 best/, or HF model name)")
    ap.add_argument("--output-dir", type=Path, required=True)

    # 模型微调方式
    ap.add_argument("--llm-tune", choices=("freeze", "lora", "full"), default="freeze")
    ap.add_argument("--vision-tune", choices=("freeze", "lora", "full"), default="freeze")
    ap.add_argument("--vision-ema", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--vision-ema-decay", type=float, default=0.999)
    ap.add_argument("--lora", action="store_true",
                    help="Shorthand: --llm-tune lora --vision-tune freeze")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    # 模型加载
    ap.add_argument("--attn-implementation", default="flash_attention_2")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--max-pixels", type=int, default=602112)

    # WM warm-start
    ap.add_argument("--wm-checkpoint", type=Path, default=None,
                    help="Warm-start WM predictor checkpoint dir")
    ap.add_argument("--state-proj-checkpoint", type=Path, default=None,
                    help="Warm-start StateProjector checkpoint (.pt file)")
    ap.add_argument("--value-head-checkpoint", type=Path, default=None,
                    help="Warm-start ValueHead checkpoint dir")

    # Rollout 数据来源
    ap.add_argument("--env-url", default=None,
                    help="VAGEN env server URL for online rollout collection")
    ap.add_argument("--use-jsonl-rollout", action="store_true",
                    help="Read trajectories from pre-existing JSONL (external rollout)")
    ap.add_argument("--jsonl-sources", type=Path, nargs="+", default=None,
                    help="JSONL 文件或目录列表，用于 JSONL rollout collector（与 --use-jsonl-rollout 配合）")
    ap.add_argument(
        "--eval-jsonl-sources",
        type=Path,
        nargs="+",
        default=None,
        help="独立的 validation JSONL 文件或目录",
    )

    # 日志
    ap.add_argument("--experiment-name", default=None,
                    help="Optional run name used for wandb if WANDB_RUN_NAME is not set")

    # 训练控制
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


def main(argv: list[str] | None = None) -> int:
    """解析参数、创建阶段组件并启动 RL 训练。"""
    import torch
    from nimloth.util.distributed import is_main
    from nimloth.rollout import (
        JSONLRolloutCollector,
        VAGENNavigationRolloutCollector,
    )
    from nimloth.training.rl.trainer import train_rl
    from nimloth.wm.predictor import LatentWMPredictor
    from nimloth.wm.state_proj import StateProjector
    from nimloth.wm.value_head import ValueHead

    args = parse_rl_args(argv)
    config = load_rl_config(args.config)
    config = merge_rl_config_overrides(args, config)

    output_dir = Path(args.output_dir).resolve()

    if is_main():
        print(json.dumps({
            "config_summary": {
                "llm_tune": args.llm_tune,
                "vision_tune": args.vision_tune,
                "lora": args.lora,
                "resume": args.resume,
                "config": config.to_dict(),
                "output_dir": str(output_dir),
            }
        }, indent=2, default=str))

    # 创建 WM 组件
    from nimloth.wm.lewm import LeWMConfig

    if args.wm_checkpoint is not None:
        # Load from checkpoint — use its config to avoid shape mismatches
        wm_predictor = LatentWMPredictor.load_checkpoint(args.wm_checkpoint)
        if is_main():
            print(json.dumps({"warm_start": "wm_predictor", "source": str(args.wm_checkpoint),
                              "history_size": wm_predictor.config.history_size}))
    else:
        wm_config = LeWMConfig(
            emb_dim=config.predictor.emb_dim,
            history_size=config.predictor.history_size,
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

    # 创建 rollout 数据源
    if args.env_url:
        train_datasets = config.rollout.train_datasets
        if not train_datasets:
            raise ValueError(
                "direct env training requires rollout.train_datasets"
            )
        train_collector = VAGENNavigationRolloutCollector(
            qwen_model=None,  # filled in by trainer after model loading
            processor=None,   # filled in by trainer
            env_url=args.env_url,
            device=None,      # filled in by trainer
            temperature=config.rollout.temperature,
            top_p=config.rollout.top_p,
            eval_sets=train_datasets,
            split="train",
        )
        eval_collector = None
        if config.validation.enabled:
            if not config.rollout.eval_datasets:
                raise ValueError(
                    "validation.enabled requires rollout.eval_datasets"
                )
            eval_collector = VAGENNavigationRolloutCollector(
                qwen_model=None,
                processor=None,
                env_url=args.env_url,
                device=None,
                seed_offset=1_000_000,
                temperature=config.rollout.temperature,
                top_p=config.rollout.top_p,
                eval_sets=config.rollout.eval_datasets,
                split="eval",
            )
        if is_main():
            print(json.dumps({"rollout_mode": "env", "env_url": args.env_url,
                              "train_datasets": train_datasets,
                              "eval_datasets": config.rollout.eval_datasets,
                              "temperature": config.rollout.temperature,
                              "top_p": config.rollout.top_p}))
    else:
        jsonl_sources = args.jsonl_sources or [
            Path(path) for path in config.rollout.jsonl_train_sources
        ]
        if not jsonl_sources:
            raise ValueError(
                "JSONL rollout mode requires --jsonl-sources or "
                "rollout.jsonl_train_sources"
            )
        train_collector = JSONLRolloutCollector(sources=jsonl_sources)
        eval_collector = None
        if config.validation.enabled:
            eval_sources = args.eval_jsonl_sources or [
                Path(path) for path in config.rollout.jsonl_eval_sources
            ]
            if not eval_sources:
                raise ValueError(
                    "validation.enabled requires --eval-jsonl-sources or "
                    "rollout.jsonl_eval_sources"
                )
            eval_collector = JSONLRolloutCollector(sources=eval_sources)
        if is_main():
            print(json.dumps({"rollout_mode": "jsonl",
                              "num_sources": len(jsonl_sources),
                              "sources": [str(s) for s in jsonl_sources],
                              "eval_sources": [
                                  str(source)
                                  for source in (args.eval_jsonl_sources or [])
                              ]}))
    # 启动阶段训练
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
        train_collector=train_collector,
        eval_collector=eval_collector,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
