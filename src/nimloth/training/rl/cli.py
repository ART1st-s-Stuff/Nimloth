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
                    help="完整的 k=1 inject HF checkpoint；不能直接传 PEFT adapter 目录")
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
                    help="Resume from --output-dir/latest/")
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
    from nimloth.util.distributed import is_main
    from nimloth.environment.navigation import VAGENNavigationRolloutCollector
    from nimloth.rollout import JSONLRolloutCollector
    from nimloth.training.rl.trainer import train_rl

    args = parse_rl_args(argv)
    if args.env_url and args.use_jsonl_rollout:
        raise ValueError("--env-url and --use-jsonl-rollout are mutually exclusive")
    if not args.env_url and not args.use_jsonl_rollout:
        raise ValueError("choose --env-url or --use-jsonl-rollout explicitly")
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

    # 创建 rollout 数据源
    if args.env_url:
        train_datasets = config.rollout.train_datasets
        if not train_datasets:
            raise ValueError(
                "direct env training requires rollout.train_datasets"
            )
        train_collector = VAGENNavigationRolloutCollector(
            policy=None,      # trainer 加载完整 Agent 后绑定
            env_url=args.env_url,
            temperature=config.rollout.temperature,
            top_p=config.rollout.top_p,
            eval_sets=train_datasets,
            split="train",
            agent_config=config.agent,
        )
        eval_collector = None
        if config.validation.enabled:
            if not config.rollout.eval_datasets:
                raise ValueError(
                    "validation.enabled requires rollout.eval_datasets"
                )
            eval_collector = VAGENNavigationRolloutCollector(
                policy=None,
                env_url=args.env_url,
                seed_offset=1_000_000,
                temperature=config.rollout.temperature,
                top_p=config.rollout.top_p,
                eval_sets=config.rollout.eval_datasets,
                split="eval",
                agent_config=config.agent,
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
        train_collector=train_collector,
        eval_collector=eval_collector,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
