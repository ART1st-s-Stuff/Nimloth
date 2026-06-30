"""Argparse CLI for SFT2 training."""

from __future__ import annotations

import argparse
from pathlib import Path

from nimloth.training.common.config import apply_yaml_defaults


def build_sft2_arg_parser(config_path: Path | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SFT2: latent WM + value head alignment")
    applied_config = apply_yaml_defaults(ap, config_path)

    ap.add_argument(
        "--config",
        type=Path,
        default=applied_config,
        help="YAML config for defaults (configs/training/sft2/latent_wm_value.yaml)",
    )
    ap.add_argument("--model", type=Path, required=True, help="Init HF dir (SFT1 hf_merged or resume best/)")
    ap.add_argument("--wm-predictor-checkpoint", type=Path, default=None)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr-qwen-start", type=float, default=1e-8)
    ap.add_argument("--lr-qwen-peak", type=float, default=5e-7)
    ap.add_argument("--qwen-lr-warmup-ratio", type=float, default=0.15)
    ap.add_argument("--state-proj-lr", type=float, default=1e-4)
    ap.add_argument("--wm-predictor-lr", type=float, default=3e-4)
    ap.add_argument("--value-head-lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--emb-dim", type=int, default=1024)
    ap.add_argument("--max-train-records", type=int, default=-1)
    ap.add_argument("--max-val-records", type=int, default=-1)
    ap.add_argument("--max-val-batches", type=int, default=-1)
    ap.add_argument("--success-only", action="store_true", help="Train on successful rollouts only")
    ap.add_argument("--lambda-ce", type=float, default=1.0)
    ap.add_argument("--lambda-value", type=float, default=1.0)
    ap.add_argument("--value-rank-margin", type=float, default=0.1)
    ap.add_argument("--value-rank-lambda", type=float, default=1.0)
    ap.add_argument("--value-gamma", type=float, default=1.0)
    ap.add_argument("--lambda-sigreg", type=float, default=0.1)
    ap.add_argument("--sigreg-num-proj", type=int, default=1024)
    ap.add_argument("--sigreg-knots", type=int, default=17)
    ap.add_argument("--lambda-wm-start", type=float, default=0.1)
    ap.add_argument("--lambda-wm-end", type=float, default=1.0)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint dir to resume from (under output-dir if relative). "
        "Default with --resume: latest epoch_* or best/ by saved step.",
    )
    ap.add_argument("--train-wm-predictor", action="store_true", default=True)
    ap.add_argument("--freeze-wm-predictor", action="store_true")
    ap.add_argument("--llm-tune", choices=("freeze", "lora", "full"), default="freeze")
    ap.add_argument("--vision-tune", choices=("freeze", "lora", "full"), default="full")
    ap.add_argument("--vision-ema", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--vision-ema-decay", type=float, default=0.999)
    ap.add_argument("--lora", action="store_true", help="Shorthand: --llm-tune lora --vision-tune freeze")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument(
        "--early-stop-metric",
        choices=("val_success_rate", "val_wm_mse"),
        default="val_success_rate",
    )
    ap.add_argument(
        "--preprocess-cache-dir",
        type=Path,
        default=None,
        help="Disk cache for transition prefix processor outputs (enables DataLoader workers).",
    )
    ap.add_argument("--preprocess-workers", type=int, default=4, help="Workers for building preprocess cache.")
    ap.add_argument("--force-rebuild-cache", action="store_true")
    ap.add_argument(
        "--dataloader-workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1: 0 without cache, 4 with cache).",
    )
    ap.add_argument(
        "--step-timing",
        action="store_true",
        help="Log rolling-average per-section step timings (profiling only).",
    )
    ap.add_argument(
        "--step-timing-interval",
        type=int,
        default=50,
        help="Log step timings every N optimizer steps when --step-timing is set.",
    )
    ap.add_argument(
        "--checkpoint-interval-minutes",
        type=float,
        default=20.0,
        help="Save resumable latest checkpoint every N minutes during training.",
    )
    ap.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=0,
        help="Save resumable step_NNNNNN checkpoints every N optimizer steps (0 disables).",
    )
    ap.add_argument(
        "--checkpoint-keep-last",
        type=int,
        default=0,
        help="Keep only the last N step_NNNNNN checkpoints when step checkpointing is enabled (0 keeps all).",
    )
    ap.add_argument(
        "--trajectory-aware-batching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Batch consecutive prefixes from the same trajectory as independent rows. "
            "This improves padding/next-target locality without full-trajectory forward."
        ),
    )
    ap.add_argument(
        "--full-trajectory-batching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Each micro-batch is one complete trajectory (all transitions for one record). "
            "Qwen still sees per-prefix independent rows (NOT packed-forward). "
            "This lets SIGReg access the full trajectory's projected embeddings "
            "while respecting Qwen-VL prefix non-invariance. "
            "Micro-batch size = trajectory length (variable). "
            "Do NOT combine with --packed-forward."
        ),
    )
    ap.add_argument(
        "--max-steps-per-trajectory",
        type=int,
        default=8,
        help=(
            "When --full-trajectory-batching is enabled, chunk records longer "
            "than this number of steps into multiple micro-batches.  Default 8 "
            "keeps GPU memory bounded while benefiting SIGReg with long runs."
        ),
    )
    ap.add_argument(
        "--packed-forward",
        action="store_true",
        help="Use full-trajectory single forward (research-only; not semantic-equivalent for default Qwen-VL SFT2).",
    )
    ap.add_argument(
        "--allow-approx-trajectory-once",
        action="store_true",
        help="Explicitly allow non-equivalent trajectory-once packed forward for research/profiling only.",
    )
    return ap


def parse_sft2_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, remaining = pre.parse_known_args(argv)
    ap = build_sft2_arg_parser(pre_args.config)
    return ap.parse_args(remaining)
