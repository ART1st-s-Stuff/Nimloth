"""Argparse CLI for SFT2 training."""

from __future__ import annotations

import argparse
from pathlib import Path

from nimloth.latent import LATENT_QUERY_MODES, query_labels_are_masked, resolve_latent_query_mode
from nimloth.config.sft2 import apply_sft2_yaml_defaults


def build_sft2_arg_parser(config_path: Path | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SFT2: latent WM + value head alignment")
    applied_config = apply_sft2_yaml_defaults(ap, config_path)

    ap.add_argument(
        "--config",
        type=Path,
        default=applied_config,
        help="YAML config for defaults (configs/training/sft2/latent_wm_value.yaml)",
    )
    ap.add_argument("--model", type=Path, required=True, help="Init HF dir (SFT1 hf_merged or resume best/)")
    ap.add_argument("--wm-predictor-checkpoint", type=Path, default=None)
    ap.add_argument(
        "--objective",
        choices=("latent", "dino_grid"),
        default="latent",
    )
    ap.add_argument("--dino-grid-cache", type=Path, default=None)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--distributed-strategy", choices=("ddp", "fsdp"), default="ddp")
    ap.add_argument("--fsdp-wrap-granularity", choices=("linear", "block"), default="linear",
                    help="FSDP handle grouping; block groups decoder/vision block internals.")
    ap.add_argument("--activation-offload", action=argparse.BooleanOptionalAction, default=False,
                    help="Store forward autograd saved tensors on CPU; computation and backward remain on GPU.")
    ap.add_argument("--stop-after-steps", type=int, default=0,
                    help="Stop at this absolute optimizer step with a partial resumable checkpoint; 0 disables.")
    ap.add_argument("--diagnose-outcome-gradients", action="store_true",
                    help="Measure first-batch outcome versus WM+DINO predictor gradients without an update.")
    ap.add_argument("--batch-size", type=int, default=2, help="Complete trajectories per rank and microbatch.")
    ap.add_argument("--grad-accum", type=int, default=4, help="Complete-trajectory microbatches per optimizer update.")
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
    ap.add_argument("--grid-size", type=int, default=4)
    ap.add_argument("--grid-predictor-kind", choices=("direct", "residual"), default="direct")
    ap.add_argument("--grid-wm-depth", type=int, default=6)
    ap.add_argument("--grid-wm-heads", type=int, default=16)
    ap.add_argument("--grid-wm-dim-head", type=int, default=64)
    ap.add_argument("--grid-wm-mlp-dim", type=int, default=2048)
    ap.add_argument("--grid-wm-dropout", type=float, default=0.1)
    ap.add_argument(
        "--history-size",
        type=int,
        default=1,
        help=(
            "Stage3 predicts each window from one current state; H must be 1."
        ),
    )
    ap.add_argument(
        "--prediction-horizon",
        type=int,
        default=1,
        help=(
            "Autoregressive SFT2 target length T. T>1 uses consecutive actions "
            "from the recorded rollout and currently requires history_size=1."
        ),
    )
    ap.add_argument(
        "--latent-token-count",
        type=int,
        default=1,
        help="Number of latent query tokens per step.",
    )
    ap.add_argument(
        "--latent-query-mode",
        choices=LATENT_QUERY_MODES,
        default=None,
        help="inject: framework supplies query slots; generate: model emits query token IDs.",
    )
    ap.add_argument(
        "--query-tune",
        choices=("freeze", "adapter", "selected_rows"),
        default="freeze",
        help="Freeze Query, tune an additive adapter, or train selected input/head rows.",
    )
    ap.add_argument("--query-lr", type=float, default=5e-5)
    ap.add_argument("--protocol-lr", type=float, default=2e-5)
    ap.add_argument("--max-train-records", type=int, default=-1)
    ap.add_argument("--max-val-records", type=int, default=-1)
    ap.add_argument("--max-val-batches", type=int, default=-1)
    ap.add_argument("--diagnostic-steps", type=int, nargs="+", default=[])
    ap.add_argument("--diagnostic-dir", type=Path, default=None)
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="Load the configured model/checkpoint and evaluate without updates or checkpoint writes.",
    )
    ap.add_argument(
        "--feature-export-dir",
        type=Path,
        default=None,
        help="With --eval-only, export full Stage3 DINO diagnostic grids for offline rendering.",
    )
    ap.add_argument(
        "--frozen-wm-cache-dir",
        type=Path,
        default=None,
        help=(
            "With --eval-only, export each complete trajectory's fixed Stage2 and DINO grids "
            "once for the standalone frozen-WM diagnostic."
        ),
    )
    ap.add_argument(
        "--frozen-wm-cache-split",
        choices=("train", "eval"),
        default=None,
        help="Dataset loader exported by --frozen-wm-cache-dir; must be explicit.",
    )
    ap.add_argument("--success-only", action="store_true", help="Train on successful rollouts only")
    ap.add_argument("--lambda-ce", type=float, default=1.0)
    ap.add_argument("--lambda-dino", type=float, default=0.5)
    ap.add_argument("--lambda-value", type=float, default=1.0)
    ap.add_argument("--outcome-head", action="store_true")
    ap.add_argument("--outcome-eval-dir", type=Path, default=None)
    ap.add_argument("--lambda-outcome", type=float, default=0.0)
    ap.add_argument("--outcome-head-lr", type=float, default=1e-4)
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
        "--checkpoint-metric",
        choices=("val_wm_mse",),
        default="val_wm_mse",
        help="Model-derived validation metric used to select the best checkpoint.",
    )
    ap.add_argument(
        "--preprocess-cache-dir",
        type=Path,
        default=None,
        help="Disk cache for transition prefix processor outputs (enables DataLoader workers).",
    )
    ap.add_argument(
        "--preprocess-cache-processor-source",
        type=Path,
        default=None,
        help=(
            "Original model path recorded by a required prebuilt cache. Use only "
            "when model weights were re-exported without changing processor files."
        ),
    )
    ap.add_argument("--preprocess-workers", type=int, default=4, help="Workers for building preprocess cache.")
    ap.add_argument(
        "--preprocess-cache-image-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="On-disk compact image tensor dtype; bfloat16 matches GPU Qwen visual dtype.",
    )
    ap.add_argument("--preprocess-cache-image-shard-size", type=int, default=128)
    ap.add_argument("--preprocess-cache-transition-shard-size", type=int, default=256)
    ap.add_argument("--preprocess-cache-shard-lru", type=int, default=2)
    ap.add_argument(
        "--require-prebuilt-cache",
        action="store_true",
        help="Refuse to build cache inside the GPU training job.",
    )
    ap.add_argument("--force-rebuild-cache", action="store_true")
    ap.add_argument(
        "--dataloader-workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1: 0 without cache, 4 with cache).",
    )
    ap.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per persistent DataLoader worker when cache is enabled.",
    )
    ap.add_argument(
        "--step-timing",
        action="store_true",
        help="Log cumulative sampled per-section step timings (profiling only).",
    )
    ap.add_argument(
        "--step-timing-interval",
        type=int,
        default=50,
        help="Log every N profiled optimizer updates when --step-timing is set.",
    )
    ap.add_argument(
        "--step-timing-sample-interval",
        type=int,
        default=1,
        help="Profile the first local update and every N updates thereafter; must be positive.",
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
    ap.add_argument("--checkpoint-latest-only", action="store_true", default=False,
                    help="Keep only the newest complete resumable checkpoint across epochs and steps; record best metrics without separate best weights.")
    ap.add_argument("--deduplicate-epoch-checkpoints", action="store_true", default=False,
                    help="Hardlink immutable epoch/best/final artifacts in a fresh output directory.")
    ap.add_argument(
        "--checkpoint-keep-last",
        type=int,
        default=0,
        help="Keep only the last N step_NNNNNN checkpoints when step checkpointing is enabled (0 keeps all).",
    )
    ap.add_argument(
        "--batch-mode",
        choices=("trajectory_online_cache",),
        default="trajectory_online_cache",
        help=(
            "Process rank-local trajectory lanes in time order and reuse detached "
            "history states from their earlier current-step forwards."
        ),
    )
    # set_defaults must run after add_argument: registering an argument with an
    # explicit default otherwise overwrites the YAML value set earlier.
    apply_sft2_yaml_defaults(ap, applied_config)
    ap.set_defaults(config=applied_config)
    return ap


def parse_sft2_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, remaining = pre.parse_known_args(argv)
    ap = build_sft2_arg_parser(pre_args.config)
    args = ap.parse_args(remaining)
    args.latent_query_mode = resolve_latent_query_mode(
        args.latent_query_mode,
        default="inject",
    )
    if not 0 <= args.lambda_outcome < float("inf"):
        ap.error("lambda_outcome must be finite and nonnegative")
    if not 0 <= args.lambda_sigreg < float("inf"):
        ap.error("lambda_sigreg must be finite and nonnegative")
    if args.lambda_outcome > 0 and not args.outcome_head:
        ap.error("lambda_outcome > 0 requires --outcome-head")
    if args.outcome_head and args.objective != "dino_grid":
        ap.error("outcome head requires dino_grid objective")
    if args.grid_predictor_kind == "residual" and args.objective != "dino_grid":
        ap.error("residual grid predictor requires dino_grid objective")
    if bool(args.diagnostic_steps) != (args.diagnostic_dir is not None):
        ap.error("diagnostic-steps and diagnostic-dir must be supplied together")
    if args.diagnostic_steps:
        if args.objective != "dino_grid" or args.eval_only:
            ap.error("fixed-step diagnostics require dino_grid training")
        if min(args.diagnostic_steps) < 0 or len(set(args.diagnostic_steps)) != len(args.diagnostic_steps):
            ap.error("diagnostic steps must be distinct nonnegative updates")
        if args.diagnostic_dir.is_symlink():
            ap.error("diagnostic-dir must not be a symlink")
        if args.diagnostic_dir.exists() and not args.resume:
            ap.error("fresh diagnostic-dir must not already exist")
        if args.diagnostic_dir.resolve() == args.output_dir.resolve() or args.diagnostic_dir.resolve() in args.output_dir.resolve().parents:
            ap.error("diagnostic-dir must not contain the training output")
    if args.feature_export_dir is not None and not args.eval_only:
        ap.error("feature_export_dir requires --eval-only")
    if args.frozen_wm_cache_dir is not None and not args.eval_only:
        ap.error("frozen_wm_cache_dir requires --eval-only")
    if args.frozen_wm_cache_dir is not None and args.objective != "dino_grid":
        ap.error("frozen_wm_cache_dir requires the dino_grid objective")
    if args.frozen_wm_cache_dir is not None and args.frozen_wm_cache_split is None:
        ap.error("frozen_wm_cache_dir requires --frozen-wm-cache-split")
    if args.frozen_wm_cache_dir is None and args.frozen_wm_cache_split is not None:
        ap.error("frozen_wm_cache_split requires --frozen-wm-cache-dir")
    if args.frozen_wm_cache_dir is not None and args.max_val_batches > 0:
        ap.error("frozen_wm_cache_dir requires complete evaluation with max_val_batches=-1")
    if (
        args.frozen_wm_cache_dir is not None
        and args.frozen_wm_cache_split == "train"
        and args.max_train_records != -1
    ):
        ap.error("train frozen-WM export requires max_train_records=-1")
    if (
        args.frozen_wm_cache_dir is not None
        and args.frozen_wm_cache_split == "eval"
        and args.max_val_records != -1
    ):
        ap.error("eval frozen-WM export requires max_val_records=-1")
    if args.feature_export_dir is not None and args.frozen_wm_cache_dir is not None:
        ap.error("feature_export_dir and frozen_wm_cache_dir are mutually exclusive")
    if not 0 < args.outcome_head_lr < float("inf"):
        ap.error("outcome_head_lr must be finite and positive")
    if args.step_timing_sample_interval < 1:
        ap.error("step_timing_sample_interval must be positive")
    if args.history_size != 1:
        ap.error("trajectory-native Stage3 requires history_size=1")
    if args.stop_after_steps < 0:
        ap.error("stop_after_steps must be nonnegative")
    if args.diagnose_outcome_gradients and args.lambda_outcome <= 0:
        ap.error("outcome gradient diagnostic requires positive lambda_outcome")
    args.mask_latent_query_labels = query_labels_are_masked(args.latent_query_mode)
    return args
