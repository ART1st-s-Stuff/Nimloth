"""SFT1/SFT2 命令行、YAML 默认值与阶段参数验证。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from nimloth.latent import (
    LATENT_QUERY_MODES,
    query_labels_are_masked,
    resolve_latent_query_mode,
)

from .config import sft1_yaml_defaults
from .convergence import ConvergencePolicy
from .loss import validate_action_weight


def parse_args(argv: list[str] | None = None, *, stage: str = "format"):
    if stage not in {"format", "query"}:
        raise ValueError(f"unsupported early training stage: {stage}")
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", type=Path, default=None)
    probed, _ = config_probe.parse_known_args(argv)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=probed.config)
    if stage == "query":
        ap.add_argument("--dino-cache-root", type=Path, required=True)
        ap.add_argument("--grid-size", type=int, default=4)
        ap.add_argument("--projector-hidden-dim", type=int, default=2048)
        ap.add_argument("--weight-lm", type=float, default=1.0)
        ap.add_argument("--weight-dino", type=float, default=1.0)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--until-converged", action="store_true")
    ap.add_argument(
        "--max-optimizer-steps", type=int, default=None,
        help="Pause with a resume checkpoint and exit 75 at this absolute optimizer step.",
    )
    ap.add_argument("--convergence-min-epochs", type=int)
    ap.add_argument("--convergence-patience-epochs", type=int)
    ap.add_argument("--convergence-min-relative-improvement", type=float)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--distributed-strategy", choices=("ddp", "fsdp"), default="ddp")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--action-token-loss-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument(
        "--embedding-lr",
        type=float,
        default=None,
        help="LR for embed_tokens and lm_head (default: same as --lr).",
    )
    ap.add_argument("--projector-lr", type=float, default=None, help="Stage2 projector LR (default: --lr).")
    ap.add_argument("--embedding-master-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=20000)
    if stage == "query":
        ap.add_argument(
            "--latent-token-count",
            type=int,
            default=int(os.environ.get("LATENT_TOKEN_COUNT", "1")),
            help="Number of latent query tokens per action block; 1 keeps legacy SFT1 behavior.",
        )
        ap.add_argument(
            "--latent-query-mode",
            choices=LATENT_QUERY_MODES,
            default=None,
            help="inject: framework supplies query slots; generate: model emits query token IDs.",
        )
        ap.add_argument(
            "--mask-latent-query-labels",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Deprecated compatibility alias: true=inject, false=generate.",
        )
    ap.add_argument("--max-train-records", type=int, default=-1)
    ap.add_argument("--max-val-records", type=int, default=-1)
    ap.add_argument("--max-val-batches", type=int, default=-1)
    ap.add_argument("--max-images-per-record", type=int, default=-1)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--save-initial-checkpoint", action="store_true",
        help="Save exact epoch_000 model before the first update for paired evaluation.",
    )
    ap.add_argument(
        "--resume-save-steps",
        type=int,
        default=10,
        help="Publish an atomic resume checkpoint every N completed optimizer steps.",
    )
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=602112,
        help="Cap vision tokens per image (default ~768*28*28, lower than Qwen2.5-VL factory max).",
    )
    ap.add_argument(
        "--min-pixels",
        type=int,
        default=3136,
        help="Minimum pixels per image (~4*28*28).",
    )
    ap.add_argument(
        "--format-eval-samples",
        type=int,
        default=32,
        help="Val samples for Nimloth format correctness each epoch.",
    )
    ap.add_argument("--wandb-run-name", default=None, help="Optional wandb run name.")
    ap.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging and dataset upload.",
    )
    ap.add_argument(
        "--lora",
        action="store_true",
        help="Train LoRA adapters (+ embed/lm_head), freeze base weights.",
    )
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules (language model blocks).",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes for cached tensors.",
    )
    ap.add_argument(
        "--prefetch-factor", type=int, default=2, help="DataLoader prefetch per worker."
    )
    ap.add_argument(
        "--preprocess-workers",
        type=int,
        default=8,
        help="CPU processes for one-time preprocess cache build (rank 0).",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Root dir for preprocess cache (default: <output-dir>/preprocess_cache).",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable preprocess cache and use online collate.",
    )
    ap.add_argument(
        "--cache-only",
        action="store_true",
        help="Build/validate preprocess cache and exit before model load.",
    )
    ap.add_argument(
        "--require-prebuilt-cache",
        action="store_true",
        help="Refuse to build cache inside the GPU training job.",
    )
    ap.add_argument(
        "--rebuild-cache", action="store_true", help="Force rebuild preprocess cache."
    )
    ap.add_argument(
        "--cache-pixel-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="On-disk cached pixel dtype; bfloat16 matches the GPU visual encoder dtype.",
    )
    if probed.config is not None:
        ap.set_defaults(**sft1_yaml_defaults(probed.config, stage=stage))
    if stage == "format" and any(
        name in os.environ for name in ("LATENT_TOKEN_COUNT", "NIMLOTH_LATENT_TOKEN_COUNT", "LATENT_QUERY_MODE", "MASK_LATENT_QUERY_LABELS")
    ):
        raise ValueError("stage1 format supervision does not accept latent/query environment overrides")
    if stage == "query" and os.environ.get("LATENT_QUERY_MODE"):
        ap.set_defaults(latent_query_mode=os.environ["LATENT_QUERY_MODE"])
    if stage == "query":
        ap.set_defaults(no_cache=True)
        if probed.config is None:
            ap.set_defaults(latent_token_count=16)
    # argparse required flags do not recognize YAML defaults by themselves.
    for action in ap._actions:
        if action.required and action.default is not None:
            action.required = False
    args = ap.parse_args(argv)
    if args.projector_lr is not None and (stage != "query" or not 0 < args.projector_lr < float("inf")):
        raise ValueError("projector_lr requires Stage2 and a finite positive value")
    if args.embedding_master_dtype not in ("bfloat16", "float32"):
        raise ValueError("unsupported embedding master dtype")
    if args.embedding_master_dtype == "float32" and not (args.lora and args.distributed_strategy == "fsdp"):
        raise ValueError("FP32 embedding masters require LoRA and FSDP")
    if args.max_optimizer_steps is not None and args.max_optimizer_steps < 1:
        raise ValueError("--max-optimizer-steps must be positive")
    args.action_token_loss_weight = validate_action_weight(args.action_token_loss_weight)
    if stage != "format" and args.action_token_loss_weight != 1:
        raise ValueError("action token loss weighting is supported only for format stage1")
    policy_values = (
        args.convergence_min_epochs, args.convergence_patience_epochs,
        args.convergence_min_relative_improvement,
    )
    if args.until_converged:
        if args.epochs is not None:
            raise ValueError("--until-converged cannot be combined with epochs (CLI or YAML)")
        if any(value is None for value in policy_values):
            raise ValueError("--until-converged requires all three convergence policy parameters")
        ConvergencePolicy(*policy_values)
        if args.max_val_batches is not None and args.max_val_batches > 0:
            raise ValueError("convergence requires the full validation loader")
    else:
        if any(value is not None for value in policy_values):
            raise ValueError("convergence policy requires --until-converged")
        if args.epochs is None:
            args.epochs = 20
        if args.epochs < 1:
            raise ValueError("epochs must be positive")
    if not 0 <= args.warmup_ratio <= 1:
        raise ValueError("warmup ratio must be in [0, 1]")
    query_config = None
    if stage == "query":
        from nimloth.training.sft.stage2.config import QueryAlignmentConfig

        query_config = QueryAlignmentConfig(
            grid_size=args.grid_size,
            projector_hidden_dim=args.projector_hidden_dim,
            weight_lm=args.weight_lm,
            weight_dino=args.weight_dino,
        )
        if args.cache_only or args.require_prebuilt_cache:
            raise ValueError(
                "query alignment uses online tokenization and its explicit DINO cache"
            )
        if args.latent_token_count != query_config.grid_tokens:
            raise ValueError(
                "query token count must equal the DINO spatial grid size squared"
            )
    if stage == "query":
        args.latent_query_mode = resolve_latent_query_mode(
            args.latent_query_mode, args.mask_latent_query_labels, default="inject"
        )
        args.mask_latent_query_labels = query_labels_are_masked(args.latent_query_mode)
        if args.latent_token_count < 1:
            raise ValueError("--latent-token-count must be >= 1")
    else:
        # None identifies the format stage; it is not a zero-slot query protocol.
        args.latent_token_count = None
        args.latent_query_mode = None
        args.mask_latent_query_labels = None
    if args.resume_save_steps < 1:
        raise ValueError("--resume-save-steps must be >= 1")

    return args, query_config
