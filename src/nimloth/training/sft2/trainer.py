"""SFT2 training loop (Qwen latent WM + value head)."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import torch
from nimloth.backbone.qwen25vl.loading import load_qwen_processor
from nimloth.backbone.qwen25vl.transition import QwenTransitionEncoder
from nimloth.latent import (
    query_labels_are_masked,
    resolve_latent_query_mode,
)
from nimloth.backbone.qwen25vl.tuning import resolve_tune_modes, uses_lora
from nimloth.backbone.qwen25vl.vision_ema import resolve_vision_ema
from nimloth.training.sft2.checkpoint import (
    SFT2CheckpointManager,
    resolve_resume_checkpoint_dir,
)
from nimloth.training.sft2.cli import parse_sft2_args
from nimloth.training.sft2.components import build_sft2_components
from nimloth.training.sft2.data.factory import build_data_bundle
from nimloth.training.sft2.algorithm import SFT2Algorithm
from nimloth.training.sft2.loop import (
    SFT2TrainingLoop,
    load_sft2_loop_state,
)
from nimloth.util.distributed import cleanup_dist, is_main, setup_dist
from nimloth.util.csv_log import CSVRecordWriter
from nimloth.util.wandb import init_wandb_run


def train_sft2(args=None) -> int:
    if args is None:
        args = parse_sft2_args()
    args.latent_token_count = int(getattr(args, "latent_token_count", 1))
    args.latent_query_mode = resolve_latent_query_mode(
        getattr(args, "latent_query_mode", None),
        getattr(args, "mask_latent_query_labels", None),
        default="inject",
    )
    args.mask_latent_query_labels = query_labels_are_masked(args.latent_query_mode)
    args.query_tune = str(getattr(args, "query_tune", "freeze"))
    args.query_lr = float(getattr(args, "query_lr", 5e-5))
    if args.query_tune not in {"freeze", "adapter"}:
        raise ValueError(f"query_tune must be freeze or adapter, got {args.query_tune!r}")
    if args.latent_token_count < 1:
        raise ValueError(f"--latent-token-count must be >= 1, got {args.latent_token_count}")

    llm_tune, vision_tune = resolve_tune_modes(args)
    if args.query_tune == "adapter" and uses_lora(args):
        raise ValueError("query_tune=adapter is not supported with LoRA tuning")
    vision_ema_enabled = resolve_vision_ema(args, vision_tune)
    train_wm_predictor = args.train_wm_predictor and not args.freeze_wm_predictor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rank, world, local_rank, device = setup_dist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = os.environ.get("WANDB_RUN_PREFIX", "")
    wandb_run = init_wandb_run(
        rank=rank,
        output_dir=args.output_dir,
        enabled=not args.no_wandb,
        default_project="nimloth",
        run_name=args.wandb_run_name or f"{prefix}sft2-latentwm-value",
        config=vars(args),
        metric_definitions=(
            ("global_step", None),
            ("train/*", "global_step"),
            ("epoch", None),
            ("val/*", "epoch"),
        ),
    )

    resume_ckpt_dir: Path | None = None
    if args.resume:
        resume_ckpt_dir = resolve_resume_checkpoint_dir(args.output_dir, args.resume_from)
    resume_state_path = (
        resume_ckpt_dir / "training_state.pt" if resume_ckpt_dir is not None else None
    )
    processor_bundle = load_qwen_processor(
        args.model,
        max_pixels=args.max_pixels,
        latent_token_count=args.latent_token_count,
    )
    processor = processor_bundle.processor
    added_special_token_count = processor_bundle.added_special_token_count
    token_id_map = processor_bundle.token_id_map

    if is_main():
        print(
            json.dumps(
                {
                    "llm_tune": llm_tune,
                    "vision_tune": vision_tune,
                    "vision_ema": vision_ema_enabled,
                    "vision_ema_decay": args.vision_ema_decay,
                    "train_wm_predictor": train_wm_predictor,
                    "resume": args.resume,
                    "resume_from": str(resume_ckpt_dir) if resume_ckpt_dir is not None else None,
                    "init_model": str(args.model),
                    "wm_predictor_checkpoint": str(args.wm_predictor_checkpoint) if args.wm_predictor_checkpoint else None,
                    "output_dir": str(args.output_dir),
                    "batch_mode": args.batch_mode,
                    "latent_token_count": args.latent_token_count,
                    "latent_query_mode": args.latent_query_mode,
                    "query_tune": args.query_tune,
                    "mask_latent_query_labels": args.mask_latent_query_labels,
                    "preprocess_cache_format": args.preprocess_cache_format,
                    "preprocess_cache_image_dtype": args.preprocess_cache_image_dtype,
                    "require_prebuilt_cache": args.require_prebuilt_cache,
                }
            )
        )

    data = build_data_bundle(args, processor, rank=rank, world_size=world)
    train_loader = data.train_loader
    val_loader = data.val_loader
    train_sampler = data.train_sampler
    train_batch_sampler = data.train_batch_sampler

    components = build_sft2_components(
        args,
        processor,
        token_id_map,
        added_special_token_count,
        resume_ckpt_dir,
        device=device,
        world_size=world,
        train_wm_predictor=train_wm_predictor,
        vision_ema_enabled=vision_ema_enabled,
    )
    model = components.model
    state_proj = components.state_proj
    wm_predictor = components.wm_predictor
    value_head = components.value_head
    sigreg = components.sigreg
    vision_ema = components.vision_ema
    optimizer = components.optimizer
    base_model_path = components.base_model_path

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    qwen_warmup_steps = max(1, int(total_steps * args.qwen_lr_warmup_ratio))
    checkpoint_invariants = {
        "seed": int(args.seed),
        "world_size": int(world),
        "grad_accum": int(args.grad_accum),
        "latent_query_mode": args.latent_query_mode,
        "query_tune": args.query_tune,
        "train_micro_batches": int(len(train_loader)),
        "rng_schedule_version": "epoch_micro_rank_v1",
    }
    checkpoint_manager = SFT2CheckpointManager(
        output_dir=args.output_dir,
        model=model,
        state_proj=state_proj,
        processor=processor,
        wm_predictor=wm_predictor,
        value_head=value_head,
        vision_ema=vision_ema,
        optimizer=optimizer,
        training_invariants=checkpoint_invariants,
        lora=uses_lora(args),
        base_model_path=base_model_path,
        llm_tune=llm_tune,
        vision_tune=vision_tune,
        latent_query_mode=args.latent_query_mode,
        query_tune=args.query_tune,
    )

    log_writer = CSVRecordWriter(
        args.output_dir / "train_step_log.csv",
        (
            "time",
            "epoch",
            "global_step",
            "total_loss",
            "wm_mse",
            "sigreg_loss",
            "value_total",
            "value_reg",
            "value_rank",
            "lm_ce",
            "lambda_wm",
            "lambda_sigreg",
            "qwen_lr",
            "val_wm_mse",
        ),
    )
    if is_main():
        log_writer.ensure_header()

    pad_token_id = processor.tokenizer.pad_token_id
    qwen_transition_encoder = QwenTransitionEncoder(
        model=model,
        processor=processor,
        token_id_map=token_id_map,
        device=device,
        max_length=args.max_length,
        pad_token_id=pad_token_id,
        latent_token_count=args.latent_token_count,
        mask_latent_query_labels=args.mask_latent_query_labels,
        vision_ema=vision_ema,
    )
    algorithm = SFT2Algorithm(
        qwen=qwen_transition_encoder,
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
        sigreg=sigreg,
        value_rank_margin=args.value_rank_margin,
        value_rank_weight=args.value_rank_lambda,
    )

    loop_state = load_sft2_loop_state(
        resume=args.resume,
        resume_state_path=resume_state_path,
        resume_checkpoint_dir=resume_ckpt_dir,
        optimizer=optimizer,
        training_invariants=checkpoint_invariants,
    )
    training_loop = SFT2TrainingLoop(
        args=args,
        rank=rank,
        world_size=world,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        train_batch_sampler=train_batch_sampler,
        components=components,
        algorithm=algorithm,
        checkpoint_manager=checkpoint_manager,
        log_writer=log_writer,
        wandb_run=wandb_run,
        state=loop_state,
        train_wm_predictor=train_wm_predictor,
        total_steps=total_steps,
        qwen_warmup_steps=qwen_warmup_steps,
        llm_tune=llm_tune,
        vision_tune=vision_tune,
    )
    training_loop.run()
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_dist()
    return 0


def main() -> int:
    return train_sft2()
