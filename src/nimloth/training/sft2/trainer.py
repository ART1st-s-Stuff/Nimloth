"""SFT2 training loop (Qwen latent WM + value head)."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.common.config import merge_cli_over_yaml
from nimloth.training.common.dist import cleanup_dist, is_main, setup_dist
from nimloth.training.common.metrics import MetricAccumulator
from nimloth.training.common.qwen_batch import build_qwen_batch
from nimloth.backbone.qwen_tuning import configure_qwen_tuning, resolve_tune_modes, uses_lora
from nimloth.backbone.vision_ema import VisionEncoderEMA, resolve_vision_ema
from nimloth.training.common.schedules import qwen_lr_schedule, set_optimizer_group_lr
from nimloth.training.common.wandb_logging import log_train_step, log_val_epoch, maybe_init_wandb
from nimloth.training.sft2.checkpoint import load_aux_checkpoint, load_lora_adapter_state, save_checkpoint
from nimloth.training.sft2.cli import parse_sft2_args
from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch
from nimloth.training.sft2.evaluate import evaluate
from nimloth.training.sft2.loss import compute_combined_loss, wm_loss_weight_schedule
from nimloth.eval.rollout import val_rollout_success_rate
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.training.sft2.step import compute_step_value_loss, compute_step_wm_loss
from nimloth.wm import LeWMConfig, LatentWMPredictor, StateProjector, ValueHead


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def train_sft2(args=None) -> int:
    if args is None:
        args = parse_sft2_args()
    merge_cli_over_yaml(args, args.config)

    llm_tune, vision_tune = resolve_tune_modes(args)
    vision_ema_enabled = resolve_vision_ema(args, vision_tune)
    train_wm_predictor = args.train_wm_predictor and not args.freeze_wm_predictor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rank, world, local_rank, device = setup_dist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = maybe_init_wandb(args)

    resume_ckpt_dir = args.output_dir / "best"
    resume_state_path = resume_ckpt_dir / "training_state.pt"
    resume_adapter = resume_ckpt_dir / "adapter_config.json"

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

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
                    "init_model": str(args.model),
                    "wm_predictor_checkpoint": str(args.wm_predictor_checkpoint) if args.wm_predictor_checkpoint else None,
                    "output_dir": str(args.output_dir),
                }
            )
        )

    train_ds = TransitionQwenDataset(args.train_jsonl, max_records=args.max_train_records, success_only=args.success_only)
    val_ds = TransitionQwenDataset(args.val_jsonl, max_records=args.max_val_records)
    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_transition_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_transition_batch,
    )

    base_model_path = args.model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        # DDP + reentrant activation checkpointing can fire reducer hooks for the
        # same trainable Qwen parameter twice when the training step uses Qwen
        # hidden states in several downstream losses.  The non-reentrant variant
        # is the PyTorch-recommended checkpointing mode for DDP.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.resize_token_embeddings(len(processor.tokenizer))

    if args.resume and resume_state_path.exists() and resume_adapter.exists():
        saved = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        if not uses_lora(args):
            raise ValueError("--resume with LoRA adapter requires llm_tune and/or vision_tune lora")
        saved_base = saved.get("base_model_path")
        if saved_base:
            base_model_path = Path(saved_base)
        if is_main():
            print(json.dumps({"resume_lora_adapter": str(resume_ckpt_dir), "base_model_path": str(base_model_path)}))
        model = configure_qwen_tuning(model, args)
        load_lora_adapter_state(model, resume_ckpt_dir)
    elif args.resume and resume_state_path.exists() and (resume_ckpt_dir / "config.json").exists():
        if uses_lora(args):
            raise ValueError("cannot --resume full HF checkpoint with lora tuning")
        if is_main():
            print(json.dumps({"resume_full": str(resume_ckpt_dir)}))
        # Full-finetune checkpoints save the Qwen weights under best/.  Reload
        # them before constructing the optimizer, then re-apply tuning flags so
        # the trainable parameter set matches the saved optimizer groups.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            resume_ckpt_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.resize_token_embeddings(len(processor.tokenizer))
        model = configure_qwen_tuning(model, args)
    else:
        model = configure_qwen_tuning(model, args)
        if is_main():
            print(json.dumps({"init": "configured_tuning", "base_model_path": str(base_model_path)}))

    model.to(device)

    wm_cfg = LeWMConfig(emb_dim=args.emb_dim)
    if args.wm_predictor_checkpoint is not None:
        wm_predictor = LatentWMPredictor.load_checkpoint(args.wm_predictor_checkpoint, map_location=device).to(device)
    else:
        wm_predictor = LatentWMPredictor.create(wm_cfg).to(device)
    if not train_wm_predictor:
        for param in wm_predictor.parameters():
            param.requires_grad = False

    hidden_size = model.config.hidden_size
    model_dtype = next(model.parameters()).dtype
    state_proj = StateProjector(hidden_size, wm_predictor.emb_dim).to(device=device, dtype=model_dtype)
    value_head = ValueHead(wm_predictor.emb_dim).to(device=device, dtype=model_dtype)

    if args.resume and resume_state_path.exists():
        load_aux_checkpoint(resume_ckpt_dir, state_proj, wm_predictor, value_head, device)

    if world > 1:
        # Every trainable branch is exercised on every rank (terminal-only WM
        # batches use dummy aux forwards), so unused-parameter graph traversal is
        # unnecessary and interacts badly with multi-forward/checkpointed steps.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if uses_lora(args):
            model._set_static_graph()
        state_proj = DDP(state_proj, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        value_head = DDP(value_head, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if train_wm_predictor:
            wm_predictor = DDP(wm_predictor, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    vision_ema: VisionEncoderEMA | None = None
    if vision_ema_enabled:
        vision_ema = VisionEncoderEMA(decay=args.vision_ema_decay)
        vision_ema.reset(model)
        ema_path = resume_ckpt_dir / "vision_ema.pt"
        if args.resume and ema_path.is_file():
            loaded_ema = VisionEncoderEMA.load_checkpoint(ema_path, map_location=device)
            vision_ema.decay = loaded_ema.decay
            vision_ema.shadow = {k: v.to(device) for k, v in loaded_ema.shadow.items()}
        if is_main():
            print(json.dumps({"vision_ema": True, "shadow_params": len(vision_ema.shadow), "decay": vision_ema.decay}))

    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr_qwen_start, "name": "qwen"},
        {"params": state_proj.parameters(), "lr": args.state_proj_lr, "name": "state_proj"},
        {"params": value_head.parameters(), "lr": args.value_head_lr, "name": "value_head"},
    ]
    if train_wm_predictor:
        pred_params = wm_predictor.parameters() if not hasattr(wm_predictor, "module") else wm_predictor.module.parameters()
        param_groups.append({"params": list(pred_params), "lr": args.wm_predictor_lr, "name": "wm_predictor"})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    qwen_warmup_steps = max(1, int(total_steps * args.qwen_lr_warmup_ratio))

    log_path = args.output_dir / "train_step_log.csv"
    if is_main() and not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "time",
                    "epoch",
                    "global_step",
                    "total_loss",
                    "wm_mse",
                    "value_total",
                    "value_reg",
                    "value_rank",
                    "lm_ce",
                    "lambda_wm",
                    "qwen_lr",
                    "val_wm_mse",
                    "val_success_rate",
                ]
            )

    global_step = 0
    best_val_success_rate = -1.0
    best_val_wm_mse = float("inf")
    start_epoch = 1
    if args.resume and resume_state_path.exists():
        state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        global_step = int(state.get("step", 0))
        best_val_success_rate = float(state.get("best_val_success_rate", -1.0))
        best_val_wm_mse = float(state.get("best_val_wm_mse", state.get("best_val", float("inf"))))
        if "epoch" in state:
            start_epoch = int(state["epoch"]) + 1
        if state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if is_main():
            print(
                json.dumps(
                    {
                        "resume": True,
                        "resume_ckpt": str(resume_ckpt_dir),
                        "start_epoch": start_epoch,
                        "global_step": global_step,
                        "best_val_success_rate": best_val_success_rate,
                        "best_val_wm_mse": best_val_wm_mse,
                    }
                )
            )

    def _optimizer_step(epoch: int, *, lambda_wm: float) -> None:
        nonlocal global_step
        qwen_lr = qwen_lr_schedule(
            global_step,
            warmup_steps=qwen_warmup_steps,
            total_steps=total_steps,
            start_lr=args.lr_qwen_start,
            peak_lr=args.lr_qwen_peak,
        )
        set_optimizer_group_lr(optimizer, "qwen", qwen_lr)

        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            1.0,
        )
        optimizer.step()
        if vision_ema is not None:
            vision_ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if is_main():
            avg = accum.averages()
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        time.time(),
                        epoch,
                        global_step,
                        avg.get("total_loss", ""),
                        avg.get("wm_mse", ""),
                        avg.get("value_total", ""),
                        avg.get("value_reg", ""),
                        avg.get("value_rank", ""),
                        avg.get("lm_ce", ""),
                        lambda_wm,
                        qwen_lr,
                        "",
                        "",
                    ]
                )
            accum.reset()
            log_train_step(wandb_run, global_step, avg)

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        accum = MetricAccumulator()
        micro = 0

        for batch_samples in train_loader:
            items = batch_samples
            enc = build_qwen_batch(items, processor, args.max_length)
            latent_hidden, lm_loss = extract_qwen_latents(model, enc, token_id_map, device)
            lambda_wm = wm_loss_weight_schedule(
                global_step,
                total_steps,
                start=args.lambda_wm_start,
                end=args.lambda_wm_end,
            )

            wm_loss, wm_metrics = compute_step_wm_loss(
                model,
                items,
                latent_hidden,
                processor,
                token_id_map,
                device,
                state_proj,
                wm_predictor,
                args.max_length,
                vision_ema=vision_ema,
            )
            value_loss, value_metrics = compute_step_value_loss(
                latent_hidden,
                items,
                state_proj,
                value_head,
                device,
                rank_margin=args.value_rank_margin,
                lambda_rank=args.value_rank_lambda,
            )
            loss, metrics = compute_combined_loss(
                wm_loss=wm_loss,
                value_loss=value_loss,
                lm_loss=lm_loss,
                lambda_wm=lambda_wm if wm_loss is not None else 0.0,
                lambda_value=args.lambda_value,
                lambda_ce=args.lambda_ce,
            )
            metrics.update(wm_metrics)
            metrics.update(value_metrics)

            (loss / args.grad_accum).backward()
            accum.update(metrics)
            micro += 1

            if micro % args.grad_accum == 0:
                _optimizer_step(epoch, lambda_wm=lambda_wm)

        if micro % args.grad_accum != 0:
            _optimizer_step(epoch, lambda_wm=lambda_wm)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        val_metrics = evaluate(
            model,
            _unwrap(state_proj),
            _unwrap(wm_predictor),
            _unwrap(value_head),
            val_loader,
            processor,
            token_id_map,
            device,
            max_batches=args.max_val_batches,
            max_length=args.max_length,
            vision_ema=vision_ema,
        )
        val_wm = val_metrics.get("wm_mse", float("inf"))
        val_success = val_metrics.get("success_rate", 0.0)
        val_rollout_success = val_rollout_success_rate(args.val_jsonl, max_records=args.max_val_records)
        if is_main():
            val_metrics["val_rollout_success_rate"] = val_rollout_success

        if is_main():
            log_val_epoch(
                wandb_run,
                epoch,
                {
                    **val_metrics,
                    "rollout_success_rate": val_rollout_success,
                },
            )
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        time.time(),
                        epoch,
                        global_step,
                        "",
                        val_metrics.get("wm_mse", ""),
                        val_metrics.get("value_total", ""),
                        val_metrics.get("value_reg", ""),
                        val_metrics.get("value_rank", ""),
                        "",
                        "",
                        "",
                        val_metrics.get("wm_mse", ""),
                        val_rollout_success,
                    ]
                )
            save_checkpoint(
                model,
                state_proj,
                processor,
                args.output_dir / f"epoch_{epoch:03d}",
                wm_predictor=_unwrap(wm_predictor),
                value_head=_unwrap(value_head),
                vision_ema=vision_ema,
                optimizer=optimizer,
                step=global_step,
                epoch=epoch,
                best_val_success_rate=best_val_success_rate,
                best_val_wm_mse=best_val_wm_mse,
                lora=uses_lora(args),
                base_model_path=base_model_path,
                llm_tune=llm_tune,
                vision_tune=vision_tune,
            )
            improved = False
            if args.early_stop_metric == "val_success_rate":
                if val_rollout_success > best_val_success_rate:
                    best_val_success_rate = val_rollout_success
                    improved = True
            elif val_wm < best_val_wm_mse:
                best_val_wm_mse = val_wm
                improved = True
            if val_wm < best_val_wm_mse:
                best_val_wm_mse = val_wm
            if improved:
                save_checkpoint(
                    model,
                    state_proj,
                    processor,
                    args.output_dir / "best",
                    wm_predictor=_unwrap(wm_predictor),
                    value_head=_unwrap(value_head),
                    vision_ema=vision_ema,
                    optimizer=optimizer,
                    step=global_step,
                    epoch=epoch,
                    best_val_success_rate=best_val_success_rate,
                    best_val_wm_mse=best_val_wm_mse,
                    lora=uses_lora(args),
                    base_model_path=base_model_path,
                    llm_tune=llm_tune,
                    vision_tune=vision_tune,
                )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_metrics": val_metrics,
                        "val_rollout_success_rate": val_rollout_success,
                        "best_val_success_rate": best_val_success_rate,
                        "best_val_wm_mse": best_val_wm_mse,
                        "early_stop_metric": args.early_stop_metric,
                        "llm_tune": llm_tune,
                        "vision_tune": vision_tune,
                    }
                )
            )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if is_main():
        save_checkpoint(
            model,
            state_proj,
            processor,
            args.output_dir / "final",
            wm_predictor=_unwrap(wm_predictor),
            value_head=_unwrap(value_head),
            vision_ema=vision_ema,
            optimizer=optimizer,
            step=global_step,
            epoch=args.epochs,
            best_val_success_rate=best_val_success_rate,
            best_val_wm_mse=best_val_wm_mse,
            lora=uses_lora(args),
            base_model_path=base_model_path,
            llm_tune=llm_tune,
            vision_tune=vision_tune,
        )
    cleanup_dist()
    return 0


def main() -> int:
    return train_sft2()
