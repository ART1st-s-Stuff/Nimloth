"""SFT2 training loop (Qwen latent WM + value head)."""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoProcessor

from nimloth.latent import (
    add_special_tokens,
    query_labels_are_masked,
    resolve_latent_query_mode,
    special_token_ids,
)
from nimloth.training.common.dist import cleanup_dist, is_main, setup_dist
from nimloth.training.common.metrics import MetricAccumulator
from nimloth.backbone.qwen25vl.tuning import resolve_tune_modes, uses_lora
from nimloth.backbone.qwen25vl.vision_ema import resolve_vision_ema
from nimloth.training.common.schedules import qwen_lr_schedule, set_optimizer_group_lr
from nimloth.training.common.wandb_logging import log_train_step, log_val_epoch, maybe_init_wandb
from nimloth.training.sft2.checkpoint import (
    SFT2CheckpointManager,
    read_checkpoint_step,
    resolve_resume_checkpoint_dir,
    resume_epoch_and_micro_step,
)
from nimloth.training.sft2.cli import parse_sft2_args
from nimloth.training.sft2.components import build_sft2_components
from nimloth.training.sft2.data.factory import build_data_bundle
from nimloth.training.sft2.engine import SFT2StepRunner
from nimloth.training.sft2.evaluate import evaluate
from nimloth.training.sft2.objectives import compute_combined_loss, wm_loss_weight_schedule
from nimloth.training.sft2.profiling import StepTimer
from nimloth.training.sft2.utils import (
    no_sync_if_needed,
    seed_training_micro_step,
)


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
    wandb_run = maybe_init_wandb(args)

    resume_ckpt_dir: Path | None = None
    if args.resume:
        resume_ckpt_dir = resolve_resume_checkpoint_dir(args.output_dir, args.resume_from)
    resume_state_path = (
        resume_ckpt_dir / "training_state.pt" if resume_ckpt_dir is not None else None
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    added_special_token_count = add_special_tokens(
        processor.tokenizer,
        latent_token_count=args.latent_token_count,
    )
    token_id_map = special_token_ids(processor.tokenizer, latent_token_count=args.latent_token_count)

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
    qwen_pair_parallel = components.qwen_pair_parallel
    ddp_static_graph = components.ddp_static_graph
    lambda_sigreg_val = args.lambda_sigreg

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
                    "sigreg_loss",
                    "value_total",
                    "value_reg",
                    "value_rank",
                    "lm_ce",
                    "lambda_wm",
                    "lambda_sigreg",
                    "qwen_lr",
                    "val_wm_mse",
                ]
            )

    global_step = 0
    best_val_wm_mse = float("inf")
    start_epoch = 1
    resume_micro_step = 0
    if args.resume and resume_state_path is not None and resume_state_path.exists():
        state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        global_step = int(state.get("step", 0))
        best_val_wm_mse = float(state.get("best_val_wm_mse", state.get("best_val", float("inf"))))
        saved_invariants = state.get("training_invariants")
        if saved_invariants is not None:
            mismatches = {
                key: (saved_invariants.get(key), value)
                for key, value in checkpoint_invariants.items()
                if saved_invariants.get(key) != value
            }
            if mismatches:
                raise ValueError(f"resume training invariants mismatch: {mismatches}")
        if "epoch" in state:
            start_epoch, resume_micro_step = resume_epoch_and_micro_step(state)
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
                        "resume_micro_step": resume_micro_step,
                        "best_val_wm_mse": best_val_wm_mse,
                    }
                )
            )

    def _prune_step_checkpoints() -> None:
        keep = int(getattr(args, "checkpoint_keep_last", 0) or 0)
        if keep <= 0:
            return
        ckpts = sorted(
            (
                (read_checkpoint_step(path), path)
                for path in args.output_dir.glob("step_*")
                if path.is_dir() and path.name.startswith("step_") and (path / "training_state.pt").is_file()
            ),
            key=lambda item: item[0],
        )
        for _, path in ckpts[:-keep]:
            shutil.rmtree(path, ignore_errors=True)

    def _optimizer_step(epoch: int, *, lambda_wm: float, lambda_sigreg: float) -> None:
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
                        avg.get("sigreg_loss", ""),
                        avg.get("value_total", ""),
                        avg.get("value_reg", ""),
                        avg.get("value_rank", ""),
                        avg.get("lm_ce", ""),
                        lambda_wm,
                        lambda_sigreg,
                        qwen_lr,
                        "",
                    ]
                )
            accum.reset()
            log_train_step(wandb_run, global_step, avg)

    step_timer = StepTimer(enabled=args.step_timing, log_interval=args.step_timing_interval)
    pad_token_id = processor.tokenizer.pad_token_id
    step_runner = SFT2StepRunner(
        model=model,
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
        processor=processor,
        token_id_map=token_id_map,
        device=device,
        max_length=args.max_length,
        pad_token_id=pad_token_id,
        latent_token_count=args.latent_token_count,
        mask_latent_query_labels=args.mask_latent_query_labels,
        vision_ema=vision_ema,
        sigreg_module=sigreg,
        value_rank_margin=args.value_rank_margin,
        value_rank_lambda=args.value_rank_lambda,
    )
    last_periodic_ckpt_time = time.monotonic()

    for epoch in range(start_epoch, args.epochs + 1):
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        elif train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        accum = MetricAccumulator()
        micro = 0

        num_micro_batches = len(train_loader)
        ddp_modules = [model, state_proj, value_head]
        if train_wm_predictor:
            ddp_modules.append(wm_predictor)
        # PyTorch 2.8 has an upstream DDP regression where static_graph=True
        # combined with no_sync() crashes in Reducer::finalize_backward before
        # the first optimizer step (expect_autograd_hooks_ assertion). Keep the
        # static graph required by repeated Qwen forwards/checkpointing, and
        # synchronize each accumulation micro-batch. All-reduce is linear, so
        # this preserves the accumulated gradient while trading extra comms for
        # correctness on the pinned runtime.
        use_ddp_no_sync = world > 1 and not qwen_pair_parallel and not ddp_static_graph
        if is_main() and world > 1 and args.grad_accum > 1 and not use_ddp_no_sync:
            print(
                json.dumps(
                    {
                        "ddp_gradient_accumulation": "sync_each_microbatch",
                        "reason": "torch_2_8_static_graph_no_sync_regression",
                    }
                )
            )

        train_iter = iter(train_loader)
        micro_idx = 0
        if epoch == start_epoch and resume_micro_step:
            if resume_micro_step > num_micro_batches:
                raise ValueError(
                    "checkpoint micro_step_in_epoch exceeds current DataLoader length: "
                    f"{resume_micro_step} > {num_micro_batches}"
                )
            if resume_micro_step % args.grad_accum != 0 and resume_micro_step != num_micro_batches:
                raise ValueError(
                    "partial-epoch checkpoint was not saved at an optimizer boundary: "
                    f"micro_step={resume_micro_step}, grad_accum={args.grad_accum}"
                )
            for _ in range(resume_micro_step):
                next(train_iter)
            micro_idx = resume_micro_step
            micro = resume_micro_step
            if is_main():
                print(
                    json.dumps(
                        {
                            "resume_data_position": {
                                "epoch": epoch,
                                "skipped_micro_batches": resume_micro_step,
                                "total_micro_batches": num_micro_batches,
                            }
                        }
                    )
                )

        while True:
            t0 = step_timer.start("dataloader")
            try:
                batch_samples = next(train_iter)
            except StopIteration:
                break
            step_timer.stop("dataloader", t0)
            micro_idx += 1
            seed_training_micro_step(args.seed, epoch, micro_idx, rank)
            sync_gradients = (micro_idx % args.grad_accum == 0) or (micro_idx == num_micro_batches)
            with no_sync_if_needed(ddp_modules, enabled=not sync_gradients and use_ddp_no_sync):
                t0 = step_timer.start("forward")
                step_output = step_runner.forward(batch_samples, training=True)
                step_timer.stop("forward", t0)

                lambda_wm = wm_loss_weight_schedule(
                    global_step,
                    total_steps,
                    start=args.lambda_wm_start,
                    end=args.lambda_wm_end,
                )
                t0 = step_timer.start("loss_combine")
                loss, metrics = compute_combined_loss(
                    wm_loss=step_output.wm_loss,
                    value_loss=step_output.value_loss,
                    lm_loss=step_output.lm_loss,
                    lambda_wm=lambda_wm if step_output.wm_loss is not None else 0.0,
                    sigreg_loss=step_output.sigreg_loss,
                    lambda_sigreg=lambda_sigreg_val,
                    lambda_value=args.lambda_value,
                    lambda_ce=args.lambda_ce,
                )
                metrics.update(step_output.metrics)
                step_timer.stop("loss_combine", t0)

                t0 = step_timer.start("backward")
                (loss / args.grad_accum).backward()
                step_timer.stop("backward", t0)
            accum.update(metrics)
            micro += 1

            if sync_gradients:
                t0 = step_timer.start("optimizer")
                _optimizer_step(epoch, lambda_wm=lambda_wm, lambda_sigreg=lambda_sigreg_val)
                step_timer.stop("optimizer", t0)
                step_timer.on_optimizer_step(global_step=global_step, epoch=epoch)

                should_save_step = bool(
                    args.checkpoint_interval_steps
                    and args.checkpoint_interval_steps > 0
                    and global_step % args.checkpoint_interval_steps == 0
                )
                should_save_latest = False
                if args.checkpoint_interval_minutes > 0:
                    if is_main() and (time.monotonic() - last_periodic_ckpt_time) >= args.checkpoint_interval_minutes * 60.0:
                        should_save_latest = True
                    if dist.is_available() and dist.is_initialized():
                        flag = torch.tensor([1 if should_save_latest else 0], device=device, dtype=torch.int32)
                        dist.broadcast(flag, src=0)
                        should_save_latest = bool(flag.item())
                if should_save_latest:
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    if is_main():
                        checkpoint_manager.save(
                            "latest",
                            step=global_step,
                            epoch=epoch,
                            best_val_wm_mse=best_val_wm_mse,
                            epoch_complete=False,
                            micro_step_in_epoch=micro_idx,
                        )
                        last_periodic_ckpt_time = time.monotonic()
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                if should_save_step:
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    if is_main():
                        checkpoint_manager.save(
                            f"step_{global_step:06d}",
                            step=global_step,
                            epoch=epoch,
                            best_val_wm_mse=best_val_wm_mse,
                            epoch_complete=False,
                            micro_step_in_epoch=micro_idx,
                        )
                        _prune_step_checkpoints()
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        val_metrics = evaluate(step_runner, val_loader, max_batches=args.max_val_batches)
        val_wm = val_metrics.get("wm_mse", float("inf"))

        if is_main():
            log_val_epoch(
                wandb_run,
                epoch,
                val_metrics,
                global_step=global_step,
            )
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        time.time(),
                        epoch,
                        global_step,
                        "",
                        val_metrics.get("wm_mse", ""),
                        val_metrics.get("sigreg_loss", ""),
                        val_metrics.get("value_total", ""),
                        val_metrics.get("value_reg", ""),
                        val_metrics.get("value_rank", ""),
                        "",
                        "",
                        "",
                        "",
                        val_metrics.get("wm_mse", ""),
                    ]
                )
            improved = val_wm < best_val_wm_mse
            if improved:
                best_val_wm_mse = val_wm
            checkpoint_manager.save(
                f"epoch_{epoch:03d}",
                step=global_step,
                epoch=epoch,
                best_val_wm_mse=best_val_wm_mse,
            )
            if improved:
                checkpoint_manager.save(
                    "best",
                    step=global_step,
                    epoch=epoch,
                    best_val_wm_mse=best_val_wm_mse,
                )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_metrics": val_metrics,
                        "best_val_wm_mse": best_val_wm_mse,
                        "checkpoint_metric": args.checkpoint_metric,
                        "llm_tune": llm_tune,
                        "vision_tune": vision_tune,
                    }
                )
            )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if is_main():
        checkpoint_manager.save(
            "final",
            step=global_step,
            epoch=args.epochs,
            best_val_wm_mse=best_val_wm_mse,
        )
    cleanup_dist()
    return 0


def main() -> int:
    return train_sft2()
