"""SFT2 training loop (Qwen latent WM + value head)."""

from __future__ import annotations

import contextlib
import csv
import json
import math
import random
import time
from functools import partial
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
from nimloth.backbone.qwen_tuning import configure_qwen_tuning, resolve_tune_modes, uses_lora
from nimloth.backbone.vision_ema import VisionEncoderEMA, resolve_vision_ema
from nimloth.training.common.schedules import qwen_lr_schedule, set_optimizer_group_lr
from nimloth.training.common.wandb_logging import log_train_step, log_val_epoch, maybe_init_wandb
from nimloth.training.sft2.checkpoint import (
    load_aux_checkpoint,
    load_lora_adapter_state,
    resolve_resume_checkpoint_dir,
    save_checkpoint,
)
from nimloth.training.sft2.cli import parse_sft2_args
from nimloth.training.sft2.dataset import (
    TrajectoryRecordDataset,
    TransitionQwenDataset,
    collate_packed_trajectory_batch,
    collate_trajectory_record_batch,
    collate_transition_batch,
)
from nimloth.training.sft2.evaluate import evaluate
from nimloth.training.sft2.loss import compute_combined_loss, wm_loss_weight_schedule
from nimloth.training.sft2.preprocess_cache import (
    CachedTransitionDataset,
    build_transition_preprocess_cache,
    collate_cached_transition_batch,
    unpack_transition_batch,
)
from nimloth.training.sft2.profiling import StepTimer
from nimloth.eval.rollout import val_rollout_success_rate
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.training.sft2.step import (
    compute_step_value_loss,
    compute_step_wm_loss,
    compute_trajectory_wm_loss,
)
from nimloth.training.sft2.trajectory_batching import assert_packed_batch
from nimloth.training.sft2.trajectory_once import forward_trajectory_once
from nimloth.training.sft2.trajectory_sampler import TrajectoryAwareBatchSampler
from nimloth.wm import LeWMConfig, LatentWMPredictor, StateProjector, ValueHead
from nimloth.wm.dataset import TransitionJsonlDataset, TransitionSample


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def _resolve_dataloader_workers(args) -> int:
    if args.dataloader_workers >= 0:
        return args.dataloader_workers
    return 4 if args.preprocess_cache_dir is not None else 0


def _prepare_transition_datasets(args, processor):
    train_samples = TransitionJsonlDataset(
        args.train_jsonl,
        max_records=args.max_train_records,
        success_only=args.success_only,
        value_gamma=args.value_gamma,
    ).samples
    val_samples = TransitionJsonlDataset(
        args.val_jsonl,
        max_records=args.max_val_records,
        value_gamma=args.value_gamma,
    ).samples

    if args.preprocess_cache_dir is None:
        if args.packed_forward:
            train_ds = TrajectoryRecordDataset(train_samples)
            val_ds = TrajectoryRecordDataset(val_samples)
            train_collate = collate_trajectory_record_batch
            val_collate = collate_trajectory_record_batch
        else:
            train_ds = TransitionQwenDataset.from_samples(train_samples)
            val_ds = TransitionQwenDataset.from_samples(val_samples)
            train_collate = collate_transition_batch
            val_collate = collate_transition_batch
        return train_ds, val_ds, train_collate, val_collate, train_samples, val_samples

    if args.packed_forward:
        cache_root = args.preprocess_cache_dir
        train_cache_dir = cache_root / "train_trajectory"
        val_cache_dir = cache_root / "val_trajectory"
        min_pixels = 3136
        build_kwargs = dict(
            model_path=args.model,
            processor=processor,
            max_length=args.max_length,
            max_pixels=args.max_pixels,
            min_pixels=min_pixels,
            preprocess_workers=args.preprocess_workers,
            force=args.force_rebuild_cache,
        )
        if is_main():
            from nimloth.training.sft2.preprocess_cache import build_trajectory_preprocess_cache

            build_trajectory_preprocess_cache(
                jsonl_path=args.train_jsonl,
                cache_dir=train_cache_dir,
                max_records=args.max_train_records,
                success_only=args.success_only,
                **build_kwargs,
            )
            build_trajectory_preprocess_cache(
                jsonl_path=args.val_jsonl,
                cache_dir=val_cache_dir,
                max_records=args.max_val_records,
                success_only=False,
                **build_kwargs,
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        from nimloth.training.sft2.preprocess_cache import CachedTrajectoryDataset

        return (
            CachedTrajectoryDataset(train_cache_dir, train_samples),
            CachedTrajectoryDataset(val_cache_dir, val_samples),
            collate_trajectory_record_batch,
            collate_trajectory_record_batch,
            train_samples,
            val_samples,
        )

    cache_root = args.preprocess_cache_dir
    train_cache_dir = cache_root / "train"
    val_cache_dir = cache_root / "val"
    min_pixels = 3136
    build_kwargs = dict(
        model_path=args.model,
        processor=processor,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        min_pixels=min_pixels,
        preprocess_workers=args.preprocess_workers,
        force=args.force_rebuild_cache,
    )
    if is_main():
        build_transition_preprocess_cache(
            jsonl_path=args.train_jsonl,
            cache_dir=train_cache_dir,
            max_records=args.max_train_records,
            success_only=args.success_only,
            **build_kwargs,
        )
        build_transition_preprocess_cache(
            jsonl_path=args.val_jsonl,
            cache_dir=val_cache_dir,
            max_records=args.max_val_records,
            success_only=False,
            **build_kwargs,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    pad_token_id = processor.tokenizer.pad_token_id
    cached_collate = partial(collate_cached_transition_batch, pad_token_id=pad_token_id)
    return (
        CachedTransitionDataset(train_cache_dir, train_samples),
        CachedTransitionDataset(val_cache_dir, val_samples),
        cached_collate,
        cached_collate,
        train_samples,
        val_samples,
    )


def _unpack_train_batch(batch, processor, max_length: int, *, packed_forward: bool, pad_token_id: int):
    if isinstance(batch, dict) and "transition_samples" in batch:
        return (
            batch["items"],
            None,
            None,
            batch["transition_samples"],
            batch.get("full_enc"),
        )
    items, enc, next_rows = unpack_transition_batch(
        batch,
        processor,
        max_length,
        pad_token_id=pad_token_id,
    )
    return items, enc, next_rows, None, None


def _no_sync_if_needed(modules, *, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    stack = contextlib.ExitStack()
    for module in modules:
        no_sync = getattr(module, "no_sync", None)
        if no_sync is not None:
            stack.enter_context(no_sync())
    return stack


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

    resume_ckpt_dir: Path | None = None
    if args.resume:
        resume_ckpt_dir = resolve_resume_checkpoint_dir(args.output_dir, args.resume_from)
    resume_state_path = (
        resume_ckpt_dir / "training_state.pt" if resume_ckpt_dir is not None else None
    )
    resume_adapter = resume_ckpt_dir / "adapter_config.json" if resume_ckpt_dir is not None else None

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
                    "resume_from": str(resume_ckpt_dir) if resume_ckpt_dir is not None else None,
                    "init_model": str(args.model),
                    "wm_predictor_checkpoint": str(args.wm_predictor_checkpoint) if args.wm_predictor_checkpoint else None,
                    "output_dir": str(args.output_dir),
                    "packed_forward": args.packed_forward,
                    "trajectory_aware_batching": args.trajectory_aware_batching,
                }
            )
        )

    if args.packed_forward and not args.allow_approx_trajectory_once:
        raise ValueError(
            "--packed-forward trajectory-once is not semantic-equivalent for default multi-image Qwen-VL SFT2; "
            "pass --allow-approx-trajectory-once only for research/profiling."
        )
    if args.trajectory_aware_batching and args.packed_forward:
        raise ValueError("--trajectory-aware-batching is for legacy per-prefix batching; do not combine with --packed-forward")

    train_ds, val_ds, train_collate, val_collate, train_samples, val_samples = _prepare_transition_datasets(
        args, processor
    )
    dataloader_workers = _resolve_dataloader_workers(args)
    loader_kwargs: dict = {
        "num_workers": dataloader_workers,
        "pin_memory": True,
    }
    if dataloader_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_sampler = None
    train_batch_sampler = None
    val_sampler = None
    if args.trajectory_aware_batching:
        train_batch_sampler = TrajectoryAwareBatchSampler(
            train_samples,
            batch_size=args.batch_size,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
    elif world > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    if world > 1:
        val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)

    if train_batch_sampler is not None:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            collate_fn=train_collate,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=1 if args.packed_forward else args.batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None and not args.packed_forward,
            collate_fn=train_collate,
            **loader_kwargs,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=1 if args.packed_forward else args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=val_collate,
        **loader_kwargs,
    )

    qwen_gpu_stride = int(__import__("os").environ.get("NIMLOTH_DDP_GPU_STRIDE", "1"))
    qwen_pair_parallel = qwen_gpu_stride > 1 and torch.cuda.is_available()
    qwen_load_kwargs = {}
    if qwen_pair_parallel:
        primary_idx = int(str(device).split(":")[-1])
        pair = [primary_idx + i for i in range(qwen_gpu_stride)]
        qwen_load_kwargs = {
            "device_map": "auto",
            "max_memory": {i: "74GiB" for i in pair} | {"cpu": "64GiB"},
            "low_cpu_mem_usage": True,
        }
        if is_main():
            print(json.dumps({"qwen_pair_parallel": True, "gpu_stride": qwen_gpu_stride, "rank0_pair": pair}))
    base_model_path = args.model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        **qwen_load_kwargs,
    )
    if args.gradient_checkpointing:
        # DDP + reentrant activation checkpointing can fire reducer hooks for the
        # same trainable Qwen parameter twice when the training step uses Qwen
        # hidden states in several downstream losses.  The non-reentrant variant
        # is the PyTorch-recommended checkpointing mode for DDP.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.resize_token_embeddings(len(processor.tokenizer))

    if args.resume and resume_state_path is not None and resume_state_path.exists() and resume_adapter.exists():
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
    elif (
        args.resume
        and resume_state_path is not None
        and resume_state_path.exists()
        and (resume_ckpt_dir / "config.json").exists()
    ):
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
            **qwen_load_kwargs,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.resize_token_embeddings(len(processor.tokenizer))
        model = configure_qwen_tuning(model, args)
    else:
        model = configure_qwen_tuning(model, args)
        if is_main():
            print(json.dumps({"init": "configured_tuning", "base_model_path": str(base_model_path)}))

    if not qwen_pair_parallel:
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
    aux_device = device
    if qwen_pair_parallel:
        device_map = getattr(model, "hf_device_map", {}) or {}
        mapped = device_map.get("lm_head") or device_map.get("model.language_model.norm")
        if mapped is not None:
            aux_device = torch.device(f"cuda:{mapped}")
    if qwen_pair_parallel:
        wm_predictor = wm_predictor.to(aux_device)
    state_proj = StateProjector(hidden_size, wm_predictor.emb_dim).to(device=aux_device, dtype=model_dtype)
    value_head = ValueHead(wm_predictor.emb_dim).to(device=aux_device, dtype=model_dtype)

    if args.resume and resume_state_path is not None and resume_state_path.exists():
        load_aux_checkpoint(resume_ckpt_dir, state_proj, wm_predictor, value_head, device)

    if world > 1:
        # Every trainable branch is exercised on every rank (terminal-only WM
        # batches use dummy aux forwards), so unused-parameter graph traversal is
        # unnecessary and interacts badly with multi-forward/checkpointed steps.
        if qwen_pair_parallel:
            model = DDP(model, device_ids=None, output_device=None, find_unused_parameters=False)
        else:
            model = DDP(model, device_ids=[int(str(device).split(":")[-1])], output_device=int(str(device).split(":")[-1]), find_unused_parameters=False)
        if uses_lora(args) and not qwen_pair_parallel:
            model._set_static_graph()
        aux_idx = int(str(aux_device).split(":")[-1])
        state_proj = DDP(state_proj, device_ids=[aux_idx], output_device=aux_idx, find_unused_parameters=False)
        value_head = DDP(value_head, device_ids=[aux_idx], output_device=aux_idx, find_unused_parameters=False)
        if train_wm_predictor:
            wm_predictor = DDP(wm_predictor, device_ids=[aux_idx], output_device=aux_idx, find_unused_parameters=False)

    vision_ema: VisionEncoderEMA | None = None
    if vision_ema_enabled:
        vision_ema = VisionEncoderEMA(decay=args.vision_ema_decay)
        vision_ema.reset(model)
        ema_path = resume_ckpt_dir / "vision_ema.pt" if resume_ckpt_dir is not None else None
        if args.resume and ema_path is not None and ema_path.is_file():
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
    if args.resume and resume_state_path is not None and resume_state_path.exists():
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

    step_timer = StepTimer(enabled=args.step_timing, log_interval=args.step_timing_interval)
    pad_token_id = processor.tokenizer.pad_token_id

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

        train_iter = iter(train_loader)
        micro_idx = 0
        while True:
            t0 = step_timer.start("dataloader")
            try:
                batch_samples = next(train_iter)
            except StopIteration:
                break
            step_timer.stop("dataloader", t0)
            micro_idx += 1
            sync_gradients = (micro_idx % args.grad_accum == 0) or (micro_idx == num_micro_batches)
            with _no_sync_if_needed(ddp_modules, enabled=world > 1 and not sync_gradients and not qwen_pair_parallel):
                t0 = step_timer.start("batch_prep")
                items, enc, next_enc_rows, transition_samples, full_enc = _unpack_train_batch(
                    batch_samples,
                    processor,
                    args.max_length,
                    packed_forward=args.packed_forward,
                    pad_token_id=pad_token_id,
                )
                step_timer.stop("batch_prep", t0)

                t0 = step_timer.start("current_forward")
                if args.packed_forward:
                    assert transition_samples is not None
                    assert_packed_batch(transition_samples)
                    traj = forward_trajectory_once(
                        model,
                        transition_samples,
                        processor,
                        token_id_map,
                        device,
                        max_length=args.max_length,
                        vision_ema=vision_ema,
                        full_enc=full_enc,
                    )
                    latent_hidden = traj.current_latents
                    lm_loss = traj.lm_loss
                else:
                    latent_hidden, lm_loss = extract_qwen_latents(model, enc, token_id_map, device)
                step_timer.stop("current_forward", t0)

                lambda_wm = wm_loss_weight_schedule(
                    global_step,
                    total_steps,
                    start=args.lambda_wm_start,
                    end=args.lambda_wm_end,
                )

                t0 = step_timer.start("next_forward")
                if args.packed_forward:
                    wm_loss, wm_metrics = compute_trajectory_wm_loss(
                        items,
                        latent_hidden,
                        traj.next_latents,
                        state_proj,
                        wm_predictor,
                        device,
                    )
                else:
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
                        next_enc_rows=next_enc_rows,
                        pad_token_id=pad_token_id,
                    )
                step_timer.stop("next_forward", t0)

                t0 = step_timer.start("value_loss")
                value_loss, value_metrics = compute_step_value_loss(
                    latent_hidden,
                    items,
                    state_proj,
                    value_head,
                    device,
                    rank_margin=args.value_rank_margin,
                    lambda_rank=args.value_rank_lambda,
                )
                step_timer.stop("value_loss", t0)

                t0 = step_timer.start("loss_combine")
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
                step_timer.stop("loss_combine", t0)

                t0 = step_timer.start("backward")
                (loss / args.grad_accum).backward()
                step_timer.stop("backward", t0)
            accum.update(metrics)
            micro += 1

            if sync_gradients:
                t0 = step_timer.start("optimizer")
                _optimizer_step(epoch, lambda_wm=lambda_wm)
                step_timer.stop("optimizer", t0)
                step_timer.on_optimizer_step(global_step=global_step, epoch=epoch)

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
            pad_token_id=pad_token_id,
            packed_forward=args.packed_forward,
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
