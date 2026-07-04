"""Train RCDM on frozen Qwen visual-encoder image features.

This is an oracle-image-feature baseline for the SFT2 latent-state RCDM
visualization.  Qwen2.5-VL is frozen and used only to cache a pooled visual
feature for each target image.  Only the RCDM UNet is trained.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.rcdm.checkpoint import (
    init_ema_state,
    parse_ema_rates,
    save_training_checkpoint,
    unwrap_model,
    update_ema_state,
)
from nimloth.rcdm.config import RCDMConfig, create_model_and_diffusion, rcdm_config_from_args
from nimloth.rcdm.image_utils import image_to_diffusion_tensor
from nimloth.rcdm.qwen_vision_cache import (
    RCDMQwenVisionCacheDataset,
    build_rcdm_qwen_vision_cache,
    collate_rcdm_qwen_vision_cache_batch,
    qwen_vision_cache_ready,
)
from nimloth.training.common.dist import cleanup_dist, is_main, setup_dist
from nimloth.training.reconstruction.rcdm_sft2 import _maybe_init_wandb, _maybe_resume


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def _load_frozen_qwen_vision(args: argparse.Namespace, device: torch.device):
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.to(device)
    _freeze(model)
    return processor, model


def _build_metadata(args: argparse.Namespace, rcdm_config: RCDMConfig, cond_dim: int, world: int) -> dict[str, Any]:
    return {
        "task": "nimloth_qwen_vision_rcdm_reconstruction",
        "conditioning": {
            "source": "qwen_visual_encoder_mean_pool",
            "model": str(args.model),
            "image_role": args.image_role,
            "cond_dim": int(cond_dim),
            "pooling": "mean_over_visual_tokens",
        },
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "qwen_vision_cache_dir": str(args.qwen_vision_cache_dir) if args.qwen_vision_cache_dir else None,
        "rcdm_config": rcdm_config.to_metadata(),
        "train": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "batch_size_per_rank": args.batch_size,
            "effective_batch_size": args.batch_size * world,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "ema_rate": args.ema_rate,
            "world_size": world,
            "success_only": args.success_only,
            "target": f"{args.image_role}_image_from_qwen_visual_feature",
            "resume": bool(args.resume or args.resume_checkpoint is not None),
            "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
        },
        "wandb": {
            "project": args.wandb_project,
            "run_name": args.wandb_run_name,
            "id": args.wandb_id,
            "resume": args.wandb_resume,
        },
    }


def _batch_conditions_and_images(batch: dict[str, Any], *, device: torch.device, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    cond = batch["cond_emb"].to(device=device, dtype=torch.float32)
    images = torch.stack(
        [image_to_diffusion_tensor(path, image_size=image_size) for path in batch["target_image_path"]]
    ).to(device=device)
    return cond, images


@torch.no_grad()
def _evaluate_loss(*, model, diffusion, loader, device: torch.device, image_size: int, max_batches: int) -> float:
    model.eval()
    losses: list[float] = []
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        cond, images = _batch_conditions_and_images(batch, device=device, image_size=image_size)
        t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device)
        loss = diffusion.training_losses(model, images, t, model_kwargs={"feat": cond})["loss"].mean()
        losses.append(float(loss.detach().cpu()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def train_rcdm_qwen_vision(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_dist()
    try:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if torch.cuda.is_available():
            device = torch.device("cuda", rank % torch.cuda.device_count())
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        if args.qwen_vision_cache_dir is None:
            raise ValueError("--qwen-vision-cache-dir is required")
        train_cache_dir = args.qwen_vision_cache_dir / "train"
        val_cache_dir = args.qwen_vision_cache_dir / "val"
        cache_missing = not (qwen_vision_cache_ready(train_cache_dir) and qwen_vision_cache_ready(val_cache_dir))
        if args.build_qwen_vision_cache or cache_missing:
            processor, qwen_model = _load_frozen_qwen_vision(args, device)
            if is_main():
                build_rcdm_qwen_vision_cache(
                    jsonl_path=args.train_jsonl,
                    cache_dir=train_cache_dir,
                    split_name="train",
                    model_path=args.model,
                    processor=processor,
                    qwen_model=qwen_model,
                    device=device,
                    max_pixels=args.max_pixels,
                    min_pixels=args.min_pixels,
                    image_role=args.image_role,
                    max_records=args.max_train_records,
                    success_only=args.success_only,
                    batch_size=args.qwen_vision_cache_build_batch_size,
                    shard_size=args.qwen_vision_cache_shard_size,
                    compression=args.qwen_vision_cache_compression,
                    feature_dtype=args.qwen_vision_cache_dtype,
                    force=args.force_rebuild_qwen_vision_cache,
                )
                build_rcdm_qwen_vision_cache(
                    jsonl_path=args.val_jsonl,
                    cache_dir=val_cache_dir,
                    split_name="val",
                    model_path=args.model,
                    processor=processor,
                    qwen_model=qwen_model,
                    device=device,
                    max_pixels=args.max_pixels,
                    min_pixels=args.min_pixels,
                    image_role=args.image_role,
                    max_records=args.max_val_records,
                    success_only=args.success_only,
                    batch_size=args.qwen_vision_cache_build_batch_size,
                    shard_size=args.qwen_vision_cache_shard_size,
                    compression=args.qwen_vision_cache_compression,
                    feature_dtype=args.qwen_vision_cache_dtype,
                    force=args.force_rebuild_qwen_vision_cache,
                )
            if world > 1 and dist.is_available() and dist.is_initialized():
                dist.barrier()
            del qwen_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif not cache_missing:
            if is_main():
                print(json.dumps({"rcdm_qwen_vision_cache": "ready", "dir": str(args.qwen_vision_cache_dir)}))

        train_ds = RCDMQwenVisionCacheDataset(train_cache_dir)
        val_ds = RCDMQwenVisionCacheDataset(val_cache_dir)
        if train_ds.manifest.cond_dim != val_ds.manifest.cond_dim:
            raise ValueError("train/val qwen vision cache cond_dim mismatch")
        cond_dim = train_ds.manifest.cond_dim

        rcdm_config = rcdm_config_from_args(args)
        if rcdm_config.class_cond:
            raise ValueError("class_cond=True is not meaningful for Qwen visual feature conditioning")
        rcdm_model, diffusion = create_model_and_diffusion(
            rcdm_config,
            cond_dim=cond_dim,
            rcdm_root=str(args.rcdm_root) if args.rcdm_root is not None else None,
        )
        rcdm_model.to(device)
        optimizer = torch.optim.AdamW(rcdm_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ema_rates = parse_ema_rates(args.ema_rate)
        ema_states = {rate: init_ema_state(rcdm_model) for rate in ema_rates}

        train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_rcdm_qwen_vision_cache_batch,
        )
        val_loader = None
        if is_main():
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_rcdm_qwen_vision_cache_batch,
            )

        metadata = _build_metadata(args, rcdm_config, cond_dim, world)
        if is_main():
            (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            log_path = args.output_dir / "train_step_log.csv"
            if not (args.resume or args.resume_checkpoint is not None) or not log_path.exists():
                with log_path.open("w", newline="") as f:
                    csv.writer(f).writerow(["time", "epoch", "step", "loss", "val_loss"])
        wandb_run = _maybe_init_wandb(args, metadata)

        start_epoch, resume_step_in_epoch, step = _maybe_resume(
            args=args,
            model=rcdm_model,
            optimizer=optimizer,
            ema_states=ema_states,
            device=device,
        )
        if resume_step_in_epoch >= len(train_loader):
            if is_main():
                print(json.dumps({
                    "resume_epoch_boundary": True,
                    "checkpoint_epoch": start_epoch,
                    "checkpoint_step_in_epoch": resume_step_in_epoch,
                    "train_batches_per_epoch": len(train_loader),
                    "next_epoch": start_epoch + 1,
                }))
            start_epoch += 1
            resume_step_in_epoch = 0
        if world > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()

        train_model: torch.nn.Module = rcdm_model
        if world > 1:
            train_model = DDP(rcdm_model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

        last_epoch = start_epoch
        last_step_in_epoch = resume_step_in_epoch
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_model.train()
            for step_in_epoch, batch in enumerate(train_loader, start=1):
                if epoch == start_epoch and step_in_epoch <= resume_step_in_epoch:
                    continue
                last_epoch = epoch
                last_step_in_epoch = step_in_epoch
                cond, images = _batch_conditions_and_images(batch, device=device, image_size=args.image_size)
                t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device)
                loss = diffusion.training_losses(train_model, images, t, model_kwargs={"feat": cond})["loss"].mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unwrap_model(train_model).parameters(), args.grad_clip)
                optimizer.step()
                for rate, ema in ema_states.items():
                    update_ema_state(ema, train_model, rate)
                step += 1

                if step % args.log_interval == 0:
                    loss_value = loss.detach().float()
                    if world > 1 and dist.is_available() and dist.is_initialized():
                        dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
                        loss_value /= world
                    if is_main():
                        with (args.output_dir / "train_step_log.csv").open("a", newline="") as f:
                            csv.writer(f).writerow([time.time(), epoch, step, float(loss_value.item()), ""])
                        loss_float = float(loss_value.item())
                        if wandb_run is not None:
                            wandb_run.log({"rcdm_qwen_vision/train_loss": loss_float, "rcdm_qwen_vision/lr": optimizer.param_groups[0]["lr"], "epoch": epoch}, step=step)
                        print(json.dumps({"epoch": epoch, "step": step, "loss": loss_float}))

                if args.save_interval > 0 and step % args.save_interval == 0:
                    if world > 1 and dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    if is_main():
                        save_training_checkpoint(
                            output_dir=args.output_dir,
                            model=train_model,
                            optimizer=optimizer,
                            step=step,
                            epoch=epoch,
                            step_in_epoch=step_in_epoch,
                            metadata=metadata,
                            ema_states=ema_states,
                        )
                    if world > 1 and dist.is_available() and dist.is_initialized():
                        dist.barrier()

                if args.max_steps > 0 and step >= args.max_steps:
                    break

            if world > 1 and dist.is_available() and dist.is_initialized():
                dist.barrier()
            if is_main() and val_loader is not None and args.max_val_batches != 0:
                val_loss = _evaluate_loss(model=unwrap_model(train_model), diffusion=diffusion, loader=val_loader, device=device, image_size=args.image_size, max_batches=args.max_val_batches)
                with (args.output_dir / "train_step_log.csv").open("a", newline="") as f:
                    csv.writer(f).writerow([time.time(), epoch, step, "", val_loss])
                if wandb_run is not None:
                    wandb_run.log({"rcdm_qwen_vision/val_loss": val_loss, "epoch": epoch}, step=step)
                print(json.dumps({"epoch": epoch, "step": step, "val_loss": val_loss}))
            if world > 1 and dist.is_available() and dist.is_initialized():
                dist.barrier()
            if args.max_steps > 0 and step >= args.max_steps:
                break

        if is_main():
            save_training_checkpoint(
                output_dir=args.output_dir,
                model=train_model,
                optimizer=optimizer,
                step=step,
                epoch=last_epoch,
                step_in_epoch=last_step_in_epoch,
                metadata=metadata,
                ema_states=ema_states,
            )
        if wandb_run is not None:
            wandb_run.finish()
        if world > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()
        return 0
    finally:
        cleanup_dist()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train RCDM conditioned on frozen Qwen visual encoder features")
    ap.add_argument("--model", type=Path, required=True, help="Qwen/Qwen-SFT2 HF export directory")
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--rcdm-root", type=Path, default=None)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-checkpoint", type=Path, default=None)
    ap.add_argument("--wandb-project", default="nimloth")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-id", default=None)
    ap.add_argument("--wandb-resume", choices=("allow", "must", "never", "auto"), default="allow")
    ap.add_argument("--no-wandb", action="store_true")

    ap.add_argument("--qwen-vision-cache-dir", type=Path, required=True)
    ap.add_argument("--build-qwen-vision-cache", action="store_true")
    ap.add_argument("--force-rebuild-qwen-vision-cache", action="store_true")
    ap.add_argument("--qwen-vision-cache-build-batch-size", type=int, default=4)
    ap.add_argument("--qwen-vision-cache-shard-size", type=int, default=4096)
    ap.add_argument("--qwen-vision-cache-compression", choices=("gzip", "none"), default="gzip")
    ap.add_argument("--qwen-vision-cache-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--image-role", choices=("current", "next"), default="current")

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema-rate", default="0.9999")
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--max-train-records", type=int, default=-1)
    ap.add_argument("--max-val-records", type=int, default=-1)
    ap.add_argument("--max-val-batches", type=int, default=8)
    ap.add_argument("--success-only", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--min-pixels", type=int, default=3136)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--num-channels", type=int, default=256)
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-heads-upsample", type=int, default=-1)
    ap.add_argument("--num-head-channels", type=int, default=-1)
    ap.add_argument("--attention-resolutions", default="32,16,8")
    ap.add_argument("--channel-mult", default="")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--class-cond", action="store_true")
    ap.add_argument("--use-checkpoint", action="store_true")
    ap.add_argument("--use-scale-shift-norm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resblock-updown", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-fp16", action="store_true")
    ap.add_argument("--use-new-attention-order", action="store_true")
    ap.add_argument("--learn-sigma", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--diffusion-steps", type=int, default=1000)
    ap.add_argument("--noise-schedule", default="linear")
    ap.add_argument("--timestep-respacing", default="")
    ap.add_argument("--use-kl", action="store_true")
    ap.add_argument("--predict-xstart", action="store_true")
    ap.add_argument("--rescale-timesteps", action="store_true")
    ap.add_argument("--rescale-learned-sigmas", action="store_true")
    ap.add_argument("--g-shared", action="store_true")
    ap.add_argument("--pretrained", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    return train_rcdm_qwen_vision(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
