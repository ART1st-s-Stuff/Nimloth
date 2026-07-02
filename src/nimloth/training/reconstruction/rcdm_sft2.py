"""Train RCDM on Nimloth SFT2 latent states.

This is a post-hoc visualization model: Qwen, ``StateProjector`` and the WM
predictor are frozen.  Only the RCDM UNet is trained to denoise images while
conditioned on SFT2 WM state embeddings.
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

from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.rcdm.checkpoint import (
    init_ema_state,
    parse_ema_rates,
    save_training_checkpoint,
    unwrap_model,
    update_ema_state,
)
from nimloth.rcdm.config import RCDMConfig, create_model_and_diffusion, rcdm_config_from_args
from nimloth.rcdm.image_utils import image_to_diffusion_tensor
from nimloth.rcdm.state_cache import (
    RCDMStateCacheDataset,
    build_rcdm_state_cache,
    collate_rcdm_state_cache_batch,
    state_cache_ready,
)
from nimloth.training.common.dist import cleanup_dist, is_main, setup_dist
from nimloth.training.common.qwen_batch import build_qwen_batch
from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def _encode_states(
    *,
    model,
    processor,
    token_id_map: dict[str, int],
    items: list[dict[str, Any]],
    state_proj: StateProjector,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    enc = build_qwen_batch(items, processor, max_length=max_length)
    hidden, _ = extract_qwen_latents(model, enc, token_id_map, device)
    return state_proj(hidden).float()


def _load_frozen_sft2_modules(args: argparse.Namespace, device: torch.device):
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(device)
    _freeze(model)

    wm_predictor = LatentWMPredictor.load_checkpoint(args.wm_checkpoint, map_location=device).to(device)
    _freeze(wm_predictor)
    state_proj = StateProjector(model.config.hidden_size, wm_predictor.emb_dim).to(device)
    state_proj.load_state_dict(torch.load(args.state_proj_checkpoint, map_location=device, weights_only=True))
    _freeze(state_proj)
    return processor, token_id_map, model, state_proj, wm_predictor


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _latest_training_state(output_dir: Path) -> Path | None:
    candidates = [p for p in output_dir.glob("training_state_*.pt") if _checkpoint_step(p) >= 0]
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_step)


def _training_state_for_resume(args: argparse.Namespace) -> Path | None:
    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    if resume_checkpoint is not None:
        path = Path(resume_checkpoint)
        if path.name.startswith("model_"):
            return path.with_name(path.name.replace("model_", "training_state_", 1))
        if path.name.startswith("training_state_"):
            return path
        raise ValueError("--resume-checkpoint must point to model_*.pt or training_state_*.pt")
    if not getattr(args, "resume", False):
        return None
    return _latest_training_state(args.output_dir)


def _maybe_resume(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_states: dict[float, dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[int, int, int]:
    """Restore model/optimizer/EMA and return (start_epoch, skip_steps, global_step)."""

    state_path = _training_state_for_resume(args)
    if state_path is None:
        return 1, 0, 0
    if not state_path.is_file():
        raise FileNotFoundError(f"resume training state not found: {state_path}")
    step = _checkpoint_step(state_path)
    if step < 0:
        raise ValueError(f"cannot parse checkpoint step from {state_path}")
    model_path = state_path.with_name(f"model_{step:09d}.pt")
    if not model_path.is_file():
        raise FileNotFoundError(f"resume model checkpoint not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
    state = torch.load(state_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    for rate in list(ema_states.keys()):
        ema_path = state_path.with_name(f"ema_{rate}_{step:09d}.pt")
        if ema_path.is_file():
            ema_states[rate] = torch.load(ema_path, map_location=device, weights_only=True)
    epoch = int(state.get("epoch", 1))
    step_in_epoch = int(state.get("step_in_epoch", 0))
    global_step = int(state.get("step", step))
    if is_main():
        print(json.dumps({
            "resume_checkpoint": str(state_path),
            "model_checkpoint": str(model_path),
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
        }))
    return epoch, step_in_epoch, global_step


def _maybe_init_wandb(args: argparse.Namespace, metadata: dict[str, Any]):
    if getattr(args, "no_wandb", False) or not is_main():
        return None
    try:
        import wandb
    except Exception as exc:
        print(json.dumps({"wandb_init_skipped": str(exc)}))
        return None

    resume_requested = bool(getattr(args, "resume", False) or getattr(args, "resume_checkpoint", None) is not None)
    wandb_id_path = args.output_dir / "wandb_run_id.txt"
    wandb_id = getattr(args, "wandb_id", None)
    if wandb_id is None and resume_requested and wandb_id_path.is_file():
        wandb_id = wandb_id_path.read_text(encoding="utf-8").strip() or None
    resume_mode = getattr(args, "wandb_resume", "allow") if resume_requested else None
    try:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=wandb_id,
            resume=resume_mode,
            config=metadata,
            dir=str(args.output_dir),
        )
    except Exception as exc:
        print(json.dumps({"wandb_init_skipped": str(exc)}))
        return None
    if getattr(run, "id", None):
        wandb_id_path.write_text(str(run.id), encoding="utf-8")
    return run


def _build_metadata(args: argparse.Namespace, rcdm_config: RCDMConfig, cond_dim: int, world: int) -> dict[str, Any]:
    return {
        "task": "nimloth_sft2_rcdm_reconstruction",
        "model": str(args.model),
        "state_proj_checkpoint": str(args.state_proj_checkpoint),
        "wm_checkpoint": str(args.wm_checkpoint),
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "cond_dim": int(cond_dim),
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
            "target": "current_image_from_current_sft2_state",
            "resume": bool(args.resume or args.resume_checkpoint is not None),
            "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
            "state_cache_dir": str(args.state_cache_dir) if args.state_cache_dir is not None else None,
            "build_state_cache": args.build_state_cache,
            "state_cache_shard_size": args.state_cache_shard_size,
            "state_cache_compression": args.state_cache_compression,
            "state_cache_dtype": args.state_cache_dtype,
        },
        "wandb": {
            "project": args.wandb_project,
            "run_name": args.wandb_run_name,
            "id": args.wandb_id,
            "resume": args.wandb_resume,
            "enabled": not args.no_wandb,
        },
    }


@torch.no_grad()
def _batch_states_and_images(
    *,
    batch,
    qwen_model,
    processor,
    token_id_map: dict[str, int] | None,
    state_proj: StateProjector | None,
    device: torch.device,
    image_size: int,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict) and "state_emb" in batch:
        states = batch["state_emb"].to(device=device, dtype=torch.float32)
        paths = batch["current_image_path"]
    else:
        if token_id_map is None or state_proj is None:
            raise ValueError("online Qwen state extraction requires token_id_map and state_proj")
        states = _encode_states(
            model=qwen_model,
            processor=processor,
            token_id_map=token_id_map,
            items=batch,
            state_proj=state_proj,
            device=device,
            max_length=max_length,
        )
        paths = [item["current_image_path"] for item in batch]
    images = torch.stack([
        image_to_diffusion_tensor(path, image_size=image_size, device=device)
        for path in paths
    ])
    return states, images


def _evaluate_loss(
    *,
    model,
    diffusion,
    loader,
    qwen_model,
    processor,
    token_id_map: dict[str, int] | None,
    state_proj: StateProjector | None,
    device: torch.device,
    image_size: int,
    max_length: int,
    max_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_idx, items in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        states, images = _batch_states_and_images(
            batch=items,
            qwen_model=qwen_model,
            processor=processor,
            token_id_map=token_id_map,
            state_proj=state_proj,
            device=device,
            image_size=image_size,
            max_length=max_length,
        )
        t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device)
        loss = diffusion.training_losses(model, images, t, model_kwargs={"feat": states})["loss"].mean()
        losses.append(float(loss.detach().cpu().item()))
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def train_rcdm_sft2(args: argparse.Namespace) -> int:
    rank, world, _local_rank, device = setup_dist()
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(int(args.seed) + rank)

        qwen_model = None
        processor = None
        token_id_map: dict[str, int] | None = None
        state_proj: StateProjector | None = None
        cond_dim: int

        if args.state_cache_dir is not None:
            train_cache_dir = args.state_cache_dir / "train"
            val_cache_dir = args.state_cache_dir / "val"
            need_cache_build = (
                args.build_state_cache
                or args.force_rebuild_state_cache
                or not state_cache_ready(train_cache_dir)
                or not state_cache_ready(val_cache_dir)
            )
            if is_main() and need_cache_build:
                processor, token_id_map, qwen_model, state_proj, wm_predictor = _load_frozen_sft2_modules(args, device)
                build_kwargs = dict(
                    model_path=args.model,
                    state_proj_checkpoint=args.state_proj_checkpoint,
                    wm_checkpoint=args.wm_checkpoint,
                    processor=processor,
                    qwen_model=qwen_model,
                    token_id_map=token_id_map,
                    state_proj=state_proj,
                    device=device,
                    max_length=args.max_length,
                    max_pixels=args.max_pixels,
                    min_pixels=3136,
                    batch_size=args.state_cache_build_batch_size,
                    shard_size=args.state_cache_shard_size,
                    compression=args.state_cache_compression,
                    state_dtype=args.state_cache_dtype,
                    force=args.force_rebuild_state_cache,
                )
                build_rcdm_state_cache(
                    jsonl_path=args.train_jsonl,
                    cache_dir=train_cache_dir,
                    split_name="train",
                    max_records=args.max_train_records,
                    success_only=args.success_only,
                    **build_kwargs,
                )
                build_rcdm_state_cache(
                    jsonl_path=args.val_jsonl,
                    cache_dir=val_cache_dir,
                    split_name="val",
                    max_records=args.max_val_records,
                    success_only=False,
                    **build_kwargs,
                )
                del qwen_model, state_proj, wm_predictor
                qwen_model = None
                state_proj = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if world > 1 and dist.is_available() and dist.is_initialized():
                dist.barrier()
            train_ds = RCDMStateCacheDataset(train_cache_dir)
            val_ds = RCDMStateCacheDataset(val_cache_dir)
            cond_dim = train_ds.manifest.cond_dim
            if val_ds.manifest.cond_dim != cond_dim:
                raise ValueError("train/val state cache cond_dim mismatch")
            train_collate = collate_rcdm_state_cache_batch
            val_collate = collate_rcdm_state_cache_batch
        else:
            processor, token_id_map, qwen_model, state_proj, wm_predictor = _load_frozen_sft2_modules(args, device)
            cond_dim = wm_predictor.emb_dim
            train_ds = TransitionQwenDataset(args.train_jsonl, max_records=args.max_train_records, success_only=args.success_only)
            val_ds = TransitionQwenDataset(args.val_jsonl, max_records=args.max_val_records)
            train_collate = collate_transition_batch
            val_collate = collate_transition_batch

        rcdm_config = rcdm_config_from_args(args)
        if rcdm_config.use_fp16:
            raise ValueError("RCDM use_fp16 is not supported by this Nimloth training loop yet")
        if rcdm_config.class_cond:
            raise ValueError("class_cond=True is not meaningful for SFT2 latent conditioning")

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
            collate_fn=train_collate,
        )
        val_loader = None
        if is_main():
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=val_collate,
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
            for step_in_epoch, items in enumerate(train_loader, start=1):
                if epoch == start_epoch and step_in_epoch <= resume_step_in_epoch:
                    continue
                last_epoch = epoch
                last_step_in_epoch = step_in_epoch
                states, images = _batch_states_and_images(
                    batch=items,
                    qwen_model=qwen_model,
                    processor=processor,
                    token_id_map=token_id_map,
                    state_proj=state_proj,
                    device=device,
                    image_size=args.image_size,
                    max_length=args.max_length,
                )
                t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device)
                losses = diffusion.training_losses(train_model, images, t, model_kwargs={"feat": states})
                loss = losses["loss"].mean()

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
                            wandb_run.log(
                                {
                                    "rcdm/train_loss": loss_float,
                                    "rcdm/lr": optimizer.param_groups[0]["lr"],
                                    "epoch": epoch,
                                },
                                step=step,
                            )
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
                val_loss = _evaluate_loss(
                    model=unwrap_model(train_model),
                    diffusion=diffusion,
                    loader=val_loader,
                    qwen_model=qwen_model,
                    processor=processor,
                    token_id_map=token_id_map,
                    state_proj=state_proj,
                    device=device,
                    image_size=args.image_size,
                    max_length=args.max_length,
                    max_batches=args.max_val_batches,
                )
                with (args.output_dir / "train_step_log.csv").open("a", newline="") as f:
                    csv.writer(f).writerow([time.time(), epoch, step, "", val_loss])
                if wandb_run is not None:
                    wandb_run.log({"rcdm/val_loss": val_loss, "epoch": epoch}, step=step)
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
    ap = argparse.ArgumentParser(description="Train RCDM conditioned on Nimloth SFT2 latent states")
    ap.add_argument("--model", type=Path, required=True, help="SFT2 HF export directory")
    ap.add_argument("--state-proj-checkpoint", type=Path, required=True)
    ap.add_argument("--wm-checkpoint", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--rcdm-root", type=Path, default=None)

    ap.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in --output-dir")
    ap.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Explicit model_*.pt or training_state_*.pt checkpoint to resume from",
    )
    ap.add_argument("--wandb-project", default="nimloth")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-id", default=None)
    ap.add_argument("--wandb-resume", choices=("allow", "must", "never", "auto"), default="allow")
    ap.add_argument("--no-wandb", action="store_true")

    ap.add_argument(
        "--state-cache-dir",
        type=Path,
        default=None,
        help="Compressed cache root for precomputed StateProjector(Qwen latent) embeddings",
    )
    ap.add_argument("--build-state-cache", action="store_true", help="Build missing RCDM state cache before training")
    ap.add_argument("--force-rebuild-state-cache", action="store_true")
    ap.add_argument("--state-cache-build-batch-size", type=int, default=1)
    ap.add_argument("--state-cache-shard-size", type=int, default=4096)
    ap.add_argument("--state-cache-compression", choices=("gzip", "none"), default="gzip")
    ap.add_argument("--state-cache-dtype", choices=("float16", "bfloat16", "float32"), default="float16")

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
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--seed", type=int, default=42)

    # RCDM model/diffusion flags. Defaults mirror upstream 128x128 RCDM.
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
    return train_rcdm_sft2(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
