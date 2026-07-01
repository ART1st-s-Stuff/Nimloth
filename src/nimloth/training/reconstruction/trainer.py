"""Train a post-hoc WM image decoder with frozen Qwen and state projector."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.eval.reconstruction import evaluate_reconstruction, image_to_tensor
from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.common.dist import cleanup_dist, is_main, setup_dist
from nimloth.training.common.qwen_batch import build_qwen_batch
from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.reconstruction import WMImageDecoder, WMImageDecoderConfig
from nimloth.wm.state_proj import StateProjector


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


@torch.no_grad()
def _encode_states(model, processor, token_id_map, items, state_proj, device, max_length: int) -> torch.Tensor:
    enc = build_qwen_batch(items, processor, max_length=max_length)
    hidden, _ = extract_qwen_latents(model, enc, token_id_map, device)
    return state_proj(hidden).float()


def _step_checkpoint_number(path: Path) -> int:
    match = re.fullmatch(r"step_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _latest_step_checkpoint(output_dir: Path) -> Path | None:
    candidates = [p for p in output_dir.glob("step_*") if p.is_dir() and _step_checkpoint_number(p) >= 0]
    if not candidates:
        return None
    return max(candidates, key=_step_checkpoint_number)


def _save_step_checkpoint(
    *,
    output_dir: Path,
    decoder: WMImageDecoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    keep_last: int,
) -> None:
    ckpt = output_dir / f"step_{global_step:09d}"
    decoder.save_checkpoint(ckpt)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "step_in_epoch": int(step_in_epoch),
            "global_step": int(global_step),
        },
        ckpt / "training_state.pt",
    )
    step_ckpts = sorted(
        [p for p in output_dir.glob("step_*") if p.is_dir() and _step_checkpoint_number(p) >= 0],
        key=_step_checkpoint_number,
    )
    for old in step_ckpts[: max(0, len(step_ckpts) - keep_last)]:
        import shutil

        shutil.rmtree(old)


def _maybe_resume(
    *,
    args: argparse.Namespace,
    decoder: WMImageDecoder,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, int]:
    if not getattr(args, "resume", False):
        return 1, 0, 0
    ckpt = _latest_step_checkpoint(args.output_dir)
    if ckpt is None:
        return 1, 0, 0
    loaded = WMImageDecoder.load_checkpoint(ckpt, map_location=device)
    decoder.load_state_dict(loaded.state_dict())
    state_path = ckpt / "training_state.pt"
    state = torch.load(state_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    epoch = int(state.get("epoch", 1))
    step_in_epoch = int(state.get("step_in_epoch", 0))
    global_step = int(state.get("global_step", _step_checkpoint_number(ckpt)))
    print(json.dumps({"resume_checkpoint": str(ckpt), "epoch": epoch, "step_in_epoch": step_in_epoch, "global_step": global_step}))
    return epoch, step_in_epoch, global_step


def _tensor_to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().clamp(0, 1).mul(255).byte().cpu()
    return image.permute(1, 2, 0).numpy()


def _fixed_val_preview_items(val_ds, max_items: int) -> list[dict]:
    """Pick fixed preview transitions from different rollout records.

    Transition datasets are expanded per step, so adjacent indices usually belong
    to the same scene/rollout.  For visual monitoring we first keep one eligible
    transition per record, then take evenly spaced records across the val set.
    """

    by_record: dict[str, dict] = {}
    for idx in range(len(val_ds)):
        item = collate_transition_batch([val_ds[idx]])[0]
        if not item.get("next_messages"):
            continue
        record_id = str(item.get("id", idx)).split(":", 1)[0]
        by_record.setdefault(record_id, item)

    candidates = list(by_record.values())
    if len(candidates) <= max_items:
        return candidates
    if max_items <= 1:
        return candidates[:max_items]
    indices = torch.linspace(0, len(candidates) - 1, steps=max_items).round().long().tolist()
    return [candidates[i] for i in indices]


@torch.no_grad()
def _log_wandb_val_images(
    *,
    wandb_run,
    model,
    processor,
    token_id_map: dict[str, int],
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    decoder: WMImageDecoder,
    items: list[dict],
    device: torch.device,
    args: argparse.Namespace,
    step: int,
) -> None:
    if wandb_run is None or not items:
        return
    try:
        import wandb
    except Exception:
        return
    model.eval()
    state_proj.eval()
    wm_predictor.eval()
    decoder.eval()

    cur_enc = build_qwen_batch(items, processor, max_length=args.max_length)
    cur_hidden, _ = extract_qwen_latents(model, cur_enc, token_id_map, device)
    s_cur = state_proj(cur_hidden).float()
    cur_recon = decoder(s_cur)

    actions = torch.tensor([item["action_index"] for item in items], device=device, dtype=torch.long)
    pred_next = decoder(wm_predictor(s_cur, actions).float())

    table = wandb.Table(
        columns=["sample_id", "current_gt", "current_recon", "next_gt", "pred_next_recon"]
    )
    log_payload = {}
    for i, item in enumerate(items):
        cur_gt = image_to_tensor(item["current_image_path"], image_size=args.image_size, device=device)
        next_gt = image_to_tensor(item["next_image_path"], image_size=args.image_size, device=device)
        sample_id = str(item.get("id", i))
        cur_gt_np = _tensor_to_hwc_uint8(cur_gt)
        cur_recon_np = _tensor_to_hwc_uint8(cur_recon[i])
        next_gt_np = _tensor_to_hwc_uint8(next_gt)
        pred_next_np = _tensor_to_hwc_uint8(pred_next[i])
        table.add_data(
            sample_id,
            wandb.Image(cur_gt_np, caption="current_gt"),
            wandb.Image(cur_recon_np, caption="current_recon"),
            wandb.Image(next_gt_np, caption="next_gt"),
            wandb.Image(pred_next_np, caption="pred_next_recon"),
        )
        composite = np.concatenate([cur_gt_np, cur_recon_np, next_gt_np, pred_next_np], axis=1)
        log_payload[f"reconstruction/preview_{i:02d}"] = wandb.Image(
            composite,
            caption=f"{sample_id}: current_gt | current_recon | next_gt | pred_next_recon",
        )
    log_payload["reconstruction/val_preview_table"] = table
    wandb_run.log(log_payload, step=step)
    print(json.dumps({"wandb_val_preview_logged": True, "step": step, "num_items": len(items)}))
    decoder.train()


def _make_train_loader(args: argparse.Namespace, train_ds, epoch: int):
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + int(epoch))
    return torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_transition_batch,
    )


def _maybe_init_wandb(args: argparse.Namespace, meta: dict) -> object | None:
    if getattr(args, "no_wandb", False) or not is_main():
        return None
    try:
        import wandb
    except Exception:
        return None
    return wandb.init(
        project="nimloth",
        name=getattr(args, "wandb_run_name", None),
        config=meta,
        dir=str(args.output_dir),
    )


def _unwrap_decoder(decoder: WMImageDecoder | DDP) -> WMImageDecoder:
    return decoder.module if isinstance(decoder, DDP) else decoder


def _reduce_mean_scalar(value: torch.Tensor, world: int) -> float:
    scalar = value.detach().float()
    if world > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
        scalar /= world
    return float(scalar.item())


def _set_decoder_lr(optimizer: torch.optim.Optimizer, *, base_lr: float, global_step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        lr = base_lr
    else:
        lr = base_lr * min(1.0, max(global_step, 1) / float(warmup_steps))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_reconstruction_decoder(args: argparse.Namespace) -> int:
    rank, world, local_rank, device = setup_dist()
    try:
        if torch.cuda.is_available():
            torch.manual_seed(int(args.seed) + rank)
        args.output_dir.mkdir(parents=True, exist_ok=True)

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

        decoder = WMImageDecoder(
            WMImageDecoderConfig(
                emb_dim=wm_predictor.emb_dim,
                image_size=args.image_size,
                patch_size=args.patch_size,
                hidden_dim=args.hidden_dim,
                depth=args.depth,
                heads=args.heads,
            )
        ).to(device)
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_ds = TransitionQwenDataset(args.train_jsonl, max_records=args.max_train_records, success_only=args.success_only)
        val_ds = TransitionQwenDataset(args.val_jsonl, max_records=args.max_val_records)
        train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_transition_batch,
        )
        val_loader = None
        val_preview_items: list[dict] = []
        if is_main():
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_transition_batch,
            )
            val_preview_items = _fixed_val_preview_items(val_ds, args.wandb_image_samples)

        meta = {
            "model": str(args.model),
            "state_proj_checkpoint": str(args.state_proj_checkpoint),
            "wm_checkpoint": str(args.wm_checkpoint),
            "train_jsonl": str(args.train_jsonl),
            "val_jsonl": str(args.val_jsonl),
            "decoder_config": decoder.config.__dict__,
            "epochs": args.epochs,
            "save_interval": args.save_interval,
            "keep_last_checkpoints": args.keep_last_checkpoints,
            "world_size": world,
            "per_rank_batch_size": args.batch_size,
            "effective_batch_size": args.batch_size * world,
            "lr": args.lr,
            "lr_warmup_steps": args.lr_warmup_steps,
        }
        if is_main():
            (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        wandb_run = _maybe_init_wandb(args, meta)

        log_path = args.output_dir / "train_step_log.csv"
        if is_main() and (not log_path.exists() or not args.resume):
            with log_path.open("w", newline="") as f:
                csv.writer(f).writerow(["time", "epoch", "step", "loss", "lr", "val_pred_mse", "val_oracle_mse"])

        start_epoch, resume_step_in_epoch, step = _maybe_resume(
            args=args,
            decoder=decoder,
            optimizer=optimizer,
            device=device,
        )
        _set_decoder_lr(optimizer, base_lr=args.lr, global_step=step, warmup_steps=args.lr_warmup_steps)
        if world > 1:
            decoder = DDP(decoder, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

        if wandb_run is not None and val_preview_items:
            _log_wandb_val_images(
                wandb_run=wandb_run,
                model=model,
                processor=processor,
                token_id_map=token_id_map,
                state_proj=state_proj,
                wm_predictor=wm_predictor,
                decoder=_unwrap_decoder(decoder),
                items=val_preview_items,
                device=device,
                args=args,
                step=step,
            )
        best_val = float("inf")
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            decoder.train()
            step_in_epoch = 0
            for items in train_loader:
                step_in_epoch += 1
                if epoch == start_epoch and step_in_epoch <= resume_step_in_epoch:
                    continue
                states = _encode_states(model, processor, token_id_map, items, state_proj, device, args.max_length)
                targets = torch.stack([
                    image_to_tensor(item["current_image_path"], image_size=args.image_size, device=device)
                    for item in items
                ])
                pred = decoder(states)
                loss = F.l1_loss(pred, targets) if args.loss == "l1" else F.mse_loss(pred, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_unwrap_decoder(decoder).parameters(), 1.0)
                optimizer.step()
                step += 1
                current_lr = _set_decoder_lr(
                    optimizer,
                    base_lr=args.lr,
                    global_step=step,
                    warmup_steps=args.lr_warmup_steps,
                )

                if args.save_interval > 0 and step % args.save_interval == 0:
                    if world > 1 and dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    if is_main():
                        _save_step_checkpoint(
                            output_dir=args.output_dir,
                            decoder=_unwrap_decoder(decoder),
                            optimizer=optimizer,
                            epoch=epoch,
                            step_in_epoch=step_in_epoch,
                            global_step=step,
                            keep_last=args.keep_last_checkpoints,
                        )
                    if world > 1 and dist.is_available() and dist.is_initialized():
                        dist.barrier()

                if (
                    wandb_run is not None
                    and args.wandb_image_interval > 0
                    and step % args.wandb_image_interval == 0
                ):
                    _log_wandb_val_images(
                        wandb_run=wandb_run,
                        model=model,
                        processor=processor,
                        token_id_map=token_id_map,
                        state_proj=state_proj,
                        wm_predictor=wm_predictor,
                        decoder=_unwrap_decoder(decoder),
                        items=val_preview_items,
                        device=device,
                        args=args,
                        step=step,
                    )

                if step % args.log_interval == 0:
                    loss_value = _reduce_mean_scalar(loss, world)
                    if is_main():
                        with log_path.open("a", newline="") as f:
                            csv.writer(f).writerow([time.time(), epoch, step, loss_value, current_lr, "", ""])
                        if wandb_run is not None:
                            wandb_run.log({"reconstruction/train_loss": loss_value, "reconstruction/lr": current_lr, "epoch": epoch}, step=step)
                        print(json.dumps({"epoch": epoch, "step": step, "loss": loss_value, "lr": current_lr, "world_size": world}))

            if world > 1 and dist.is_available() and dist.is_initialized():
                dist.barrier()
            val_metrics = None
            if is_main() and val_loader is not None:
                val_metrics = evaluate_reconstruction(
                    model=model,
                    processor=processor,
                    token_id_map=token_id_map,
                    state_proj=state_proj,
                    wm_predictor=wm_predictor,
                    decoder=_unwrap_decoder(decoder),
                    loader=val_loader,
                    device=device,
                    output_dir=args.output_dir / f"val_epoch_{epoch:03d}",
                    max_batches=args.max_val_batches,
                    max_length=args.max_length,
                    save_samples=args.save_samples,
                )
                val_pred = float(val_metrics.get("pred_mse", float("inf")))
                with log_path.open("a", newline="") as f:
                    csv.writer(f).writerow([
                        time.time(), epoch, step, "", optimizer.param_groups[0]["lr"], val_pred, val_metrics.get("oracle_mse", "")
                    ])
                if wandb_run is not None:
                    wandb_run.log(
                        {f"reconstruction/val_{key}": value for key, value in val_metrics.items()},
                        step=step,
                    )
                _unwrap_decoder(decoder).save_checkpoint(args.output_dir / f"epoch_{epoch:03d}")
                if val_pred < best_val:
                    best_val = val_pred
                    _unwrap_decoder(decoder).save_checkpoint(args.output_dir / "best")
                print(json.dumps({"epoch": epoch, "val_metrics": val_metrics, "best_val_pred_mse": best_val}))
            if world > 1 and dist.is_available() and dist.is_initialized():
                dist.barrier()

        if is_main():
            _unwrap_decoder(decoder).save_checkpoint(args.output_dir / "final")
            if wandb_run is not None:
                wandb_run.finish()
        if world > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()
        return 0
    finally:
        cleanup_dist()
