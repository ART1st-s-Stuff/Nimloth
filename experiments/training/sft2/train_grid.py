#!/usr/bin/env python3
"""SFT2: LeWM-style encoded grid dynamics with decoded DINO supervision."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.dino import (
    DEFAULT_DINO_MODEL,
    CachedDINOGridEncoder,
    FrozenDINOEncoder,
    resolve_dino_identity,
)
from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.common.dist import cleanup_dist, is_main, setup_dist
from nimloth.training.common.wandb_logging import log_train_step, log_val_epoch, maybe_init_wandb
from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch
from nimloth.training.sft2.grid_step import compute_grid_sft2_loss
from nimloth.training.sft2.preprocess_cache import (
    COMPACT_CACHE_FORMAT,
    DEFAULT_MIN_PIXELS,
    CachedTransitionDataset,
    CompactCachedTransitionCollator,
    cache_fingerprint,
    unpack_transition_batch,
)
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.training.sft2.step import _forward_next_latents, wm_eligible_indices
from nimloth.wm.grid import (
    EMATargetGridEncoder,
    LeWMGridDecoder,
    LeWMGridEncoder,
    LeWMSpatialPredictor,
    load_sft1_slot_projector,
)
from nimloth.wm.lewm import SIGReg
from nimloth.wm.value_head import ValueHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft1-checkpoint", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--dino-cache-root", type=Path)
    parser.add_argument("--require-dino-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--wm-lr", type=float, default=3e-4)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--value-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--dino-weight", type=float, default=0.5)
    parser.add_argument("--sigreg-weight", type=float, default=0.1)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--encoder-hidden-dim", type=int, default=2048)
    parser.add_argument("--decoder-hidden-dim", type=int, default=2048)
    parser.add_argument("--wm-depth", type=int, default=6)
    parser.add_argument("--wm-heads", type=int, default=16)
    parser.add_argument("--wm-dim-head", type=int, default=64)
    parser.add_argument("--wm-mlp-dim", type=int, default=2048)
    parser.add_argument("--wm-dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--max-train-records", type=int, default=-1)
    parser.add_argument("--max-val-records", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preprocess-cache-dir", type=Path)
    parser.add_argument("--require-prebuilt-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.dino_weight != 0.5:
        parser.error("the human-approved SFT2 decoded-DINO weight is fixed at 0.5")
    if args.sigreg_weight != 0.1:
        parser.error("the human-approved SFT2 SIGReg weight is fixed at 0.1")
    if args.ema_decay != 0.99:
        parser.error("the human-approved target encoder EMA decay is fixed at 0.99")
    if args.latent_weight <= 0 or args.value_weight < 0:
        parser.error("loss weights are invalid")
    return args


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def _atomic_torch_save(payload, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _prepare_datasets(args, processor):
    raw_train = TransitionQwenDataset(args.train_jsonl, max_records=args.max_train_records)
    raw_val = TransitionQwenDataset(args.val_jsonl, max_records=args.max_val_records)
    if args.preprocess_cache_dir is None:
        if args.require_prebuilt_cache:
            raise ValueError("--require-prebuilt-cache requires --preprocess-cache-dir")
        return raw_train, raw_val, collate_transition_batch, collate_transition_batch

    datasets = []
    collators = []
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("processor tokenizer must define pad_token_id")
    for split, raw in (("train", raw_train), ("val", raw_val)):
        cache_dir = args.preprocess_cache_dir / split
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"preprocess cache manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        jsonl_path = args.train_jsonl if split == "train" else args.val_jsonl
        expected_base_fingerprint = cache_fingerprint(
            jsonl_path,
            max_length=args.max_length,
            max_pixels=args.max_pixels,
            min_pixels=DEFAULT_MIN_PIXELS,
            vocab_size=len(processor.tokenizer),
            value_gamma=1.0,
            latent_token_count=16,
            mask_latent_query_labels=True,
            cache_format=COMPACT_CACHE_FORMAT,
            image_dtype="bfloat16",
            processor_source=str(args.sft1_checkpoint.resolve()),
        )
        expected = {
            "format": COMPACT_CACHE_FORMAT,
            "base_fingerprint": expected_base_fingerprint,
            "count": len(raw),
            "latent_token_count": 16,
            "mask_latent_query_labels": True,
            "max_length": args.max_length,
            "max_pixels": args.max_pixels,
            "image_dtype": "bfloat16",
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"preprocess cache mismatch at {cache_dir}: {mismatches}")
        datasets.append(CachedTransitionDataset(cache_dir, raw.samples))
        collators.append(CompactCachedTransitionCollator(cache_dir, pad_token_id=pad_id))
    return datasets[0], datasets[1], collators[0], collators[1]


def _encode_batch(model, batch, processor, token_ids, device, args):
    items, current_enc, next_rows = unpack_transition_batch(
        batch,
        processor,
        args.max_length,
        latent_token_count=16,
        mask_latent_query_labels=True,
    )
    current_enc.pop("labels", None)
    with torch.no_grad():
        current_hidden, _ = extract_qwen_latents(
            model,
            current_enc,
            token_ids,
            device,
            latent_token_count=16,
        )
    indices = wm_eligible_indices(items)
    if indices:
        next_hidden = _forward_next_latents(
            model,
            items,
            indices,
            processor,
            token_ids,
            device,
            args.max_length,
            vision_ema=None,
            next_enc_rows=next_rows,
            pad_token_id=processor.tokenizer.pad_token_id,
            latent_token_count=16,
        )
    else:
        next_hidden = current_hidden[:0]
    return items, current_hidden, next_hidden, indices


def _dino_targets(teacher, items, device):
    paths = []
    for index, item in enumerate(items):
        path = item.get("next_image_path")
        if not path:
            raise ValueError(f"transition item {index} is missing next_image_path")
        paths.append(path)
    return teacher.encode_image_paths_grid(paths, device=device, grid_size=4)


def _save_checkpoint(encoder, target, wm, decoder, value, optimizer, output, *, epoch, step, best, args):
    output.mkdir(parents=True, exist_ok=True)
    enc = _unwrap(encoder)
    pred = _unwrap(wm)
    dec = _unwrap(decoder)
    head = _unwrap(value)
    _atomic_torch_save(enc.state_dict(), output / "online_encoder.pt")
    _atomic_torch_save(target.state_dict(), output / "target_encoder_ema.pt")
    _atomic_torch_save(pred.state_dict(), output / "grid_wm.pt")
    _atomic_torch_save(dec.state_dict(), output / "dino_decoder.pt")
    head.save_checkpoint(output / "value_head")
    state = {
        "epoch": int(epoch),
        "step": int(step),
        "best_val_total": float(best),
        "optimizer": optimizer.state_dict(),
        "sft1_checkpoint": str(args.sft1_checkpoint.resolve()),
        "dino_model": str(args.dino_model),
        "dino_identity": args.dino_identity,
        "dino_cache_root": str(args.dino_cache_root.resolve()) if args.dino_cache_root else None,
        "dino_cache_fingerprint": args.dino_cache_fingerprint,
        "grid_tokens": 16,
        "state_dim": 1024,
        "latent_weight": float(args.latent_weight),
        "dino_weight": float(args.dino_weight),
        "sigreg_weight": float(args.sigreg_weight),
        "value_weight": float(args.value_weight),
        "ema_decay": float(args.ema_decay),
        "wm": {
            "depth": args.wm_depth,
            "heads": args.wm_heads,
            "dim_head": args.wm_dim_head,
            "mlp_dim": args.wm_mlp_dim,
            "dropout": args.wm_dropout,
            "attention": "bidirectional_spatial",
            "conditioning": "LeWM_AdaLN_zero",
        },
        "encoder_decoder": "LeWM_MLP_per_slot",
        "dino_target": "next_rgb_final_patch_adaptive_pool_4x4",
    }
    _atomic_torch_save(state, output / "training_state.pt")
    (output / "config.json").write_text(json.dumps({k: v for k, v in state.items() if k != "optimizer"}, indent=2), encoding="utf-8")


def _load_checkpoint(encoder, target, wm, decoder, value, optimizer, checkpoint, args):
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    invariants = {
        "sft1_checkpoint": str(args.sft1_checkpoint.resolve()),
        "dino_model": str(args.dino_model),
        "dino_identity": args.dino_identity,
        "dino_cache_root": str(args.dino_cache_root.resolve()) if args.dino_cache_root else None,
        "dino_cache_fingerprint": args.dino_cache_fingerprint,
        "grid_tokens": 16,
        "state_dim": 1024,
        "dino_weight": 0.5,
        "ema_decay": 0.99,
    }
    mismatches = {key: (state.get(key), expected) for key, expected in invariants.items() if state.get(key) != expected}
    if mismatches:
        raise ValueError(f"resume checkpoint invariant mismatch: {mismatches}")
    encoder.load_state_dict(torch.load(checkpoint / "online_encoder.pt", map_location="cpu", weights_only=True))
    target.load_state_dict(torch.load(checkpoint / "target_encoder_ema.pt", map_location="cpu", weights_only=True))
    wm.load_state_dict(torch.load(checkpoint / "grid_wm.pt", map_location="cpu", weights_only=True))
    decoder.load_state_dict(torch.load(checkpoint / "dino_decoder.pt", map_location="cpu", weights_only=True))
    loaded_value = ValueHead.load_checkpoint(checkpoint / "value_head", emb_dim=1024)
    value.load_state_dict(loaded_value.state_dict())
    optimizer.load_state_dict(state["optimizer"])
    return int(state["epoch"]) + 1, int(state["step"]), float(state["best_val_total"])


def _reduce(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _sync_ema_buffers(target: EMATargetGridEncoder) -> None:
    """Keep EMA BatchNorm buffers identical across ranks without broadcasting weights."""

    if dist.is_available() and dist.is_initialized():
        for buffer in target.buffers():
            dist.broadcast(buffer, src=0)


def _assert_finite_loss(loss: torch.Tensor, metrics: dict[str, float], *, stage: str) -> None:
    if not bool(torch.isfinite(loss.detach()).all()) or any(
        not math.isfinite(float(value)) for value in metrics.values()
    ):
        raise FloatingPointError(f"non-finite {stage} loss/metrics: {metrics}")


def _compute_batch_loss(model, projector, teacher, encoder, target, wm, decoder, value, sigreg, batch, processor, token_ids, device, args):
    items, current, nxt, indices = _encode_batch(model, batch, processor, token_ids, device, args)
    dino_target = _dino_targets(teacher, items, device)
    actions = torch.tensor([item["action_index"] for item in items], device=device, dtype=torch.long)
    values = torch.tensor([item["action_value_target"] for item in items], device=device)
    return compute_grid_sft2_loss(
        current_query_hidden=current,
        next_query_hidden=nxt,
        dino_target_grid=dino_target,
        action_indices=actions,
        value_targets=values,
        slot_projector=projector,
        online_encoder=encoder,
        target_encoder=target,
        grid_wm=wm,
        decoder=decoder,
        value_head=value,
        latent_weight=args.latent_weight,
        dino_weight=args.dino_weight,
        sigreg_weight=args.sigreg_weight,
        value_weight=args.value_weight,
        sigreg_module=sigreg,
        latent_indices=indices,
    )


@torch.no_grad()
def evaluate(model, projector, teacher, encoder, target, wm, decoder, value, sigreg, loader, processor, token_ids, device, args):
    for module in (encoder, target, wm, decoder, value):
        module.eval()
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    for batch in loader:
        loss, metrics = _compute_batch_loss(
            model, projector, teacher, encoder, target, wm, decoder, value, sigreg,
            batch, processor, token_ids, device, args,
        )
        _assert_finite_loss(loss, metrics, stage="validation")
        count = len(batch["items"]) if isinstance(batch, dict) else len(batch)
        totals += torch.tensor(
            [
                metrics["latent_mse"] * count,
                metrics["dino_grid_mse"] * count,
                metrics["sigreg"] * count,
                metrics["value_mse"] * count,
                float(loss.item()) * count,
                count,
            ],
            dtype=torch.float64,
            device=device,
        )
    totals = _reduce(totals)
    count = max(float(totals[-1].item()), 1.0)
    keys = ("latent_mse", "dino_grid_mse", "sigreg", "value_mse", "total")
    return {key: float(totals[index].item() / count) for index, key in enumerate(keys)}


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = setup_dist()
    torch.manual_seed(args.seed + rank)

    processor = AutoProcessor.from_pretrained(str(args.sft1_checkpoint), trust_remote_code=True)
    if add_special_tokens(processor.tokenizer, latent_token_count=16) != 0:
        raise ValueError("SFT1 checkpoint tokenizer is missing one or more 16-query tokens")
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = args.max_pixels
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.sft1_checkpoint),
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    if int(getattr(model.config, "nimloth_latent_token_count", -1)) != 16:
        raise ValueError("SFT1 checkpoint does not declare latent_token_count=16")
    model.requires_grad_(False).eval()
    model.config.use_cache = False
    token_ids = special_token_ids(processor.tokenizer, latent_token_count=16)
    projector = load_sft1_slot_projector(
        args.sft1_checkpoint,
        qwen_hidden_dim=int(model.config.hidden_size),
        state_dim=1024,
        grid_tokens=16,
        map_location=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    projector.requires_grad_(False).eval()
    if args.dino_cache_root is not None:
        identity = resolve_dino_identity(args.dino_model)
        teacher = CachedDINOGridEncoder.from_cache_root(
            args.dino_cache_root,
            identity=identity,
            grid_size=4,
        )
        args.dino_cache_fingerprint = teacher.cache_fingerprint
    else:
        if args.require_dino_cache:
            raise ValueError("--require-dino-cache requires --dino-cache-root")
        teacher = FrozenDINOEncoder.from_pretrained(
            args.dino_model,
            device=device,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        args.dino_cache_fingerprint = None
    if teacher.hidden_size != 1024:
        raise ValueError(f"DINO hidden size must be 1024, got {teacher.hidden_size}")
    args.dino_identity = asdict(teacher.identity)

    encoder = LeWMGridEncoder(emb_dim=1024, hidden_dim=args.encoder_hidden_dim).to(device)
    target = EMATargetGridEncoder(encoder, decay=args.ema_decay).to(device)
    wm = LeWMSpatialPredictor(
        grid_tokens=16,
        emb_dim=1024,
        action_dim=8,
        depth=args.wm_depth,
        heads=args.wm_heads,
        dim_head=args.wm_dim_head,
        mlp_dim=args.wm_mlp_dim,
        dropout=args.wm_dropout,
    ).to(device)
    decoder = LeWMGridDecoder(emb_dim=1024, hidden_dim=args.decoder_hidden_dim).to(device)
    value = ValueHead(emb_dim=1024).to(device)
    sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_num_proj).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": args.encoder_lr},
            {"params": wm.parameters(), "lr": args.wm_lr},
            {"params": decoder.parameters(), "lr": args.decoder_lr},
            {"params": value.parameters(), "lr": args.value_lr},
        ],
        weight_decay=args.weight_decay,
    )
    start_epoch, global_step, best = 1, 0, float("inf")
    if args.resume is not None:
        start_epoch, global_step, best = _load_checkpoint(
            encoder, target, wm, decoder, value, optimizer, args.resume, args
        )

    train_ds, val_ds, train_collate, val_collate = _prepare_datasets(args, processor)
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        collate_fn=train_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_collate,
    )

    if world_size > 1:
        kwargs = {"device_ids": [local_rank]} if device.type == "cuda" else {}
        encoder = DDP(encoder, find_unused_parameters=False, **kwargs)
        wm = DDP(wm, find_unused_parameters=False, **kwargs)
        decoder = DDP(decoder, find_unused_parameters=False, **kwargs)
        value = DDP(value, find_unused_parameters=False, **kwargs)
    if args.resume is None:
        # DDP broadcasts the online encoder from rank 0. Reset the unwrapped EMA
        # copy afterward so every rank starts from that exact synchronized state.
        target.encoder.load_state_dict(_unwrap(encoder).state_dict())

    csv_path = args.output / "train_step_log.csv"
    run = None
    if is_main():
        args.output.mkdir(parents=True, exist_ok=True)
        args.output_dir = args.output
        run = maybe_init_wandb(args)
        if not csv_path.exists():
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    ["global_step", "epoch", "micro_step", "total", "latent_mse", "dino_grid_mse", "sigreg", "value_mse"]
                )
        print(json.dumps({"train_transitions": len(train_ds), "val_transitions": len(val_ds), "world_size": world_size}))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for module in (encoder, wm, decoder, value):
            module.train()
        target.eval()
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch in enumerate(train_loader, start=1):
            loss, metrics = _compute_batch_loss(
                model, projector, teacher, encoder, target, wm, decoder, value, sigreg,
                batch, processor, token_ids, device, args,
            )
            _assert_finite_loss(loss, metrics, stage="training")
            (loss / args.grad_accum).backward()
            if micro_step % args.grad_accum == 0 or micro_step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
                    1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                target.update(_unwrap(encoder))
                _sync_ema_buffers(target)
                global_step += 1
                if is_main():
                    log_train_step(run, global_step, {**metrics, "epoch": float(epoch)})
                    with csv_path.open("a", newline="", encoding="utf-8") as handle:
                        csv.writer(handle).writerow(
                            [global_step, epoch, micro_step, metrics["total"], metrics["latent_mse"], metrics["dino_grid_mse"], metrics["sigreg"], metrics["value_mse"]]
                        )

        val = evaluate(
            model, projector, teacher, encoder, target, wm, decoder, value, sigreg,
            val_loader, processor, token_ids, device, args,
        )
        if is_main():
            print(json.dumps({"epoch": epoch, "step": global_step, **{f"val/{k}": v for k, v in val.items()}}, sort_keys=True))
            log_val_epoch(run, epoch, val, global_step=global_step)
            epoch_dir = args.output / f"epoch_{epoch:03d}"
            new_best = min(best, val["total"])
            _save_checkpoint(encoder, target, wm, decoder, value, optimizer, epoch_dir, epoch=epoch, step=global_step, best=new_best, args=args)
            latest = args.output / "latest"
            if latest.exists():
                shutil.rmtree(latest)
            shutil.copytree(epoch_dir, latest)
            if val["total"] < best:
                best_dir = args.output / "best"
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                shutil.copytree(epoch_dir, best_dir)
            best = new_best
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if is_main():
        _save_checkpoint(encoder, target, wm, decoder, value, optimizer, args.output / "final", epoch=args.epochs, step=global_step, best=best, args=args)
        if run is not None:
            run.finish()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    cleanup_dist()


if __name__ == "__main__":
    main()
