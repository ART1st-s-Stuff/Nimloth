#!/usr/bin/env python3
"""SFT2: Qwen latent WM predictor + Value head (see ai_tasks/sft2_exp.md).

Canonical entry for this stage. Defaults: LLM freeze, vision full + EMA.
Config: configs/training/sft2/latent_wm_value.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

_VAGEN_ROOT = Path(__file__).resolve().parents[3] / "external" / "VAGEN"
if _VAGEN_ROOT.is_dir() and str(_VAGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_VAGEN_ROOT))

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, extract_latent_state, find_last_latent_state_index, last_hidden_state, special_token_ids
from nimloth.latent.extraction import LatentActionTokens
from nimloth.training.common.config import merge_cli_over_yaml
from nimloth.training.common.qwen_tuning import configure_qwen_tuning, resolve_tune_modes
from nimloth.training.common.vision_ema import VisionEncoderEMA, resolve_vision_ema
from nimloth.training.common.wandb_logging import log_train_step, log_val_epoch, maybe_init_wandb
from nimloth.training.sft2.cli import parse_sft2_args
from nimloth.training.sft2.metrics import batch_step_success_rate, val_rollout_success_rate
from nimloth.training.sft2 import (
    LatentWMPredictor,
    MetricAccumulator,
    StateProjector,
    ValueHead,
    compute_combined_loss,
    compute_value_loss,
    compute_wm_latent_loss,
    qwen_lr_schedule,
    set_optimizer_group_lr,
    transition_collate_for_qwen,
    wm_loss_weight_schedule,
)
from nimloth.wm import LeWMConfig
from nimloth.wm.dataset import TransitionJsonlDataset, TransitionSample


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def setup_dist() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        dist.init_process_group(backend="nccl")
        return rank, world, local, torch.device(f"cuda:{local}")
    return 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def uses_lora(args: argparse.Namespace) -> bool:
    llm_tune, vision_tune = resolve_tune_modes(args)
    return llm_tune == "lora" or vision_tune == "lora"


def load_lora_adapter_state(model: torch.nn.Module, adapter_dir: Path) -> None:
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if adapter_file.is_file():
        from safetensors.torch import load_file

        state = load_file(str(adapter_file))
    else:
        bin_file = adapter_dir / "adapter_model.bin"
        if not bin_file.is_file():
            raise FileNotFoundError(f"missing adapter weights in {adapter_dir}")
        state = torch.load(bin_file, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if is_main():
        print(
            json.dumps(
                {
                    "resume_load": {
                        "adapter_dir": str(adapter_dir),
                        "missing_keys": len(incompatible.missing_keys),
                        "unexpected_keys": len(incompatible.unexpected_keys),
                    }
                }
            )
        )


class TransitionQwenDataset(Dataset):
    def __init__(self, path: Path, *, max_records: int = -1, success_only: bool = False):
        self.samples = TransitionJsonlDataset(path, max_records=max_records, success_only=success_only).samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TransitionSample:
        return self.samples[index]


def collate_transition_batch(batch: list[TransitionSample]) -> list[dict[str, Any]]:
    return transition_collate_for_qwen(batch)


def assistant_char_spans(messages: list[dict[str, Any]], processor: AutoProcessor) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prev_gen = processor.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
        cur = processor.apply_chat_template(messages[: i + 1], tokenize=False, add_generation_prompt=False)
        start = len(prev_gen)
        end = len(cur)
        if start < end:
            spans.append((start, end))
    return spans


def build_qwen_batch(items: list[dict[str, Any]], processor: AutoProcessor, max_length: int) -> dict[str, torch.Tensor]:
    texts: list[str] = []
    spans_per_item: list[list[tuple[int, int]]] = []
    all_images: list[list[Image.Image]] = []
    for item in items:
        text = processor.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
        texts.append(text)
        spans_per_item.append(assistant_char_spans(item["messages"], processor))
        imgs: list[Image.Image] = []
        for msg in item["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image":
                        imgs.append(Image.open(part["image"]).convert("RGB"))
        all_images.append(imgs)

    enc = processor(
        text=texts,
        images=all_images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    offset_batch = processor.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )["offset_mapping"]
    labels = enc["input_ids"].clone()
    labels[:] = -100
    usable = min(labels.shape[1], offset_batch.shape[1])
    for row, spans in enumerate(spans_per_item):
        for tok_idx in range(usable):
            start = int(offset_batch[row, tok_idx, 0])
            end = int(offset_batch[row, tok_idx, 1])
            if end <= start:
                continue
            if any(start < span_end and end > span_start for span_start, span_end in spans):
                labels[row, tok_idx] = enc["input_ids"][row, tok_idx]
    enc["labels"] = labels
    return enc


def extract_qwen_latents(
    model,
    enc: dict[str, torch.Tensor],
    token_id_map: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    model_inputs = {k: v.to(device) for k, v in enc.items()}
    output = model(**model_inputs, output_hidden_states=True, return_dict=True)
    hidden = last_hidden_state(output)
    tokens = LatentActionTokens()
    rows: list[torch.Tensor] = []
    for row in range(hidden.shape[0]):
        latent_index = find_last_latent_state_index(enc["input_ids"][row], token_id_map, tokens)
        rows.append(extract_latent_state(hidden[row : row + 1], latent_index))
    return torch.stack(rows, dim=0), output.loss


def wm_eligible_indices(items: list[dict[str, Any]]) -> list[int]:
    return [i for i, item in enumerate(items) if item.get("next_messages")]


def _ddp_sync_zero(module: torch.nn.Module) -> torch.Tensor:
    """Scalar zero tied to module params so every rank enters the same autograd subgraph."""

    return sum(p.sum() for p in module.parameters()) * 0.0


def compute_step_wm_loss(
    model,
    items: list[dict[str, Any]],
    current_latent: torch.Tensor,
    processor: AutoProcessor,
    token_id_map: dict[str, int],
    device: torch.device,
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    max_length: int,
    *,
    vision_ema: VisionEncoderEMA | None = None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    indices = wm_eligible_indices(items)
    if not indices:
        return _ddp_sync_zero(wm_predictor) + _ddp_sync_zero(state_proj), {}
    next_items = [{"messages": items[i]["next_messages"]} for i in indices]
    next_enc = build_qwen_batch(next_items, processor, max_length)
    ema_ctx = vision_ema.use_ema_weights(model) if vision_ema is not None else contextlib.nullcontext()
    with torch.no_grad(), ema_ctx:
        next_latent, _ = extract_qwen_latents(model, next_enc, token_id_map, device)
    action_indices = torch.tensor([items[i]["action_index"] for i in indices], device=device)
    return compute_wm_latent_loss(
        qwen_hidden_at_latent=current_latent[indices],
        qwen_hidden_at_next_latent=next_latent,
        action_indices=action_indices,
        state_proj=state_proj,
        wm_predictor=wm_predictor,
    )


def compute_step_value_loss(
    current_latent: torch.Tensor,
    items: list[dict[str, Any]],
    state_proj: StateProjector,
    value_head: ValueHead,
    device: torch.device,
    *,
    rank_margin: float,
    lambda_rank: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    action_indices = torch.tensor([item["action_index"] for item in items], device=device, dtype=torch.long)
    targets = torch.tensor(
        [float(item["action_value_target"]) for item in items],
        device=device,
        dtype=torch.float32,
    )
    state_emb = state_proj(current_latent)
    return compute_value_loss(
        state_emb=state_emb,
        action_indices=action_indices,
        action_value_targets=targets,
        value_head=value_head,
        rank_margin=rank_margin,
        lambda_rank=lambda_rank,
    )


def save_checkpoint(
    model,
    state_proj,
    processor,
    out_dir: Path,
    *,
    wm_predictor: LatentWMPredictor | None = None,
    value_head: ValueHead | None = None,
    vision_ema: VisionEncoderEMA | None = None,
    optimizer=None,
    step: int = 0,
    epoch: int = 0,
    best_val_success_rate: float = -1.0,
    best_val_wm_mse: float = float("inf"),
    lora: bool = False,
    base_model_path: Path | None = None,
    llm_tune: str = "freeze",
    vision_tune: str = "freeze",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    module.save_pretrained(out_dir, safe_serialization=True)
    processor.save_pretrained(out_dir)
    proj = state_proj.module if hasattr(state_proj, "module") else state_proj
    torch.save(proj.state_dict(), out_dir / "state_proj.pt")
    if wm_predictor is not None:
        pred = wm_predictor.module if hasattr(wm_predictor, "module") else wm_predictor
        pred.save_checkpoint(out_dir / "wm_predictor")
    if value_head is not None:
        head = value_head.module if hasattr(value_head, "module") else value_head
        head.save_checkpoint(out_dir / "value_head")
    if vision_ema is not None and vision_ema.shadow:
        vision_ema.save_checkpoint(out_dir / "vision_ema.pt")
    state: dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "best_val_success_rate": best_val_success_rate,
        "best_val_wm_mse": best_val_wm_mse,
        "best_val": best_val_wm_mse,
        "lora": lora,
        "llm_tune": llm_tune,
        "vision_tune": vision_tune,
        "vision_ema": vision_ema is not None and bool(vision_ema.shadow),
    }
    if base_model_path is not None:
        state["base_model_path"] = str(base_model_path)
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, out_dir / "training_state.pt")


def load_aux_checkpoint(
    ckpt_dir: Path,
    state_proj,
    wm_predictor: LatentWMPredictor,
    value_head: ValueHead,
    device: torch.device,
) -> None:
    sp_path = ckpt_dir / "state_proj.pt"
    if sp_path.is_file():
        proj = state_proj.module if hasattr(state_proj, "module") else state_proj
        proj.load_state_dict(torch.load(sp_path, map_location=device, weights_only=True))
    pred_path = ckpt_dir / "wm_predictor"
    if pred_path.is_dir():
        pred = wm_predictor.module if hasattr(wm_predictor, "module") else wm_predictor
        loaded = LatentWMPredictor.load_checkpoint(pred_path, map_location=device)
        pred.load_state_dict(loaded.state_dict())
    head_path = ckpt_dir / "value_head"
    if head_path.is_dir():
        head = value_head.module if hasattr(value_head, "module") else value_head
        loaded_head = ValueHead.load_checkpoint(
            head_path,
            emb_dim=head.net[0].in_features,
            map_location=device,
        )
        head.load_state_dict(loaded_head.state_dict())


@torch.no_grad()
def evaluate(
    model,
    state_proj,
    wm_predictor,
    value_head,
    loader,
    processor,
    token_id_map,
    device,
    *,
    max_batches: int = -1,
    max_length: int = 20000,
    vision_ema: VisionEncoderEMA | None = None,
) -> dict[str, float]:
    model.eval()
    state_proj.eval()
    wm_predictor.eval()
    value_head.eval()
    acc = MetricAccumulator()
    ema_ctx = vision_ema.use_ema_weights(model) if vision_ema is not None else contextlib.nullcontext()
    with ema_ctx:
        for i, batch_samples in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            items = batch_samples
            enc = build_qwen_batch(items, processor, max_length=max_length)
            latent_hidden, _ = extract_qwen_latents(model, enc, token_id_map, device)
            _, wm_metrics = compute_step_wm_loss(
                model,
                items,
                latent_hidden,
                processor,
                token_id_map,
                device,
                state_proj,
                wm_predictor,
                max_length,
                vision_ema=vision_ema,
            )
            _, value_metrics = compute_step_value_loss(
                latent_hidden,
                items,
                state_proj,
                value_head,
                device,
                rank_margin=0.0,
                lambda_rank=0.0,
            )
            success_rate = batch_step_success_rate(items)
            acc.update({**wm_metrics, **value_metrics, "success_rate": success_rate})

    model.train()
    state_proj.train()
    wm_predictor.train()
    value_head.train()
    return acc.averages()


def main() -> int:
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
        model.gradient_checkpointing_enable()
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
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if uses_lora(args):
            model._set_static_graph()
        state_proj = DDP(state_proj, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        value_head = DDP(value_head, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if train_wm_predictor:
            wm_predictor = DDP(wm_predictor, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

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

    def _unwrap(module):
        return module.module if hasattr(module, "module") else module

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
                _unwrap(state_proj),
                _unwrap(wm_predictor),
                args.max_length,
                vision_ema=vision_ema,
            )
            value_loss, value_metrics = compute_step_value_loss(
                latent_hidden,
                items,
                _unwrap(state_proj),
                _unwrap(value_head),
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


if __name__ == "__main__":
    raise SystemExit(main())
