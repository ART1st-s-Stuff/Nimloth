#!/usr/bin/env python3
"""Qwen2.5-VL supervised fine-tuning for Nimloth SFT1 records.

This is intentionally separate from verl's text-only SFTDataset so step 3 can
use the actual rollout screenshots referenced by step 2 records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup

from vagen.envs.navigation.utils.nimloth_format import SPECIAL_TOKENS

import re

NIMLOTH_FORMAT_RE = re.compile(
    r"<think>.*?</think>\s*"
    r"<\|latent_state\|>\s*<\|action_start\|>\s*<\|action_\(\d+\)\|\>\s*<\|action_end\|>",
    re.S,
)


def collect_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
    imgs: list[Image.Image] = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image":
                    path = part["image"]
                    imgs.append(Image.open(path).convert("RGB"))
    return imgs


def _encode_input_ids(
    processor: AutoProcessor,
    messages: list[dict[str, Any]],
    images: list[Image.Image],
    max_length: int,
    *,
    add_generation_prompt: bool,
) -> list[int]:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    enc = processor(
        text=[text],
        images=images or None,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return enc["input_ids"][0].tolist()


def _find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def assistant_token_spans(
    messages: list[dict[str, Any]],
    processor: AutoProcessor,
    max_length: int,
) -> list[tuple[int, int]]:
    """Return assistant token spans aligned to a single full multimodal encoding."""
    images = collect_images(messages)
    full_ids = _encode_input_ids(
        processor, messages, images, max_length, add_generation_prompt=False
    )
    spans: list[tuple[int, int]] = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix_images = collect_images(messages[:i])
        through_images = collect_images(messages[: i + 1])
        prefix_ids = _encode_input_ids(
            processor, messages[:i], prefix_images, max_length, add_generation_prompt=True
        )
        through_ids = _encode_input_ids(
            processor, messages[: i + 1], through_images, max_length, add_generation_prompt=False
        )
        start = len(prefix_ids)
        end = len(through_ids)
        if start >= end:
            continue
        if full_ids[start:end] == through_ids[start:end]:
            spans.append((start, end))
            continue
        if full_ids[:start] == prefix_ids:
            spans.append((start, end))
            continue
        anchor = prefix_ids[-min(32, len(prefix_ids)) :]
        anchor_at = _find_subsequence(full_ids, anchor)
        if anchor_at is None:
            continue
        start = anchor_at + len(anchor)
        end = start + (end - len(prefix_ids))
        if 0 <= start < end <= len(full_ids):
            spans.append((start, end))
    return spans


def collate_fn(batch: list[dict[str, Any]], processor: AutoProcessor, max_length: int) -> dict[str, torch.Tensor]:
    texts: list[str] = []
    spans_per_item: list[list[tuple[int, int]]] = []
    all_images: list[list[Image.Image]] = []
    for item in batch:
        messages = item["messages"]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        spans_per_item.append(assistant_token_spans(messages, processor, max_length))
        all_images.append(collect_images(messages))

    enc = processor(
        text=texts,
        images=all_images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    labels[:] = -100
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    for row, spans in enumerate(spans_per_item):
        for start, end in spans:
            end = min(end, labels.shape[1])
            if start >= end:
                continue
            labels[row, start:end] = enc["input_ids"][row, start:end]
        trained = labels[row][labels[row] != -100]
        if trained.numel() and image_pad_id is not None:
            if (trained == image_pad_id).any().item():
                raise ValueError("supervised labels include <|image_pad|> tokens")
    enc["labels"] = labels
    if torch.all(labels == -100):
        raise ValueError("assistant-only SFT mask is empty for the whole batch")
    return enc


def safe_cache_name(record_id: str) -> str:
    return record_id.replace("/", "__").replace(" ", "_")


def cache_fingerprint(
    jsonl_path: Path,
    max_length: int,
    max_pixels: int,
    min_pixels: int,
    vocab_size: int,
    max_images_per_record: int,
) -> str:
    stat = jsonl_path.stat()
    payload = "|".join(
        [
            str(jsonl_path.resolve()),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            str(max_length),
            str(max_pixels),
            str(min_pixels),
            str(vocab_size),
            str(max_images_per_record),
            "v2",
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def encode_sample_with_labels(
    messages: list[dict[str, Any]],
    processor: AutoProcessor,
    max_length: int,
) -> dict[str, torch.Tensor]:
    images = collect_images(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = processor(
        text=[text],
        images=images or None,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    labels[:] = -100
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    for start, end in assistant_token_spans(messages, processor, max_length):
        end = min(end, labels.shape[1])
        if start >= end:
            continue
        labels[0, start:end] = enc["input_ids"][0, start:end]
    trained = labels[0][labels[0] != -100]
    if trained.numel() and image_pad_id is not None and (trained == image_pad_id).any().item():
        raise ValueError("supervised labels include <|image_pad|> tokens")
    if torch.all(labels == -100):
        raise ValueError("assistant-only SFT mask is empty for sample")
    out: dict[str, torch.Tensor] = {}
    for key, value in enc.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.squeeze(0)
    out["labels"] = labels.squeeze(0)
    return out


def collate_cached_fn(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    if len(batch) == 1:
        return {k: (v.unsqueeze(0) if v.ndim == 1 else v) for k, v in batch[0].items()}
    out: dict[str, torch.Tensor] = {}
    if "input_ids" in batch[0]:
        out["input_ids"] = pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id
        )
    if "attention_mask" in batch[0]:
        out["attention_mask"] = pad_sequence(
            [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
        )
    if "labels" in batch[0]:
        out["labels"] = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    for key in ("pixel_values", "image_grid_thw"):
        if key in batch[0]:
            out[key] = torch.cat([item[key] for item in batch], dim=0)
    return out


_CACHE_PROCESSOR: AutoProcessor | None = None
_CACHE_MAX_LENGTH = 0


def _init_cache_worker(model_path: str, min_pixels: int, max_pixels: int, max_length: int) -> None:
    global _CACHE_PROCESSOR, _CACHE_MAX_LENGTH
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = min_pixels
    processor.image_processor.max_pixels = max_pixels
    processor.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    _CACHE_PROCESSOR = processor
    _CACHE_MAX_LENGTH = max_length


def _cache_one_sample(task: tuple[str, list[dict[str, Any]], str]) -> tuple[str, bool, str]:
    record_id, messages, out_path = task
    try:
        assert _CACHE_PROCESSOR is not None
        encoded = encode_sample_with_labels(messages, _CACHE_PROCESSOR, _CACHE_MAX_LENGTH)
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoded, path)
        return record_id, True, ""
    except Exception as exc:  # noqa: BLE001
        return record_id, False, str(exc)


def nimloth_format_correct(text: str) -> bool:
    return bool(NIMLOTH_FORMAT_RE.search(text))


def prompt_messages_before_first_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg["role"] == "assistant":
            break
        out.append(msg)
    return out


@torch.no_grad()
def evaluate_format(
    model,
    processor: AutoProcessor,
    dataset: NimlothVLSFTDataset,
    device: torch.device,
    max_samples: int = 32,
) -> float:
    if dist.is_available() and dist.is_initialized() and not is_main():
        return 0.0
    module = model.module if hasattr(model, "module") else model
    was_training = module.training
    module.eval()
    correct = 0
    total = 0
    n = min(max_samples, len(dataset))
    for idx in range(n):
        messages = dataset.get_messages(idx)
        prompt_msgs = prompt_messages_before_first_assistant(messages)
        if not prompt_msgs:
            continue
        images = collect_images(prompt_msgs)
        text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images or None, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output_ids = module.generate(**inputs, max_new_tokens=128, do_sample=False)
        new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        decoded = processor.decode(new_ids, skip_special_tokens=False)
        total += 1
        if nimloth_format_correct(decoded):
            correct += 1
    if was_training:
        module.train()
    return correct / max(total, 1)


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


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


def load_jsonl(path: Path, max_records: int = -1) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_records > 0 and len(records) >= max_records:
                break
    return records


class NimlothVLSFTDataset(Dataset):
    def __init__(
        self,
        path: Path,
        processor: AutoProcessor,
        max_records: int = -1,
        max_images_per_record: int = -1,
        *,
        cache_dir: Path | None = None,
    ):
        self.path = path
        self.records = load_jsonl(path, max_records=max_records)
        self.processor = processor
        self.max_images_per_record = max_images_per_record
        self.cache_dir = cache_dir
        self.use_cache = cache_dir is not None

    def __len__(self) -> int:
        return len(self.records)

    def _content_with_images(self, text: str, image_paths_iter) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        chunks = text.split("<image>")
        for i, chunk in enumerate(chunks):
            if chunk:
                parts.append({"type": "text", "text": chunk})
            if i < len(chunks) - 1:
                try:
                    path = next(image_paths_iter)
                except StopIteration as e:
                    raise ValueError("not enough image paths for <image> placeholders") from e
                parts.append({"type": "image", "image": path})
        return parts

    def build_messages(self, rec: dict[str, Any]) -> list[dict[str, Any]]:
        image_paths = list(rec.get("image_paths", []))
        if self.max_images_per_record > 0:
            image_paths = image_paths[: self.max_images_per_record]
        image_iter = iter(image_paths)
        messages = []
        for msg in rec["messages"]:
            content = msg["content"]
            if "<image>" in content:
                content = self._content_with_images(content, image_iter)
            messages.append({"role": msg["role"], "content": content})
        return messages

    def cache_path_for_id(self, record_id: str) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache_dir is not configured")
        return self.cache_dir / f"{safe_cache_name(record_id)}.pt"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        if self.use_cache:
            cache_path = self.cache_path_for_id(rec["id"])
            if not cache_path.is_file():
                raise FileNotFoundError(f"missing preprocess cache: {cache_path}")
            return torch.load(cache_path, map_location="cpu", weights_only=True)
        messages = self.build_messages(rec)
        image_paths = list(rec.get("image_paths", []))
        if self.max_images_per_record > 0:
            image_paths = image_paths[: self.max_images_per_record]
        return {"id": rec["id"], "messages": messages, "num_images": len(image_paths)}

    def get_messages(self, idx: int) -> list[dict[str, Any]]:
        return self.build_messages(self.records[idx])


def build_preprocess_cache(
    dataset: NimlothVLSFTDataset,
    processor: AutoProcessor,
    cache_dir: Path,
    max_length: int,
    model_path: Path,
    min_pixels: int,
    max_pixels: int,
    preprocess_workers: int,
    force: bool = False,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if not force and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("count") == len(dataset) and manifest.get("max_length") == max_length:
            missing = 0
            for rec in dataset.records:
                if not dataset.cache_path_for_id(rec["id"]).is_file():
                    missing += 1
            if missing == 0:
                if is_main():
                    print(json.dumps({"preprocess_cache": "hit", "dir": str(cache_dir), "count": len(dataset)}))
                return

    tasks: list[tuple[str, list[dict[str, Any]], str]] = []
    for rec in dataset.records:
        out_path = dataset.cache_path_for_id(rec["id"])
        if not force and out_path.is_file():
            continue
        tasks.append((rec["id"], dataset.build_messages(rec), str(out_path)))

    if is_main():
        print(
            json.dumps(
                {
                    "preprocess_cache": "build",
                    "dir": str(cache_dir),
                    "total": len(dataset),
                    "to_build": len(tasks),
                    "workers": preprocess_workers,
                }
            )
        )

    if not tasks:
        manifest_path.write_text(
            json.dumps({"count": len(dataset), "max_length": max_length, "dir": str(cache_dir)}, indent=2)
        )
        return

    workers = max(1, preprocess_workers)
    failures: list[str] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_cache_worker,
        initargs=(str(model_path), min_pixels, max_pixels, max_length),
    ) as pool:
        futures = [pool.submit(_cache_one_sample, task) for task in tasks]
        for fut in as_completed(futures):
            record_id, ok, err = fut.result()
            if not ok:
                failures.append(f"{record_id}: {err}")

    if failures:
        raise RuntimeError(f"preprocess cache failed for {len(failures)} samples; first={failures[0]}")

    manifest_path.write_text(
        json.dumps({"count": len(dataset), "max_length": max_length, "dir": str(cache_dir)}, indent=2)
    )
    if is_main():
        print(json.dumps({"preprocess_cache": "done", "dir": str(cache_dir), "count": len(dataset)}))

def save_checkpoint(
    model,
    processor,
    out_dir: Path,
    name: str,
    optimizer=None,
    scheduler=None,
    step: int = 0,
    epoch: int = 0,
    best_val: float = float("inf"),
    *,
    lora: bool = False,
    base_model_path: Path | None = None,
    merge_for_eval: bool = False,
) -> None:
    ckpt = out_dir / name
    ckpt.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    module.save_pretrained(ckpt, safe_serialization=True)
    processor.save_pretrained(ckpt)
    state = {"step": step, "epoch": epoch, "best_val": best_val, "lora": lora}
    if base_model_path is not None:
        state["base_model_path"] = str(base_model_path)
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, ckpt / "training_state.pt")
    if lora and merge_for_eval and is_main() and base_model_path is not None:
        merge_peft_checkpoint(base_model_path, ckpt, ckpt / "hf_merged", processor)


def find_latest_resume_dir(output_dir: Path) -> Path | None:
    """Return the newest epoch_* checkpoint dir, else best/ if present."""
    latest_epoch = -1
    latest_dir: Path | None = None
    for p in output_dir.glob("epoch_*"):
        if not (p / "training_state.pt").is_file():
            continue
        try:
            ep = int(p.name.split("_")[-1])
        except ValueError:
            continue
        if ep > latest_epoch:
            latest_epoch = ep
            latest_dir = p
    if latest_dir is not None:
        return latest_dir
    best = output_dir / "best"
    if (best / "training_state.pt").is_file():
        return best
    return None


def is_peft_model(model: torch.nn.Module) -> bool:
    return hasattr(model, "peft_config") or model.__class__.__name__ == "PeftModel"


def merge_peft_checkpoint(
    base_model_path: Path,
    adapter_path: Path,
    out_path: Path,
    processor: AutoProcessor,
) -> None:
    from peft import PeftModel

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    base.resize_token_embeddings(len(processor.tokenizer))
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    out_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_path, safe_serialization=True)
    processor.save_pretrained(out_path)
    del base, merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_lora(model: Qwen2_5_VLForConditionalGeneration, args: argparse.Namespace):
    from peft import LoraConfig, get_peft_model

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_config)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    if is_main():
        model.print_trainable_parameters()
    return model


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


@torch.no_grad()
def evaluate(model, loader, device: torch.device, max_batches: int = -1) -> float:
    model.eval()
    total = torch.tensor(0.0, device=device)
    count = torch.tensor(0, device=device)
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        loss = model(**batch).loss.detach()
        total += loss
        count += 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    model.train()
    return (total / count.clamp_min(1)).item()


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    embedding_lr: float | None,
    weight_decay: float,
) -> torch.optim.AdamW:
    embed_lr = embedding_lr if embedding_lr is not None else lr
    embed_keys = ("embed_tokens", "lm_head")
    embed_params: list[torch.nn.Parameter] = []
    base_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in embed_keys):
            embed_params.append(param)
        else:
            base_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": base_params, "lr": lr},
            {"params": embed_params, "lr": embed_lr},
        ],
        weight_decay=weight_decay,
    )


def maybe_init_wandb(args: argparse.Namespace) -> Any | None:
    if not is_main() or args.no_wandb:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print(json.dumps({"wandb": "skipped", "reason": "WANDB_API_KEY not set"}))
        return None
    import wandb

    prefix = os.environ.get("WANDB_RUN_PREFIX", "")
    run_name = args.wandb_run_name or f"{prefix}sft1-qwen25vl"
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "nimloth"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=run_name,
        mode=os.environ.get("WANDB_MODE", "online"),
        config={
            "model": str(args.model),
            "train_jsonl": str(args.train_jsonl),
            "val_jsonl": str(args.val_jsonl),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "embedding_lr": args.embedding_lr,
            "max_length": args.max_length,
            "seed": args.seed,
            "lora": args.lora,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "num_workers": args.num_workers,
            "preprocess_workers": args.preprocess_workers,
            "use_cache": not args.no_cache,
        },
    )
    # Train charts use global_step; val/eval charts use epoch (one point per epoch).
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("eval/*", step_metric="epoch")
    return run


def upload_dataset_artifact(
    run: Any,
    train_jsonl: Path,
    val_jsonl: Path,
) -> None:
    import wandb

    artifact = wandb.Artifact(
        name=f"sft1-train-data-{run.id}",
        type="dataset",
        description="Nimloth SFT1 multimodal records used for this run",
    )
    artifact.add_file(str(train_jsonl), name=train_jsonl.name)
    artifact.add_file(str(val_jsonl), name=val_jsonl.name)
    manifest = train_jsonl.parent / "manifest.json"
    if manifest.is_file():
        artifact.add_file(str(manifest), name="manifest.json")
    run.log_artifact(artifact)
    run.summary["train_records"] = sum(1 for _ in train_jsonl.open() if _.strip())
    run.summary["val_records"] = sum(1 for _ in val_jsonl.open() if _.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument(
        "--embedding-lr",
        type=float,
        default=None,
        help="LR for embed_tokens and lm_head (default: same as --lr).",
    )
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=20000)
    ap.add_argument("--max-train-records", type=int, default=-1)
    ap.add_argument("--max-val-records", type=int, default=-1)
    ap.add_argument("--max-val-batches", type=int, default=-1)
    ap.add_argument("--max-images-per-record", type=int, default=-1)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=602112,
        help="Cap vision tokens per image (default ~768*28*28, lower than Qwen2.5-VL factory max).",
    )
    ap.add_argument("--min-pixels", type=int, default=3136, help="Minimum pixels per image (~4*28*28).")
    ap.add_argument("--format-eval-samples", type=int, default=32, help="Val samples for Nimloth format correctness each epoch.")
    ap.add_argument("--wandb-run-name", default=None, help="Optional wandb run name.")
    ap.add_argument("--no-wandb", action="store_true", help="Disable wandb logging and dataset upload.")
    ap.add_argument("--lora", action="store_true", help="Train LoRA adapters (+ embed/lm_head), freeze base weights.")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules (language model blocks).",
    )
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes for cached tensors.")
    ap.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch per worker.")
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
    ap.add_argument("--no-cache", action="store_true", help="Disable preprocess cache and use online collate.")
    ap.add_argument("--rebuild-cache", action="store_true", help="Force rebuild preprocess cache.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rank, world, local_rank, device = setup_dist()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    added = processor.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if is_main():
        print(
            json.dumps(
                {
                    "special_tokens_requested": len(SPECIAL_TOKENS),
                    "special_tokens_newly_added": added,
                    "lr": args.lr,
                    "embedding_lr": args.embedding_lr if args.embedding_lr is not None else args.lr,
                    "lora": args.lora,
                }
            )
        )

    wandb_run = maybe_init_wandb(args)
    if wandb_run is not None:
        upload_dataset_artifact(wandb_run, args.train_jsonl, args.val_jsonl)

    use_cache = not args.no_cache
    cache_root = args.cache_dir or (args.output_dir / "preprocess_cache")
    fp = cache_fingerprint(
        args.train_jsonl,
        args.max_length,
        args.max_pixels,
        args.min_pixels,
        len(processor.tokenizer),
        args.max_images_per_record,
    )
    train_cache_dir = cache_root / f"train_{args.train_jsonl.stem}_{fp}" if use_cache else None
    val_fp = cache_fingerprint(
        args.val_jsonl,
        args.max_length,
        args.max_pixels,
        args.min_pixels,
        len(processor.tokenizer),
        args.max_images_per_record,
    )
    val_cache_dir = cache_root / f"val_{args.val_jsonl.stem}_{val_fp}" if use_cache else None

    train_ds = NimlothVLSFTDataset(
        args.train_jsonl, processor, args.max_train_records, args.max_images_per_record, cache_dir=train_cache_dir
    )
    val_ds = NimlothVLSFTDataset(
        args.val_jsonl, processor, args.max_val_records, args.max_images_per_record, cache_dir=val_cache_dir
    )

    if use_cache:
        if is_main():
            build_preprocess_cache(
                train_ds,
                processor,
                train_cache_dir,
                args.max_length,
                args.model,
                args.min_pixels,
                args.max_pixels,
                args.preprocess_workers,
                force=args.rebuild_cache,
            )
            build_preprocess_cache(
                val_ds,
                processor,
                val_cache_dir,
                args.max_length,
                args.model,
                args.min_pixels,
                args.max_pixels,
                args.preprocess_workers,
                force=args.rebuild_cache,
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    train_collate = (
        (lambda b: collate_cached_fn(b, pad_token_id))
        if use_cache
        else (lambda b: collate_fn(b, processor, args.max_length))
    )
    loader_workers = args.num_workers if use_cache else 0
    loader_kwargs: dict[str, Any] = {
        "num_workers": loader_workers,
        "pin_memory": True,
        "collate_fn": train_collate,
    }
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        **loader_kwargs,
    )

    base_model_path = args.model
    resume_dir: Path | None = find_latest_resume_dir(args.output_dir) if args.resume else None
    resume_ckpt = resume_dir / "training_state.pt" if resume_dir is not None else None
    load_path = args.model
    resume_lora = False
    if args.resume and resume_ckpt is not None and resume_ckpt.exists():
        state_peek = torch.load(resume_ckpt, map_location="cpu")
        resume_lora = bool(state_peek.get("lora")) or (resume_dir / "adapter_config.json").exists()
        if resume_lora:
            load_path = state_peek.get("base_model_path", args.model)
        elif (resume_dir / "config.json").exists():
            load_path = str(resume_dir)
        if is_main():
            print(
                json.dumps(
                    {
                        "resume_load_path": load_path,
                        "resume_lora": resume_lora,
                        "resume_dir": str(resume_dir),
                    }
                )
            )
    elif args.resume and is_main():
        print(json.dumps({"warning": "--resume set but no checkpoint found under output_dir"}))

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.resize_token_embeddings(len(processor.tokenizer))

    if args.resume and resume_ckpt is not None and resume_ckpt.exists() and resume_lora:
        if not args.lora:
            raise ValueError("--resume with LoRA adapter requires --lora")
        model = apply_lora(model, args)
        load_lora_adapter_state(model, resume_dir)
        if args.gradient_checkpointing:
            model.enable_input_require_grads()
    elif args.lora:
        model = apply_lora(model, args)
    elif is_main():
        print(json.dumps({"warning": "full fine-tuning all parameters; pass --lora for adapter-only SFT"}))

    model.to(device)
    optimizer = build_optimizer(model, args.lr, args.embedding_lr, args.weight_decay)
    if world > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        if args.lora:
            # PEFT + gradient checkpointing requires static graph under DDP.
            model._set_static_graph()

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    log_path = args.output_dir / "train_step_log.csv"
    if is_main() and not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["time", "epoch", "global_step", "train_loss", "val_loss", "format_correct_rate", "lr"]
            )

    if is_main() and len(train_ds) > 0:
        if use_cache:
            probe = collate_cached_fn([train_ds[0]], pad_token_id)
        else:
            probe = collate_fn([train_ds[0]], processor, args.max_length)
        trained = probe["labels"][0][probe["labels"][0] != -100]
        decoded = processor.decode(trained.tolist(), skip_special_tokens=False)
        print(json.dumps({"mask_probe": decoded[:500]}))
        if "<|action_start|>" not in decoded:
            raise ValueError("label mask probe missing <|action_start|> in supervised tokens")

    global_step = 0
    best_val = float("inf")
    start_epoch = 1
    if args.resume and resume_ckpt is not None and resume_ckpt.exists():
        state = torch.load(resume_ckpt, map_location="cpu")
        global_step = int(state.get("step", 0))
        best_val = float(state.get("best_val", float("inf")))
        if "epoch" in state:
            start_epoch = int(state["epoch"]) + 1
        else:
            epoch_dirs = sorted(args.output_dir.glob("epoch_*"))
            start_epoch = int(epoch_dirs[-1].name.split("_")[-1]) + 1 if epoch_dirs else 1
        if best_val == float("inf") and log_path.exists():
            rows = list(csv.reader(log_path.open()))
            for row in reversed(rows):
                if len(row) >= 5 and row[4]:
                    try:
                        best_val = min(best_val, float(row[4]))
                    except ValueError:
                        pass
                    break
        if state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if is_main():
            print(
                json.dumps(
                    {
                        "resume": True,
                        "resume_dir": str(resume_dir),
                        "resume_ckpt": str(resume_ckpt),
                        "start_epoch": start_epoch,
                        "global_step": global_step,
                        "best_val": best_val,
                    }
                )
            )

    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        micro_accum = 0

        def optimizer_step(*, micro_count: int) -> None:
            nonlocal global_step, accum_loss
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(micro_count)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            step_loss = accum_loss / micro_count
            if is_main():
                with log_path.open("a", newline="") as f:
                    csv.writer(f).writerow(
                        [time.time(), epoch, global_step, step_loss, "", "", scheduler.get_last_lr()[0]]
                    )
                if wandb_run is not None:
                    import wandb

                    wandb.log(
                        {
                            "train/loss": step_loss,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/embedding_lr": scheduler.get_last_lr()[1]
                            if len(scheduler.get_last_lr()) > 1
                            else scheduler.get_last_lr()[0],
                            "global_step": global_step,
                        },
                        step=global_step,
                    )
            accum_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            accum_loss += loss.detach().float().item()
            micro_accum += 1
            if micro_accum % args.grad_accum == 0:
                optimizer_step(micro_count=micro_accum)
                micro_accum = 0
        if micro_accum > 0:
            optimizer_step(micro_count=micro_accum)

        distributed_barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        val_loss = evaluate(model, val_loader, device, args.max_val_batches)
        format_rate = evaluate_format(model, processor, val_ds, device, args.format_eval_samples)
        if is_main():
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [time.time(), epoch, global_step, "", val_loss, format_rate, scheduler.get_last_lr()[0]]
                )
            save_checkpoint(
                model,
                processor,
                args.output_dir,
                f"epoch_{epoch:03d}",
                optimizer,
                scheduler,
                global_step,
                epoch,
                best_val,
                lora=args.lora,
                base_model_path=base_model_path,
                merge_for_eval=False,
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    model,
                    processor,
                    args.output_dir,
                    "best",
                    optimizer,
                    scheduler,
                    global_step,
                    epoch,
                    best_val,
                    lora=args.lora,
                    base_model_path=base_model_path,
                    merge_for_eval=False,
                )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_loss": val_loss,
                        "format_correct_rate": format_rate,
                        "best_val": best_val,
                    }
                )
            )
            if wandb_run is not None:
                import wandb

                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/format_correct_rate": format_rate,
                        "val/best_loss": best_val,
                        "eval/val_loss": val_loss,
                        "eval/format_correct_rate": format_rate,
                        "epoch": epoch,
                    },
                )
        distributed_barrier()

    if is_main():
        save_checkpoint(
            model,
            processor,
            args.output_dir,
            "final",
            optimizer,
            scheduler,
            global_step,
            args.epochs,
            best_val,
            lora=args.lora,
            base_model_path=base_model_path,
            merge_for_eval=False,
        )
        if wandb_run is not None:
            import wandb

            wandb.finish()
    cleanup_dist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
