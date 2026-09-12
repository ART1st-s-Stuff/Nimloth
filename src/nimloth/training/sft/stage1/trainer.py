"""Shared early-stage lifecycle: format CE or current-observation query alignment.

Data/masking live in data.py; the query objective lives in stage2/model.py.
This loop owns model tuning, accumulation, validation and epoch checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import signal
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from nimloth.agent.evaluation_protocol import EarlyProtocol
from nimloth.backbone.qwen25vl.early_generation import (
    Stage1RawGenerationValidation,
    validate_stage1_sampled_tokens,
)
from nimloth.latent import (
    add_special_tokens,
    initialize_extra_latent_token_embeddings,
    latent_state_block,
    special_token_ids,
)

from .checkpoint import (
    RESUME_SCHEMA,
    capture_rng_state,
    find_latest_resume_dir,
    load_lora_adapter_state,
    objective_identities_match,
    restore_rng_state,
    save_checkpoint,
    save_resume_checkpoint,
    validate_resume_stage,
    validate_resume_state,
)
from .cli import parse_args
from .convergence import ConvergencePolicy, ConvergenceState
from .data import (
    CACHE_SCHEMA,
    FORMAT_OBJECTIVE,
    NimlothVLSFTDataset,
    build_preprocess_cache,
    cache_fingerprint,
    collate_cached_fn,
    collate_fn,
    collect_images,
    render_stage_text,
)
from .distributed import cleanup_dist, distributed_barrier, is_main, setup_dist
from .fsdp import (
    clip_grad_norm,
    generation_model,
    is_fsdp,
    load_optimizer_state,
    wrap_fsdp,
)
from .loss import (
    ACTION_TOKEN_LOSS_SCOPE,
    BOUNDARY_TOKEN_LOSS_SCOPE,
    resolve_boundary_token_ids,
    resolve_action_number_token_ids,
    training_loss,
    weighted_answer_loss,
)


def nimloth_format_correct(text: str, *, latent_token_count: int | None = None) -> bool:
    stage = "stage1" if latent_token_count is None else "stage2"
    return bool(
        EarlyProtocol(stage, latent_token_count, "generate")
        .parse(text)["format_correct"]
    )


def validate_stage1_generated_response(
    generated_token_ids: torch.Tensor | list[int],
    tokenizer: Any,
    *,
    max_new_tokens: int = 128,
) -> Stage1RawGenerationValidation:
    """Adapt HF generation to the shared Stage 1 sampled-token validator."""
    token_ids = [int(token_id) for token_id in generated_token_ids]
    eos_token_id = tokenizer.eos_token_id
    eos_generated = isinstance(eos_token_id, int) and eos_token_id in token_ids
    finish_reason = (
        "length"
        if len(token_ids) >= max_new_tokens and not eos_generated
        else "stop"
    )
    return validate_stage1_sampled_tokens(
        token_ids,
        tokenizer,
        finish_reason=finish_reason,
    )


def action_block_format_correct(text: str) -> bool:
    return bool(
        re.search(
            r"<\|action_start\|>\s*<\|action_\(\d+\)\|>\s*<\|action_end\|>",
            text,
            re.DOTALL,
        )
    )


def prompt_messages_before_first_assistant(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
    *,
    batch_size: int = 1,
    latent_token_count: int | None = None,
    latent_query_mode: str | None = None,
) -> tuple[float, dict[str, int], list[dict[str, Any]]]:
    if dist.is_available() and dist.is_initialized() and not is_main() and not is_fsdp(model):
        return 0.0, {}, []
    module = model.module if hasattr(model, "module") else model
    was_training = module.training
    correct = 0
    total = 0
    reasons: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    strict_stage1 = latent_query_mode is None
    if strict_stage1 and max_samples != 32:
        raise ValueError("Stage 1 format evaluation requires exactly 32 samples")
    if strict_stage1 and len(dataset) < max_samples:
        raise ValueError(
            f"Stage 1 format-eval dataset has {len(dataset)} records; need 32"
        )
    if batch_size < 1:
        raise ValueError("format evaluation batch_size must be >= 1")
    n = max_samples if strict_stage1 else min(max_samples, len(dataset))
    prepared: list[dict[str, Any]] = []
    for idx in range(n):
        messages = dataset.get_messages(idx)
        prompt_msgs = prompt_messages_before_first_assistant(messages)
        if not prompt_msgs:
            if strict_stage1:
                raise ValueError(
                    f"Stage 1 format-eval record {dataset.records[idx]['id']!r} "
                    "has no prompt before the first assistant turn"
                )
            continue
        images = collect_images(prompt_msgs)
        text = processor.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        if latent_query_mode == "inject":
            first_assistant = next(
                (m for m in messages if m["role"] == "assistant"), None
            )
            if first_assistant is None:
                continue
            content = str(first_assistant["content"])
            think_match = re.search(r"<think>.*?</think>", content, re.DOTALL)
            if think_match is None:
                continue
            # Isolate action formatting from thought generation: teacher-force
            # the reference thought, inject deterministic query slots, then ask
            # the model to generate only the action block.
            text += think_match.group(0) + latent_state_block(latent_token_count)
        text = render_stage_text(text, latent_token_count)
        prepared.append(
            {
                "dataset_index": idx,
                "record_id": str(dataset.records[idx]["id"]),
                "text": text,
                "images": images,
            }
        )

    tokenizer = processor.tokenizer
    original_padding_side = tokenizer.padding_side
    started_at = time.monotonic()
    batch_count = math.ceil(len(prepared) / batch_size)
    module.eval()
    try:
        tokenizer.padding_side = "left"
        with generation_model(model) as generation_module:
            for batch_index, start in enumerate(
                range(0, len(prepared), batch_size), 1
            ):
                batch = prepared[start : start + batch_size]
                batch_images = [item["images"] for item in batch]
                inputs = processor(
                    text=[item["text"] for item in batch],
                    # Qwen flattens nested image lists in prompt order.  Keep the
                    # per-prompt grouping whenever the batch contains images,
                    # but retain the previous text-only ``images=None`` path.
                    images=batch_images if any(batch_images) else None,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                output_ids = generation_module.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    **({"synced_gpus": True} if is_fsdp(model) else {}),
                )
                prompt_width = inputs["input_ids"].shape[1]
                for row, item in enumerate(batch):
                    new_ids = output_ids[row, prompt_width:]
                    eos_token_id = tokenizer.eos_token_id
                    if isinstance(eos_token_id, int):
                        eos_positions = (new_ids == eos_token_id).nonzero(
                            as_tuple=False
                        )
                        if eos_positions.numel():
                            eos_end = int(eos_positions[0].item()) + 1
                            suffix = new_ids[eos_end:]
                            pad_token_id = tokenizer.pad_token_id
                            if (
                                suffix.numel()
                                and isinstance(pad_token_id, int)
                                and bool(torch.all(suffix == pad_token_id).item())
                            ):
                                new_ids = new_ids[:eos_end]
                    total += 1
                    if latent_query_mode == "inject":
                        decoded = processor.decode(new_ids, skip_special_tokens=False)
                        is_correct = action_block_format_correct(decoded)
                        correct += int(is_correct)
                        reason = "ok" if is_correct else "invalid_action_block"
                        reasons[reason] += 1
                        raw_response = parsed_body = decoded
                    elif latent_query_mode is not None:
                        decoded = processor.decode(new_ids, skip_special_tokens=False)
                        is_correct = nimloth_format_correct(
                            decoded, latent_token_count=latent_token_count
                        )
                        correct += int(is_correct)
                        reason = "ok" if is_correct else "invalid_response_envelope"
                        reasons[reason] += 1
                        raw_response = parsed_body = decoded
                    else:
                        result = validate_stage1_generated_response(new_ids, tokenizer)
                        correct += int(result.format_correct)
                        reasons[result.reason] += 1
                        is_correct = result.format_correct
                        reason = result.reason
                        raw_response = result.raw_response
                        parsed_body = result.parsed_body
                    sampled_token_ids = [int(token_id) for token_id in new_ids]
                    samples.append(
                        {
                            "dataset_index": item["dataset_index"],
                            "record_id": item["record_id"],
                            "sampled_token_ids": sampled_token_ids,
                            "raw_response": raw_response,
                            "parsed_body": parsed_body,
                            "eos_generated": (
                                isinstance(eos_token_id, int)
                                and eos_token_id in sampled_token_ids
                            ),
                            "format_correct": is_correct,
                            "reason": reason,
                        }
                    )
                if is_main():
                    print(
                        json.dumps(
                            {
                                "event": "format_eval_batch",
                                "batch": batch_index,
                                "batches": batch_count,
                                "completed_samples": min(
                                    start + len(batch), len(prepared)
                                ),
                                "total_samples": len(prepared),
                                "elapsed_seconds": time.monotonic() - started_at,
                            }
                        ),
                        flush=True,
                    )
    finally:
        tokenizer.padding_side = original_padding_side
        if was_training:
            module.train()
    return correct / max(total, 1), dict(sorted(reasons.items())), samples


def is_peft_model(model: torch.nn.Module) -> bool:
    return hasattr(model, "peft_config") or model.__class__.__name__ == "PeftModel"


def resize_token_embeddings_and_sync_vocab(
    model: torch.nn.Module, vocab_size: int
) -> None:
    model.resize_token_embeddings(vocab_size)
    config = model.config
    config.vocab_size = vocab_size
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config.vocab_size = vocab_size
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.vocab_size = vocab_size


def prepare_query_vocabulary(
    model: torch.nn.Module,
    vocabulary_size: int,
    token_id_map: dict[str, int],
    *,
    added_tokens: int,
    latent_token_count: int | None,
) -> None:
    """Extend a base model without resetting trained query rows on full resume."""
    vocabulary_grows = model.get_input_embeddings().weight.shape[0] < vocabulary_size
    resize_token_embeddings_and_sync_vocab(model, vocabulary_size)
    if latent_token_count is not None and added_tokens > 0 and vocabulary_grows:
        initialize_extra_latent_token_embeddings(
            model,
            token_id_map,
            latent_token_count=latent_token_count,
        )


def enable_gradient_checkpointing(model) -> None:
    # 视觉输入无需梯度，但视觉 LoRA 参数仍必须参与反向传播。
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )


def apply_lora(model: Qwen2_5_VLForConditionalGeneration, args: argparse.Namespace):
    from peft import LoraConfig, get_peft_model

    target_modules = [
        m.strip() for m in args.lora_target_modules.split(",") if m.strip()
    ]
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


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    max_batches: int = -1,
    *,
    return_components: bool = False,
    include_batches: list[bool] | None = None,
    action_token_ids: tuple[int, ...] = (),
    boundary_token_ids: tuple[int, ...] = (),
    action_weight: float = 1.0,
    boundary_weight: float = 1.0,
) -> float | dict[str, float]:
    model.eval()
    weighted = bool(action_token_ids)
    if weighted and return_components:
        raise ValueError("weighted validation is only supported for Stage 1")
    total = torch.zeros(3 if return_components else (2 if weighted else 1), device=device)
    count = torch.tensor(0, device=device)
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        output = model(**batch)
        if include_batches is not None and not include_batches[i]:
            continue
        losses = [output.loss.detach()]
        if return_components:
            losses.extend([output.lm_loss.detach(), output.dino_loss.detach()])
        if weighted:
            losses.append(weighted_answer_loss(output.logits, batch["labels"], action_token_ids, action_weight,
                boundary_token_ids=boundary_token_ids, boundary_weight=boundary_weight))
        total += torch.stack(losses)
        count += 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    model.train()
    if count.item() == 0:
        raise ValueError("validation loader produced no monitored loss")
    means = total / count
    if return_components:
        return dict(zip(
            ("validation_total_loss", "validation_lm_loss", "validation_dino_loss"),
            means.tolist(), strict=True,
        ))
    if weighted:
        return {"validation_lm_loss": means[0].item(), "validation_weighted_lm_loss": means[1].item()}
    return means.item()


def distributed_validation_inclusion(
    dataset_size: int, *, world: int, rank: int
) -> list[bool]:
    """Exclude DistributedSampler padding while retaining equal forward counts."""
    if dataset_size < 1 or world < 1 or rank not in range(world):
        raise ValueError("invalid validation dataset/distributed identity")
    per_rank = math.ceil(dataset_size / world)
    padded_size = per_rank * world
    inclusion = [True] * dataset_size + [False] * (padded_size - dataset_size)
    return inclusion[rank:padded_size:world]


def validation_sampling_contract(
    stage: str,
    *,
    dataset_size: int,
    world: int,
    rank: int,
    configured_batch_size: int,
) -> tuple[int, list[bool] | None]:
    """Keep exact-record Stage 1 validation isolated from Stage 2 batching."""

    if stage == "format":
        return 1, distributed_validation_inclusion(
            dataset_size, world=world, rank=rank
        )
    if stage == "query":
        return configured_batch_size, None
    raise ValueError(f"unknown training stage: {stage}")


def convergence_monitor(stage: str) -> str:
    return "validation_total_loss" if stage == "query" else "validation_lm_loss"


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
        foreach=False,
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
    run_id_path = args.output_dir / "wandb_run_id.txt"
    requested_run_id = os.environ.get("WANDB_RUN_ID")
    if requested_run_id is None and run_id_path.is_file():
        requested_run_id = run_id_path.read_text(encoding="utf-8").strip() or None
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "nimloth-sft1"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=run_name,
        id=requested_run_id,
        resume="allow" if requested_run_id is not None else None,
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
            "action_token_loss_weight": args.action_token_loss_weight,
            "boundary_token_loss_weight": args.boundary_token_loss_weight,
            "format_eval_samples": args.format_eval_samples,
            "format_eval_batch_size": args.format_eval_batch_size,
            "max_length": args.max_length,
            "seed": args.seed,
            "lora": args.lora,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "num_workers": args.num_workers,
            "preprocess_workers": args.preprocess_workers,
            "use_cache": not args.no_cache,
            "latent_token_count": args.latent_token_count,
            "latent_query_mode": args.latent_query_mode,
            "mask_latent_query_labels": args.mask_latent_query_labels,
        },
    )
    run_id_path.write_text(f"{run.id}\n", encoding="utf-8")
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
    format_eval_jsonl: Path | None = None,
) -> None:
    import wandb

    artifact = wandb.Artifact(
        name=f"sft1-train-data-{run.id}",
        type="dataset",
        description="Nimloth SFT1 multimodal records used for this run",
    )
    artifact.add_file(str(train_jsonl), name=train_jsonl.name)
    artifact.add_file(str(val_jsonl), name=val_jsonl.name)
    if format_eval_jsonl is not None:
        artifact.add_file(str(format_eval_jsonl), name=format_eval_jsonl.name)
    manifest = train_jsonl.parent / "manifest.json"
    if manifest.is_file():
        artifact.add_file(str(manifest), name="manifest.json")
    run.log_artifact(artifact)
    run.summary["train_records"] = sum(1 for _ in train_jsonl.open() if _.strip())
    run.summary["val_records"] = sum(1 for _ in val_jsonl.open() if _.strip())
    if format_eval_jsonl is not None:
        run.summary["format_eval_records"] = sum(
            1 for line in format_eval_jsonl.open() if line.strip()
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_format_samples(
    output_dir: Path,
    *,
    epoch: int,
    global_step: int,
    source: Path,
    samples: list[dict[str, Any]],
) -> Path:
    directory = output_dir / "format_eval"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"epoch_{epoch:03d}_step_{global_step:08d}.jsonl"
    payload = "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"refusing to replace different format samples: {path}")
    else:
        path.write_text(payload, encoding="utf-8")
    manifest = {
        "epoch": epoch,
        "global_step": global_step,
        "source": str(source.resolve()),
        "source_sha256": _file_sha256(source),
        "samples": len(samples),
        "denominator": len(samples),
        "sample_record_ids": [sample["record_id"] for sample in samples],
        "jsonl": str(path),
        "jsonl_sha256": _file_sha256(path),
    }
    manifest_path = path.with_suffix(".manifest.json")
    manifest_payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != manifest_payload:
            raise FileExistsError(
                f"refusing to replace different format manifest: {manifest_path}"
            )
    else:
        manifest_path.write_text(manifest_payload, encoding="utf-8")
    return path


def _resume_identity(
    args: argparse.Namespace, *, stage: str, world: int, train_size: int
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "stage": stage,
        "format_objective": FORMAT_OBJECTIVE if stage == "format" else None,
        "action_token_loss_weight": getattr(args, "action_token_loss_weight", 1.0),
        "boundary_token_loss_weight": getattr(args, "boundary_token_loss_weight", 1.0),
        "world_size": world,
        "model": str(Path(args.model).resolve()),
        "train_jsonl": str(args.train_jsonl.resolve()),
        "train_jsonl_sha256": _file_sha256(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl.resolve()),
        "val_jsonl_sha256": _file_sha256(args.val_jsonl),
        "train_size": train_size,
        "max_train_records": args.max_train_records,
        "max_val_records": args.max_val_records,
        "max_images_per_record": args.max_images_per_record,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "embedding_lr": args.embedding_lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
        "latent_token_count": args.latent_token_count,
        "latent_query_mode": args.latent_query_mode,
        "gradient_checkpointing": args.gradient_checkpointing,
        "attn_implementation": args.attn_implementation,
        "preprocess_cache_fingerprint": args.train_cache_fingerprint,
        "lora": args.lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
    }
    if stage == "format":
        identity.update(
            {
                "action_token_loss_scope": ACTION_TOKEN_LOSS_SCOPE,
                "boundary_token_loss_scope": BOUNDARY_TOKEN_LOSS_SCOPE,
                "format_eval_jsonl": str(args.format_eval_jsonl.resolve()),
                "format_eval_jsonl_sha256": _file_sha256(
                    args.format_eval_jsonl
                ),
            }
        )
    if getattr(args, "distributed_strategy", "ddp") == "fsdp":
        identity["distributed_strategy"] = "fsdp_full_shard_orig_params_v1"
    if getattr(args, "until_converged", False):
        identity["convergence"] = {
            "monitor": convergence_monitor(stage),
            "min_epochs": args.convergence_min_epochs,
            "patience_epochs": args.convergence_patience_epochs,
            "min_relative_improvement": args.convergence_min_relative_improvement,
            "scheduler": "constant_with_warmup_first_epoch",
        }
    if stage == "query":
        identity.update(
            {
                "dino_cache_root": str(args.dino_cache_root.resolve()),
                "dino_cache_fingerprint": args.dino_cache_fingerprint,
                "grid_size": args.grid_size,
                "projector_hidden_dim": args.projector_hidden_dim,
                "weight_lm": args.weight_lm,
                "weight_dino": args.weight_dino,
            }
        )
    return identity


def main(*, stage: str = "format") -> int:
    args, query_config = parse_args(stage=stage)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rank, world, local_rank, device = setup_dist()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    added = add_special_tokens(
        processor.tokenizer, latent_token_count=args.latent_token_count
    )
    token_id_map = special_token_ids(
        processor.tokenizer, latent_token_count=args.latent_token_count
    )
    action_ids = (
        resolve_action_number_token_ids(processor.tokenizer)
        if stage == "format"
        else ()
    )
    boundary_ids = resolve_boundary_token_ids(processor.tokenizer) if stage == "format" else ()
    if is_main():
        print(json.dumps({"boundary_token_loss_weight": args.boundary_token_loss_weight, "weighted_boundary_token_ids": boundary_ids, "action_token_loss_weight": args.action_token_loss_weight, "weighted_action_token_ids": action_ids}))
        print(
            json.dumps(
                {
                    "special_tokens_requested": len(token_id_map),
                    "special_tokens_newly_added": added,
                    "latent_token_count": args.latent_token_count,
                    "latent_query_mode": args.latent_query_mode,
                    "mask_latent_query_labels": args.mask_latent_query_labels,
                    "lr": args.lr,
                    "embedding_lr": args.embedding_lr
                    if args.embedding_lr is not None
                    else args.lr,
                    "lora": args.lora,
                    "cache_pixel_dtype": args.cache_pixel_dtype,
                    "format_eval_samples": args.format_eval_samples,
                    "format_eval_batch_size": args.format_eval_batch_size,
                }
            )
        )

    wandb_run = None
    if not args.cache_only:
        wandb_run = maybe_init_wandb(args)
        if wandb_run is not None:
            upload_dataset_artifact(
                wandb_run,
                args.train_jsonl,
                args.val_jsonl,
                args.format_eval_jsonl if stage == "format" else None,
            )

    use_cache = not args.no_cache
    if args.cache_only and not use_cache:
        raise ValueError("--cache-only cannot be combined with --no-cache")
    cache_root = args.cache_dir or (args.output_dir / "preprocess_cache")
    fp = cache_fingerprint(
        args.train_jsonl,
        args.max_length,
        args.max_pixels,
        args.min_pixels,
        len(processor.tokenizer),
        args.max_images_per_record,
        args.latent_token_count,
        args.mask_latent_query_labels,
        args.cache_pixel_dtype,
        str(Path(args.model).resolve()),
        latent_query_mode=args.latent_query_mode,
    )
    train_cache_dir = (
        cache_root / f"train_{args.train_jsonl.stem}_{fp}" if use_cache else None
    )
    val_fp = cache_fingerprint(
        args.val_jsonl,
        args.max_length,
        args.max_pixels,
        args.min_pixels,
        len(processor.tokenizer),
        args.max_images_per_record,
        args.latent_token_count,
        args.mask_latent_query_labels,
        args.cache_pixel_dtype,
        str(Path(args.model).resolve()),
        latent_query_mode=args.latent_query_mode,
    )
    val_cache_dir = (
        cache_root / f"val_{args.val_jsonl.stem}_{val_fp}" if use_cache else None
    )
    args.train_cache_fingerprint = fp if use_cache else None

    train_ds = NimlothVLSFTDataset(
        args.train_jsonl,
        processor,
        args.max_train_records,
        args.max_images_per_record,
        cache_dir=train_cache_dir,
    )
    val_ds = NimlothVLSFTDataset(
        args.val_jsonl,
        processor,
        args.max_val_records,
        args.max_images_per_record,
        cache_dir=val_cache_dir,
    )
    format_eval_ds = (
        NimlothVLSFTDataset(
            args.format_eval_jsonl,
            processor,
            -1,
            args.max_images_per_record,
        )
        if stage == "format"
        else val_ds
    )

    if use_cache:
        if is_main() and not args.require_prebuilt_cache:
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
                latent_token_count=args.latent_token_count,
                mask_latent_query_labels=args.mask_latent_query_labels,
                cache_pixel_dtype=args.cache_pixel_dtype,
                latent_query_mode=args.latent_query_mode,
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
                latent_token_count=args.latent_token_count,
                mask_latent_query_labels=args.mask_latent_query_labels,
                cache_pixel_dtype=args.cache_pixel_dtype,
                latent_query_mode=args.latent_query_mode,
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        for required_dir in (train_cache_dir, val_cache_dir):
            manifest_path = required_dir / "manifest.json"
            if not manifest_path.is_file():
                mode = "required prebuilt" if args.require_prebuilt_cache else "built"
                raise FileNotFoundError(
                    f"{mode} SFT1 preprocess cache missing manifest: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get("cache_schema") != CACHE_SCHEMA
                or manifest.get("format_objective") != FORMAT_OBJECTIVE
                or manifest.get("latent_token_count") is not None
                or manifest.get("latent_query_mode") is not None):
                raise ValueError(f"incompatible format-only cache: {manifest_path}")
        if args.cache_only:
            if is_main():
                print(
                    json.dumps(
                        {"preprocess_cache": "cache_only_done", "root": str(cache_root)}
                    )
                )
            cleanup_dist()
            return 0

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    train_collate = (
        (lambda b: collate_cached_fn(b, pad_token_id))
        if use_cache
        else (
            lambda b: collate_fn(
                b,
                processor,
                args.max_length,
                latent_token_count=args.latent_token_count,
                mask_latent_query_labels=args.mask_latent_query_labels,
            )
        )
    )
    if query_config is not None:
        from nimloth.backbone.dino_grid import (
            DINOV2_LARGE_IDENTITY,
            CachedDINOGridTargets,
        )
        from nimloth.training.sft.stage2.data import (
            AnswerPrefixDataset,
            QueryAlignmentCollator,
        )

        targets = CachedDINOGridTargets.from_cache_root(
            args.dino_cache_root,
            identity=DINOV2_LARGE_IDENTITY,
            grid_size=query_config.grid_size,
        )
        args.dino_cache_fingerprint = targets.cache_fingerprint
        train_ds = AnswerPrefixDataset(train_ds)
        val_ds = AnswerPrefixDataset(val_ds)
        train_collate = QueryAlignmentCollator(
            processor,
            args.max_length,
            query_config.grid_tokens,
            targets,
            mask_latent_query_labels=args.mask_latent_query_labels,
            last_answer_only=True,
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

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
        if world > 1
        else None
    )
    val_batch_size, val_include_batches = validation_sampling_contract(
        stage,
        dataset_size=len(val_ds),
        world=world,
        rank=rank,
        configured_batch_size=args.batch_size,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        # Stage 1 uses a scalar loss per record so padded cross-rank forwards
        # retain collective order without entering the exact-record mean.
        # Stage 2 retains its established batched aggregate and throughput.
        batch_size=val_batch_size,
        sampler=val_sampler,
        shuffle=False,
        **loader_kwargs,
    )

    base_model_path = args.model
    resume_dir: Path | None = (
        find_latest_resume_dir(args.output_dir) if args.resume else None
    )
    resume_ckpt = resume_dir / "training_state.pt" if resume_dir is not None else None
    load_path = args.model
    resume_lora = False
    if args.resume and resume_ckpt is not None and resume_ckpt.exists():
        state_peek = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        validate_resume_stage(state_peek, resume_dir, stage)
        saved_mode = state_peek.get("latent_query_mode")
        if stage == "query" and saved_mode is None and "mask_latent_query_labels" in state_peek:
            saved_mode = (
                "inject" if state_peek["mask_latent_query_labels"] else "generate"
            )
        if saved_mode is not None and saved_mode != args.latent_query_mode:
            raise ValueError(
                "checkpoint latent_query_mode mismatch: "
                f"checkpoint={saved_mode}, current={args.latent_query_mode}"
            )
        resume_lora = (
            bool(state_peek.get("lora"))
            or (resume_dir / "adapter_config.json").exists()
        )
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
        print(
            json.dumps(
                {"warning": "--resume set but no checkpoint found under output_dir"}
            )
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    prepare_query_vocabulary(
        model,
        len(processor.tokenizer),
        token_id_map,
        added_tokens=added,
        latent_token_count=args.latent_token_count,
    )

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
        print(
            json.dumps(
                {
                    "warning": "full fine-tuning all parameters; pass --lora for adapter-only SFT"
                }
            )
        )

    if query_config is not None:
        from nimloth.training.sft.stage2.model import QueryAlignmentModel

        model = QueryAlignmentModel.build(model, processor.tokenizer, query_config)
        if resume_dir is not None:
            model.restore_projector(resume_dir)
        elif (args.model / "grid_state_config.json").is_file():
            model.restore_projector(args.model)
    model.config.nimloth_training_stage = stage
    model.to(device)
    optimizer = build_optimizer(model, args.lr, args.embedding_lr, args.weight_decay)
    if getattr(args, "distributed_strategy", "ddp") == "fsdp":
        if world < 2 or device.type != "cuda":
            raise ValueError("FSDP requires multi-rank CUDA training")
        model = wrap_fsdp(model, device)
        optimizer = build_optimizer(model, args.lr, args.embedding_lr, args.weight_decay)
    elif world > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        if args.lora:
            # PEFT + gradient checkpointing requires static graph under DDP.
            model._set_static_graph()

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    convergence_policy = (
        ConvergencePolicy(args.convergence_min_epochs, args.convergence_patience_epochs,
                          args.convergence_min_relative_improvement)
        if args.until_converged else None
    )
    convergence = ConvergenceState()
    if convergence_policy is not None:
        scheduler = get_constant_schedule_with_warmup(
            optimizer, math.ceil(steps_per_epoch * args.warmup_ratio)
        )
    else:
        total_steps = steps_per_epoch * args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, int(total_steps * args.warmup_ratio), total_steps
        )
    resume_identity = _resume_identity(
        args, stage=stage, world=world, train_size=len(train_ds)
    )

    log_path = args.output_dir / "train_step_log.csv"
    if is_main() and not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "time",
                    "epoch",
                    "global_step",
                    "train_loss",
                    "val_loss",
                    "format_correct_rate",
                    "lr",
                ]
            )

    if is_main() and len(train_ds) > 0:
        if use_cache:
            probe = collate_cached_fn([train_ds[0]], pad_token_id)
        else:
            probe = collate_fn(
                [train_ds[0]],
                processor,
                args.max_length,
                latent_token_count=args.latent_token_count,
                mask_latent_query_labels=args.mask_latent_query_labels,
            )
        trained = probe["labels"][0][probe["labels"][0] != -100]
        decoded = processor.decode(trained.tolist(), skip_special_tokens=False)
        print(json.dumps({"mask_probe": decoded[:500]}))
        if "<|action_start|>" not in decoded:
            raise ValueError(
                "label mask probe missing <|action_start|> in supervised tokens"
            )

    global_step = 0
    best_val = float("inf")
    start_epoch = 1
    resume_next_micro_batch = 0
    resume_rank_rng: dict[str, Any] | None = None
    resume_at_epoch_boundary = False
    if args.resume and resume_ckpt is not None and resume_ckpt.exists():
        state = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        if not objective_identities_match(state.get("identity"), resume_identity):
            raise ValueError("weighted loss resume checkpoint objective identity mismatch")
        if convergence_policy is not None:
            if state.get("convergence_state") is None:
                raise ValueError("resume checkpoint lacks convergence state")
            convergence = ConvergenceState.from_state_dict(state["convergence_state"])
        global_step = int(state.get("step", 0))
        if args.max_optimizer_steps is not None and global_step >= args.max_optimizer_steps:
            raise ValueError(
                "resume step already reaches --max-optimizer-steps; raise or remove the cap"
            )
        best_val = float(state.get("best_val", float("inf")))
        if state.get("resume_schema") == RESUME_SCHEMA:
            validate_resume_state(
                state, expected_identity=resume_identity, rank=rank, world=world
            )
            start_epoch = int(state["epoch"])
            resume_next_micro_batch = int(state["next_micro_batch"])
            if resume_next_micro_batch > len(train_loader):
                raise ValueError(
                    "resume checkpoint data cursor exceeds current epoch length"
                )
            resume_rank_rng = state["rank_rng_states"][rank]
        elif "epoch" in state:
            if (
                state.get("identity") is not None
                and not objective_identities_match(state.get("identity"), resume_identity)
            ):
                raise ValueError(
                    "epoch checkpoint stage/dataset/objective identity mismatch"
                )
            if (
                state.get("world_size") is not None
                and int(state["world_size"]) != world
            ):
                raise ValueError(
                    f"epoch checkpoint world size mismatch: "
                    f"{state.get('world_size')} != {world}"
                )
            start_epoch = int(state["epoch"]) + 1
            if convergence_policy is not None:
                rng_states = state.get("rank_rng_states")
                if not isinstance(rng_states, list) or len(rng_states) != world:
                    raise ValueError("epoch checkpoint lacks per-rank RNG for faithful resume")
                resume_rank_rng = rng_states[rank]
                resume_at_epoch_boundary = True
        else:
            epoch_dirs = sorted(args.output_dir.glob("epoch_*"))
            start_epoch = (
                int(epoch_dirs[-1].name.split("_")[-1]) + 1 if epoch_dirs else 1
            )
        if convergence_policy is not None and convergence.last_epoch != start_epoch - 1:
            raise ValueError("convergence history does not match resume data cursor")
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
            load_optimizer_state(model, optimizer, state["optimizer"])
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
                        "next_micro_batch": resume_next_micro_batch,
                        "best_val": best_val,
                    }
                )
            )

    stop_after_boundary = False
    stop_requested = False

    def request_boundary_stop(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        if is_main():
            print(
                json.dumps(
                    {"signal": signum, "action": "stop_after_optimizer_boundary"}
                )
            )

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_boundary_stop)

    model.train()
    epoch_rng_states = state.get("rank_rng_states") if args.resume and resume_ckpt is not None else None
    epoch = start_epoch - 1
    epoch_numbers = (itertools.count(start_epoch) if args.until_converged
                     else range(start_epoch, args.epochs + 1))
    for epoch in epoch_numbers:
        if convergence.converged:
            epoch = convergence.last_epoch
            break
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        micro_accum = 0
        next_micro_batch = resume_next_micro_batch if epoch == start_epoch else 0

        def optimizer_step(
            *,
            micro_count: int,
            next_batch: int,
            epoch_number: int = epoch,
            best_at_epoch_start: float = best_val,
        ) -> None:
            nonlocal global_step, accum_loss, stop_after_boundary
            stop_after_boundary = stop_requested
            if world > 1:
                stop_request = torch.tensor(int(stop_after_boundary), device=device)
                dist.all_reduce(stop_request, op=dist.ReduceOp.MAX)
                stop_after_boundary = bool(stop_request.item())
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(micro_count)
            clip_grad_norm(model, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if args.max_optimizer_steps is not None and global_step >= args.max_optimizer_steps:
                stop_after_boundary = True
                if is_main():
                    print(json.dumps({"action": "pause_at_optimizer_step_cap",
                                      "global_step": global_step}))
            step_loss = accum_loss / micro_count
            if is_main():
                with log_path.open("a", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            time.time(),
                            epoch_number,
                            global_step,
                            step_loss,
                            "",
                            "",
                            scheduler.get_last_lr()[0],
                        ]
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

            if global_step % args.resume_save_steps == 0 or stop_after_boundary:
                save_resume_checkpoint(
                    model,
                    processor,
                    args.output_dir,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch=epoch_number,
                    next_micro_batch=next_batch,
                    best_val=best_at_epoch_start,
                    identity=resume_identity,
                    convergence_state=convergence.state_dict() if convergence_policy else None,
                    rank=rank,
                    world=world,
                    lora=args.lora,
                    base_model_path=base_model_path,
                    latent_token_count=args.latent_token_count,
                    mask_latent_query_labels=args.mask_latent_query_labels,
                    latent_query_mode=args.latent_query_mode,
                )

        if epoch == start_epoch and resume_rank_rng is not None and resume_at_epoch_boundary:
            restore_rng_state(resume_rank_rng)
            resume_rank_rng = None
        train_iterator = iter(train_loader)
        for _ in range(next_micro_batch):
            try:
                next(train_iterator)
            except StopIteration as error:
                raise ValueError(
                    "resume checkpoint data cursor is not reproducible"
                ) from error
        if epoch == start_epoch and resume_rank_rng is not None:
            restore_rng_state(resume_rank_rng)
            resume_rank_rng = None
        for batch_index, batch in enumerate(train_iterator, start=next_micro_batch):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = training_loss(model, batch, action_token_ids=action_ids,
                                 action_weight=args.action_token_loss_weight,
                                 boundary_token_ids=boundary_ids, boundary_weight=args.boundary_token_loss_weight)
            loss.backward()
            accum_loss += loss.detach().float().item()
            micro_accum += 1
            if micro_accum % args.grad_accum == 0:
                optimizer_step(micro_count=micro_accum, next_batch=batch_index + 1)
                micro_accum = 0
                if stop_after_boundary:
                    cleanup_dist()
                    return 75
        if micro_accum > 0:
            optimizer_step(micro_count=micro_accum, next_batch=len(train_loader))
            if stop_after_boundary:
                cleanup_dist()
                return 75

        if stop_after_boundary:
            save_resume_checkpoint(
                model,
                processor,
                args.output_dir,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                epoch=epoch,
                next_micro_batch=len(train_loader),
                best_val=best_val,
                identity=resume_identity,
                convergence_state=convergence.state_dict() if convergence_policy else None,
                rank=rank,
                world=world,
                lora=args.lora,
                base_model_path=base_model_path,
                latent_token_count=args.latent_token_count,
                mask_latent_query_labels=args.mask_latent_query_labels,
                latent_query_mode=args.latent_query_mode,
            )
            cleanup_dist()
            return 75

        resume_next_micro_batch = 0

        distributed_barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        validation_started_at = time.monotonic()
        if stage == "query":
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                args.max_val_batches,
                return_components=True,
                include_batches=val_include_batches,
            )
            val_loss = val_metrics["validation_total_loss"]
        else:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                args.max_val_batches,
                include_batches=val_include_batches,
                action_token_ids=action_ids, boundary_token_ids=boundary_ids,
                action_weight=args.action_token_loss_weight, boundary_weight=args.boundary_token_loss_weight,
            )
            val_loss = val_metrics["validation_lm_loss"]
            val_metrics.update(action_token_loss_weight=args.action_token_loss_weight, boundary_token_loss_weight=args.boundary_token_loss_weight)
        validation_seconds = time.monotonic() - validation_started_at
        if is_main():
            print(
                json.dumps(
                    {
                        "event": "validation_complete",
                        "epoch": epoch,
                        "global_step": global_step,
                        "validation_seconds": validation_seconds,
                    }
                ),
                flush=True,
            )
        format_eval_started_at = time.monotonic()
        format_rate, format_reasons, format_samples = evaluate_format(
            model,
            processor,
            format_eval_ds,
            device,
            args.format_eval_samples,
            batch_size=args.format_eval_batch_size,
            latent_token_count=args.latent_token_count,
            latent_query_mode=args.latent_query_mode,
        )
        format_eval_seconds = time.monotonic() - format_eval_started_at
        if convergence_policy is not None:
            convergence.observe(epoch=epoch, loss=val_loss, policy=convergence_policy)
        local_epoch_rng = capture_rng_state()
        epoch_rng_states = [local_epoch_rng]
        if world > 1:
            epoch_rng_states = [None] * world
            dist.all_gather_object(epoch_rng_states, local_epoch_rng)
        previous_best_val = best_val
        best_val = min(best_val, val_loss)
        if is_main():
            format_sample_path = _publish_format_samples(
                args.output_dir,
                epoch=epoch,
                global_step=global_step,
                source=(
                    args.format_eval_jsonl if stage == "format" else args.val_jsonl
                ),
                samples=format_samples,
            )
            with (args.output_dir / "validation_metrics.jsonl").open("a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "global_step": global_step,
                    "monitor": convergence_monitor(stage), **val_metrics,
                    "format_correct_rate": format_rate,
                    "format_failure_reasons": format_reasons,
                    "format_samples": str(format_sample_path),
                    "validation_seconds": validation_seconds,
                    "format_eval_seconds": format_eval_seconds,
                }) + "\n")
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        time.time(),
                        epoch,
                        global_step,
                        "",
                        val_loss,
                        format_rate,
                        scheduler.get_last_lr()[0],
                    ]
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
            latent_token_count=args.latent_token_count,
            mask_latent_query_labels=args.mask_latent_query_labels,
            latent_query_mode=args.latent_query_mode,
            world_size=world,
            identity=resume_identity,
            convergence_state=convergence.state_dict() if convergence_policy else None,
            rank_rng_states=epoch_rng_states,
        )
        if stage == "format" and args.prune_intermediate_checkpoints and is_main():
            from .checkpoint import prune_intermediate_checkpoints

            prune_intermediate_checkpoints(args.output_dir, epoch, global_step)
        if val_loss < previous_best_val:
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
                latent_token_count=args.latent_token_count,
                mask_latent_query_labels=args.mask_latent_query_labels,
                latent_query_mode=args.latent_query_mode,
                world_size=world,
                identity=resume_identity,
                convergence_state=convergence.state_dict() if convergence_policy else None,
                rank_rng_states=epoch_rng_states,
            )
        if is_main():
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_loss": val_loss,
                        **val_metrics,
                        "convergence": convergence.state_dict() if convergence_policy else None,
                        "format_correct_rate": format_rate,
                        "format_failure_reasons": format_reasons,
                        "format_eval_protocol": args.latent_query_mode,
                        "validation_seconds": validation_seconds,
                        "format_eval_seconds": format_eval_seconds,
                        "best_val": best_val,
                    }
                )
            )
            if wandb_run is not None:
                import wandb

                wandb.log(
                    {
                        "val/loss": val_loss,
                        **{f"val/{key}": value for key, value in val_metrics.items()},
                        "val/format_correct_rate": format_rate,
                        "val/validation_seconds": validation_seconds,
                        "val/format_eval_seconds": format_eval_seconds,
                        "val/best_loss": best_val,
                        "eval/val_loss": val_loss,
                        "eval/format_correct_rate": format_rate,
                        "epoch": epoch,
                    },
                    step=global_step,
                )
        distributed_barrier()
        if convergence.converged:
            break
        # 验证期间收到的暂停请求在完整 epoch checkpoint 发布后统一退出。
        epoch_stop = torch.tensor(int(stop_requested), device=device)
        if world > 1:
            dist.all_reduce(epoch_stop, op=dist.ReduceOp.MAX)
        if epoch_stop.item():
            cleanup_dist()
            return 75

    save_checkpoint(
        model,
        processor,
        args.output_dir,
        "final",
        optimizer,
        scheduler,
        global_step,
        epoch,
        best_val,
        lora=args.lora,
        base_model_path=base_model_path,
        merge_for_eval=False,
        latent_token_count=args.latent_token_count,
        mask_latent_query_labels=args.mask_latent_query_labels,
        latent_query_mode=args.latent_query_mode,
        world_size=world,
        identity=resume_identity,
        convergence_state=convergence.state_dict() if convergence_policy else None,
        rank_rng_states=epoch_rng_states,
    )
    if is_main():
        if convergence_policy is not None:
            (args.output_dir / "CONVERGED.json").write_text(
                json.dumps({"monitor": convergence_monitor(stage), "policy": convergence_policy.state_dict(),
                            "state": convergence.state_dict(), "global_step": global_step}) + "\n",
                encoding="utf-8",
            )
        if wandb_run is not None:
            import wandb

            wandb.finish()
    cleanup_dist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
