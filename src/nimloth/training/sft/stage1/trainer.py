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

from nimloth.latent import (
    add_special_tokens,
    initialize_extra_latent_token_embeddings,
    latent_state_block,
    latent_state_tokens,
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
from .continuation import (validate_epoch_continuation, restart_schedule, replay_convergence, continuation_provenance)
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
    prepare_embedding_masters,
    restore_exported_embedding_masters,
)
from .loss import resolve_action_token_ids, training_loss


def data_loader_kwargs(args, collator, *, use_cache: bool, query: bool) -> dict[str, Any]:
    """Parallelize CPU query preprocessing without forking initialized CUDA."""
    workers = args.num_workers if use_cache or query else 0
    kwargs: dict[str, Any] = {
        "num_workers": workers, "pin_memory": True, "collate_fn": collator,
    }
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
        if query:
            kwargs.update(multiprocessing_context="spawn", worker_init_fn=_init_query_worker)
    return kwargs


def _init_query_worker(_worker_id: int) -> None:
    # DataLoader sets the worker Torch/Python/NumPy seeds and Torch threads=1.
    # Tokenizer threads would otherwise multiply across ranks and workers.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _nimloth_format_re(latent_token_count: int | None = None) -> re.Pattern[str]:
    latent_block = r"\s*".join(
        re.escape(token) for token in (latent_state_tokens(latent_token_count) if latent_token_count is not None else ())
    )
    return re.compile(
        r"<think>.*?</think>\s*"
        + latent_block
        + r"\s*<\|action_start\|>\s*<\|action_\(\d+\)\|\>\s*<\|action_end\|>",
        re.DOTALL,
    )


def nimloth_format_correct(text: str, *, latent_token_count: int | None = None) -> bool:
    return bool(_nimloth_format_re(latent_token_count).search(text))


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
    latent_token_count: int | None = None,
    latent_query_mode: str | None = None,
) -> float:
    if dist.is_available() and dist.is_initialized() and not is_main() and not is_fsdp(model):
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
        inputs = processor(text=[text], images=images or None, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with generation_model(model) as generation_module:
            output_ids = generation_module.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                **({"synced_gpus": True} if is_fsdp(model) else {}),
            )
        new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        decoded = processor.decode(new_ids, skip_special_tokens=False)
        total += 1
        if latent_query_mode == "inject":
            correct += int(action_block_format_correct(decoded))
        else:
            correct += int(
                nimloth_format_correct(decoded, latent_token_count=latent_token_count)
            )
    if was_training:
        module.train()
    return correct / max(total, 1)


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
    model, loader, device: torch.device, max_batches: int = -1,
    *, return_components: bool = False, weight_lm: float = 1.0, weight_dino: float = 1.0,
) -> float | dict[str, float]:
    model.eval()
    total = torch.zeros(2 if return_components else 1, device=device)
    count = torch.zeros_like(total)
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        output = model(**batch)
        if return_components:
            total += torch.stack([output.lm_loss_sum.detach(), output.dino_loss_sum.detach()])
            count += torch.stack([output.lm_answer_count, output.answer_count])
        else:
            total += output.loss.detach()
            count += 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    model.train()
    if count[-1].item() == 0:
        raise ValueError("validation loader produced no monitored loss")
    means = total / count.clamp_min(1)
    if return_components:
        lm, dino = means.tolist()
        return {"validation_total_loss": weight_lm * lm + weight_dino * dino,
                "validation_lm_loss": lm, "validation_dino_loss": dino}
    return means.item()


def convergence_monitor(stage: str) -> str:
    return "validation_total_loss" if stage == "query" else "validation_lm_loss"


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    embedding_lr: float | None,
    weight_decay: float,
    projector_lr: float | None = None,
) -> torch.optim.AdamW:
    embed_lr = embedding_lr if embedding_lr is not None else lr
    embed_keys = ("embed_tokens", "lm_head")
    embed_params: list[torch.nn.Parameter] = []
    base_params: list[torch.nn.Parameter] = []
    projector_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if projector_lr is not None and "projector" in name.split("."):
            projector_params.append(param)
        elif any(key in name for key in embed_keys):
            embed_params.append(param)
        else:
            base_params.append(param)
    groups = [{"params": base_params, "lr": lr},
              {"params": embed_params, "lr": embed_lr}]
    if projector_lr is not None:
        if not projector_params:
            raise ValueError("projector_lr requires trainable projector parameters")
        groups.append({"params": projector_params, "lr": projector_lr})
    return torch.optim.AdamW(
        groups,
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resume_identity(
    args: argparse.Namespace, *, stage: str, world: int, train_size: int
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "stage": stage,
        "format_objective": FORMAT_OBJECTIVE if stage == "format" else None,
        "action_token_loss_weight": getattr(args, "action_token_loss_weight", 1.0),
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
    if getattr(args, "projector_lr", None) is not None:
        identity["projector_lr"] = args.projector_lr
    if getattr(args, "embedding_master_dtype", "bfloat16") != "bfloat16":
        identity["embedding_master_dtype"] = args.embedding_master_dtype
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
                "query_batching": "full_trajectory_success_lm_all_dino_v2",
            }
        )
    return identity


def main(*, stage: str = "format") -> int:
    args, query_config = parse_args(stage=stage)
    if query_config is not None and args.action_token_loss_weight != 1:
        raise ValueError("query alignment uses answer-equal LM loss and requires action weight 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rank, world, local_rank, device = setup_dist()
    continuing = getattr(args, "continue_from_epoch", None) is not None
    output_exists = [args.output_dir.exists() if rank == 0 else None]
    if continuing and world > 1:
        dist.broadcast_object_list(output_exists, src=0)
    if continuing and output_exists[0]:
        raise FileExistsError("epoch continuation requires a new output directory")
    if continuing and stage != "query":
        raise ValueError("epoch continuation is supported only for Stage2")
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
    action_ids = resolve_action_token_ids(processor.tokenizer) if stage == "format" else ()
    if is_main():
        print(json.dumps({"action_token_loss_weight": args.action_token_loss_weight, "weighted_action_token_ids": action_ids}))
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
                }
            )
        )

    wandb_run = None
    if not args.cache_only:
        wandb_run = maybe_init_wandb(args)
        if wandb_run is not None:
            upload_dataset_artifact(wandb_run, args.train_jsonl, args.val_jsonl)

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
    format_eval_ds = val_ds
    if query_config is not None:
        from nimloth.backbone.dino_grid import (
            DINOV2_LARGE_IDENTITY,
            CachedDINOGridTargets,
        )
        from nimloth.training.sft.stage2.data import QueryAlignmentCollator

        targets = CachedDINOGridTargets.from_cache_root(
            args.dino_cache_root,
            identity=DINOV2_LARGE_IDENTITY,
            grid_size=query_config.grid_size,
        )
        args.dino_cache_fingerprint = targets.cache_fingerprint
        train_collate = QueryAlignmentCollator(
            processor,
            args.max_length,
            query_config.grid_tokens,
            targets,
            mask_latent_query_labels=args.mask_latent_query_labels,
        )
    loader_kwargs = data_loader_kwargs(
        args, train_collate, use_cache=use_cache, query=query_config is not None
    )

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
        if world > 1
        else None
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
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        **loader_kwargs,
    )

    base_model_path = args.model
    resume_dir: Path | None = (
        args.continue_from_epoch if continuing else (find_latest_resume_dir(args.output_dir) if args.resume else None)
    )
    resume_ckpt = resume_dir / "training_state.pt" if resume_dir is not None else None
    if continuing and not resume_ckpt.is_file():
        raise FileNotFoundError("continuation checkpoint training_state.pt is missing")
    load_path = args.model
    resume_lora = False
    if (args.resume or continuing) and resume_ckpt is not None and resume_ckpt.exists():
        state_peek = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        validate_resume_stage(state_peek, resume_dir, stage)
        if continuing:
            validate_epoch_continuation(resume_dir, state_peek, _resume_identity(args, stage=stage, world=world, train_size=len(train_ds)), world=world)
            if not args.until_converged and args.epochs <= int(state_peek["epoch"]):
                raise ValueError("--epochs must exceed the completed source epoch")
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
    if getattr(args, "embedding_master_dtype", "bfloat16") == "float32":
        restore_exported_embedding_masters(model, load_path)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    prepare_query_vocabulary(
        model,
        len(processor.tokenizer),
        token_id_map,
        added_tokens=added,
        latent_token_count=args.latent_token_count,
    )

    if (args.resume or continuing) and resume_ckpt is not None and resume_ckpt.exists() and resume_lora:
        if not args.lora:
            raise ValueError("--resume with LoRA adapter requires --lora")
        model = apply_lora(model, args)
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
    # Build the projector in the original BF16 dtype before promoting PEFT copies.
    language_model = model.language_model if query_config is not None else model
    prepare_embedding_masters(language_model, getattr(args, "embedding_master_dtype", "bfloat16"))
    if (args.resume or continuing) and resume_ckpt is not None and resume_ckpt.exists() and resume_lora:
        load_lora_adapter_state(language_model, resume_dir)
    model.config.nimloth_training_stage = stage
    model.to(device)
    optimizer = build_optimizer(model, args.lr, args.embedding_lr, args.weight_decay, args.projector_lr)
    if getattr(args, "distributed_strategy", "ddp") == "fsdp":
        if world < 2 or device.type != "cuda":
            raise ValueError("FSDP requires multi-rank CUDA training")
        model = wrap_fsdp(model, device)
        optimizer = build_optimizer(model, args.lr, args.embedding_lr, args.weight_decay, args.projector_lr)
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
    configured_learning_rates = [group["lr"] for group in optimizer.param_groups]
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
    if (args.resume or continuing) and resume_ckpt is not None and resume_ckpt.exists():
        state = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        if args.action_token_loss_weight != 1 and not objective_identities_match(state.get("identity"), resume_identity):
            raise ValueError("weighted loss resume checkpoint objective identity mismatch")
        if continuing:
            validate_epoch_continuation(resume_dir, state, resume_identity, world=world)
        if convergence_policy is not None and continuing:
            convergence, history = replay_convergence(resume_dir.parent / "validation_metrics.jsonl", int(state["epoch"]), convergence_policy, convergence_monitor(stage))
            if is_main():
                (args.output_dir / "validation_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
        elif continuing:
            source_history = resume_dir.parent / "validation_metrics.jsonl"
            history = [json.loads(line) for line in source_history.read_text().splitlines() if line.strip()]
            history = [row for row in history if row["epoch"] <= int(state["epoch"])]
            if [row["epoch"] for row in history] != list(range(1, int(state["epoch"]) + 1)):
                raise ValueError("continuation validation history is incomplete or duplicated")
            if is_main():
                (args.output_dir / "validation_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
        elif convergence_policy is not None:
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
                not continuing and state.get("identity") is not None
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
            if convergence_policy is not None or continuing:
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
        if continuing:
            scheduler = restart_schedule(optimizer, configured_learning_rates, steps_per_epoch=steps_per_epoch, remaining_epochs=(args.epochs - int(state["epoch"])) if args.epochs is not None else 0, warmup_ratio=args.warmup_ratio, until_converged=args.until_converged)
            if is_main():
                (args.output_dir / "continuation.json").write_text(json.dumps(continuation_provenance(resume_dir, resume_identity), indent=2) + "\n")
        elif state.get("scheduler") is not None:
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

    if getattr(args, "save_initial_checkpoint", False) and resume_dir is None:
        if (args.output_dir / "epoch_000").exists():
            raise FileExistsError("refusing to overwrite initial checkpoint")
        initial_rng = [capture_rng_state()]
        if world > 1:
            local_rng = initial_rng[0]
            initial_rng = [None] * world
            dist.all_gather_object(initial_rng, local_rng)
        save_checkpoint(
            model, processor, args.output_dir, "epoch_000", optimizer, scheduler,
            step=0, epoch=0, best_val=best_val, lora=args.lora,
            base_model_path=base_model_path, merge_for_eval=False,
            latent_token_count=args.latent_token_count,
            mask_latent_query_labels=args.mask_latent_query_labels,
            latent_query_mode=args.latent_query_mode, world_size=world,
            identity=resume_identity,
            convergence_state=convergence.state_dict() if convergence_policy else None,
            rank_rng_states=initial_rng,
        )
        if world > 1:
            dist.barrier()

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
    epoch_rng_states = state.get("rank_rng_states") if (args.resume or continuing) and resume_ckpt is not None else None
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
            if stage == "query":
                global_loss = torch.tensor(accum_loss, device=device)
                if world > 1:
                    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
                gradient_scale = 1.0
                step_loss = global_loss.item() / world
            else:
                gradient_scale = 1 / micro_count
                step_loss = accum_loss / micro_count
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(gradient_scale)
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
        def normalized_batches(iterator=train_iterator):
            if stage != "query":
                for item in iterator:
                    yield item, None
                return
            while group := list(itertools.islice(iterator, args.grad_accum)):
                # 先统计整个更新组，再逐个前向；不保留多个 Qwen 计算图。
                counts = torch.tensor([
                    sum(int(item["lm_answer_mask"].sum()) for item in group),
                    sum(item["query_positions"].shape[0] for item in group),
                ], device=device)
                if world > 1:
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                for item in group:
                    yield item, counts

        for batch_index, (batch, group_counts) in enumerate(normalized_batches(), start=next_micro_batch):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if stage == "query":
                output = model(**batch)
                loss = (args.weight_lm * output.lm_loss_sum * world / group_counts[0].clamp_min(1)
                        + args.weight_dino * output.dino_loss_sum * world / group_counts[1])
            else:
                loss = training_loss(
                    model,
                    batch,
                    action_token_ids=action_ids,
                    action_weight=args.action_token_loss_weight,
                )
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
        if stage == "query":
            val_metrics = evaluate(
                model, val_loader, device, args.max_val_batches, return_components=True,
                weight_lm=args.weight_lm, weight_dino=args.weight_dino,
            )
            val_loss = val_metrics["validation_total_loss"]
        else:
            val_loss = evaluate(model, val_loader, device, args.max_val_batches)
            val_metrics = {"validation_lm_loss": val_loss}
        format_rate = evaluate_format(
            model,
            processor,
            format_eval_ds,
            device,
            args.format_eval_samples,
            latent_token_count=args.latent_token_count,
            latent_query_mode=args.latent_query_mode,
        )
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
            with (args.output_dir / "validation_metrics.jsonl").open("a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "global_step": global_step,
                    "monitor": convergence_monitor(stage), **val_metrics,
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
                        "format_eval_protocol": args.latent_query_mode,
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
