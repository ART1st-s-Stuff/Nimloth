"""Shared early-stage lifecycle: format CE or current-observation query alignment.

Data/masking live in data.py; the query objective lives in stage2/model.py.
This loop owns model tuning, accumulation, validation and epoch checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
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
    get_cosine_schedule_with_warmup,
)

from nimloth.latent import (
    add_special_tokens,
    initialize_extra_latent_token_embeddings,
    latent_state_block,
    latent_state_tokens,
    normalize_latent_state_blocks,
    special_token_ids,
)

from .checkpoint import (
    find_latest_resume_dir,
    load_lora_adapter_state,
    save_checkpoint,
    validate_resume_stage,
)
from .cli import parse_args
from .data import (
    NimlothVLSFTDataset,
    build_preprocess_cache,
    cache_fingerprint,
    collate_cached_fn,
    collate_fn,
    collect_images,
)
from .distributed import cleanup_dist, distributed_barrier, is_main, setup_dist


def _nimloth_format_re(latent_token_count: int = 1) -> re.Pattern[str]:
    latent_block = r"\s*".join(
        re.escape(token) for token in latent_state_tokens(latent_token_count)
    )
    return re.compile(
        r"<think>.*?</think>\s*"
        + latent_block
        + r"\s*<\|action_start\|>\s*<\|action_\(\d+\)\|\>\s*<\|action_end\|>",
        re.DOTALL,
    )


def nimloth_format_correct(text: str, *, latent_token_count: int = 1) -> bool:
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
    latent_token_count: int = 1,
    latent_query_mode: str = "inject",
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
        text = normalize_latent_state_blocks(text, latent_token_count)
        inputs = processor(text=[text], images=images or None, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output_ids = module.generate(**inputs, max_new_tokens=128, do_sample=False)
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
    latent_token_count: int,
) -> None:
    """Extend a base model without resetting trained query rows on full resume."""
    vocabulary_grows = model.get_input_embeddings().weight.shape[0] < vocabulary_size
    resize_token_embeddings_and_sync_vocab(model, vocabulary_size)
    if added_tokens > 0 and vocabulary_grows:
        initialize_extra_latent_token_embeddings(
            model,
            token_id_map,
            latent_token_count=latent_token_count,
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
    if is_main():
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
        from nimloth.training.sft.stage2.data import QueryAlignmentCollator

        targets = CachedDINOGridTargets.from_cache_root(
            args.dino_cache_root,
            identity=DINOV2_LARGE_IDENTITY,
            grid_size=query_config.grid_size,
        )
        train_collate = QueryAlignmentCollator(
            processor,
            args.max_length,
            query_config.grid_tokens,
            targets,
            mask_latent_query_labels=args.mask_latent_query_labels,
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

    train_sampler = (
        DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed
        )
        if world > 1
        else None
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
    resume_dir: Path | None = (
        find_latest_resume_dir(args.output_dir) if args.resume else None
    )
    resume_ckpt = resume_dir / "training_state.pt" if resume_dir is not None else None
    load_path = args.model
    resume_lora = False
    if args.resume and resume_ckpt is not None and resume_ckpt.exists():
        state_peek = torch.load(resume_ckpt, map_location="cpu")
        validate_resume_stage(state_peek, resume_dir, stage)
        saved_mode = state_peek.get("latent_query_mode")
        if saved_mode is None and "mask_latent_query_labels" in state_peek:
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
        model.gradient_checkpointing_enable()
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
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
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
    if args.resume and resume_ckpt is not None and resume_ckpt.exists():
        state = torch.load(resume_ckpt, map_location="cpu")
        global_step = int(state.get("step", 0))
        best_val = float(state.get("best_val", float("inf")))
        if "epoch" in state:
            start_epoch = int(state["epoch"]) + 1
        else:
            epoch_dirs = sorted(args.output_dir.glob("epoch_*"))
            start_epoch = (
                int(epoch_dirs[-1].name.split("_")[-1]) + 1 if epoch_dirs else 1
            )
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

        def optimizer_step(*, micro_count: int, epoch_number: int = epoch) -> None:
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
        format_rate = evaluate_format(
            model,
            processor,
            val_ds,
            device,
            args.format_eval_samples,
            latent_token_count=args.latent_token_count,
            latent_query_mode=args.latent_query_mode,
        )
        if is_main():
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
                    latent_token_count=args.latent_token_count,
                    mask_latent_query_labels=args.mask_latent_query_labels,
                    latent_query_mode=args.latent_query_mode,
                )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_loss": val_loss,
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
                        "val/format_correct_rate": format_rate,
                        "val/best_loss": best_val,
                        "eval/val_loss": val_loss,
                        "eval/format_correct_rate": format_rate,
                        "epoch": epoch,
                    },
                    step=global_step,
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
            latent_token_count=args.latent_token_count,
            mask_latent_query_labels=args.mask_latent_query_labels,
            latent_query_mode=args.latent_query_mode,
        )
        if wandb_run is not None:
            import wandb

            wandb.finish()
    cleanup_dist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
