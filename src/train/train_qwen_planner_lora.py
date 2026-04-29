"""LoRA SFT for Qwen planner special-token output."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
from typing import Any

from PIL import Image
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
import torch
from torch.utils.data import DataLoader, Dataset

from src.visualize.wandb_tracker import init_tracker
from src.train.validate_qwen_planner_lora import _validate_record_schema, _validate_records
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.qwen_planner import load_jsonl

try:
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs
except Exception:  # pragma: no cover
    Accelerator = None
    InitProcessGroupKwargs = None

console = Console()


class PlannerSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, limit: int = 0) -> None:
        records = load_jsonl(jsonl_path, limit=limit)
        for idx, item in enumerate(records):
            _validate_record_schema(item, idx)
            if not str(item.get("response", "")).strip():
                raise ValueError(f"{item.get('id', idx)} missing required response")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


def _build_collate_fn(processor: Any):
    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [Image.open(str(item["image"])).convert("RGB") for item in batch]
        prompt_texts = []
        full_texts = []
        for item in batch:
            prompt = str(item["prompt"])
            response = str(item["response"])
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            full_messages = user_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            ]
            prompt_texts.append(
                processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
            )
            full_texts.append(
                processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            )

        inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        labels = inputs["input_ids"].clone()
        pad_id = getattr(processor.tokenizer, "pad_token_id", None)
        for row_idx, (prompt_text, image) in enumerate(zip(prompt_texts, images)):
            prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
            prompt_len = int(prompt_inputs["input_ids"].shape[1])
            labels[row_idx, :prompt_len] = -100
        if pad_id is not None:
            labels[inputs["input_ids"] == pad_id] = -100
        inputs["labels"] = labels
        return inputs

    return _collate


def _move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _build_train_table(
    *,
    epoch: int,
    epochs: int,
    global_step: int,
    loss: float,
    lr: float,
    trainable_params: int,
    samples: int,
    last_validation: dict[str, float] | None = None,
) -> Table:
    table = Table(title="Qwen Planner LoRA SFT", expand=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("epoch", f"{epoch}/{epochs}")
    table.add_row("global_step", str(global_step))
    table.add_row("loss", f"{loss:.6f}")
    table.add_row("lr", f"{lr:.8f}")
    table.add_row("samples", str(samples))
    table.add_row("trainable_params", f"{trainable_params:,}")
    if last_validation:
        table.add_row("val_format_rate", f"{float(last_validation.get('format_rate', 0.0)):.4f}")
        table.add_row("val_action_top1", f"{float(last_validation.get('action_prior_top1', 0.0)):.4f}")
        table.add_row("val_action_top3", f"{float(last_validation.get('action_prior_top3', 0.0)):.4f}")
        latent_rate = last_validation.get("latent_extract_rate")
        if latent_rate is not None:
            table.add_row("val_latent_rate", f"{float(latent_rate):.4f}")
    return table


def _load_validation_records(path: str, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    records = load_jsonl(path, limit=limit)
    for idx, item in enumerate(records):
        _validate_record_schema(item, idx)
    return records


def _run_inline_validation(
    *,
    model,
    processor,
    records: list[dict[str, Any]],
    max_new_tokens: int,
    extract_latent: bool,
) -> dict[str, Any]:
    was_training = bool(model.training)
    model.eval()
    try:
        return _validate_records(
            records=records,
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
            extract_latent=extract_latent,
            show_progress=False,
        )
    finally:
        if was_training:
            model.train()


def _resolve_resume_checkpoint(path: str, checkpoint_dir: Path) -> Path | None:
    raw = str(path or "").strip()
    if not raw:
        return None
    if raw != "latest":
        ckpt = Path(raw)
        return ckpt if ckpt.exists() else None
    candidates = sorted(checkpoint_dir.glob("step_*"))
    return candidates[-1] if candidates else None


def _write_trainer_state(path: Path, *, global_step: int, epoch: int, batch_idx: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "trainer_state.json", "w") as f:
        json.dump(
            {"global_step": int(global_step), "epoch": int(epoch), "batch_idx": int(batch_idx)},
            f,
            indent=2,
        )


def _read_trainer_state(path: Path) -> dict[str, int]:
    state_path = path / "trainer_state.json"
    if not state_path.exists():
        return {"global_step": 0, "epoch": 0, "batch_idx": -1}
    with open(state_path) as f:
        data = json.load(f)
    return {
        "global_step": int(data.get("global_step", 0)),
        "epoch": int(data.get("epoch", 0)),
        "batch_idx": int(data.get("batch_idx", -1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-jsonl", default="datasets/EB-Nav/phase2_qwen_planner_sft.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output-dir", default="models/qwen_planner_lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--save-every-steps", type=int, default=500)
    parser.add_argument("--resume-from-checkpoint", default="", help="Checkpoint dir or 'latest'.")
    parser.add_argument("--validate-every-steps", type=int, default=0, help="Run planner validation every N optimizer steps. 0 disables inline validation.")
    parser.add_argument("--validation-jsonl", default="", help="Validation JSONL. Defaults to --sft-jsonl.")
    parser.add_argument("--validation-limit", type=int, default=32, help="Number of validation samples for each inline validation run.")
    parser.add_argument("--validation-max-new-tokens", type=int, default=512)
    parser.add_argument("--validation-extract-latent", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ddp-backend",
        choices=["auto", "nccl", "gloo"],
        default="auto",
        help="Distributed backend for Accelerate. Use gloo if NCCL topology init fails.",
    )
    parser.add_argument("--ddp-timeout-minutes", type=int, default=60)
    parser.add_argument("--no-accelerate", action="store_true", help="Disable Accelerate and run the legacy single-process path.")
    parser.add_argument("--no-progress", action="store_true", help="Disable Rich TUI/progress output.")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable W&B logging for this script.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="*",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    args = parser.parse_args()

    use_accelerate = not bool(args.no_accelerate)
    accelerator = None
    if use_accelerate:
        if Accelerator is None:
            raise RuntimeError("未安装 accelerate，无法进行多卡/Accelerate 训练。请先运行 uv sync。")
        kwargs_handlers = []
        if str(args.ddp_backend) != "auto":
            if InitProcessGroupKwargs is None:
                raise RuntimeError("当前 accelerate 版本不支持 InitProcessGroupKwargs。")
            kwargs_handlers.append(
                InitProcessGroupKwargs(
                    backend=str(args.ddp_backend),
                    timeout=timedelta(minutes=max(1, int(args.ddp_timeout_minutes))),
                )
            )
        accelerator = Accelerator(
            gradient_accumulation_steps=max(1, int(args.gradient_accumulation_steps)),
            kwargs_handlers=kwargs_handlers,
        )
    is_main_process = accelerator is None or accelerator.is_main_process

    adapter = QwenVLMAdapter(
        model_name=args.model_name,
        latent_dim=4096,
        enabled=True,
        fallback_enabled=False,
        device_map=None if use_accelerate else "auto",
    )
    adapter._ensure_model()
    if adapter._model is None or adapter._processor is None:
        raise RuntimeError(f"Failed to load Qwen model: {adapter.init_error}")
    trainable = adapter.enable_language_lora(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=list(args.target_modules),
    )
    model = adapter._model
    processor = adapter._processor
    if bool(args.gradient_checkpointing):
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    model.train()
    if is_main_process:
        console.print(f"language_lora_trainable_params={trainable}")

    dataset = PlannerSFTDataset(args.sft_jsonl, limit=args.limit)
    if len(dataset) == 0:
        raise RuntimeError(f"No valid SFT records found in {args.sft_jsonl}")
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=0,
        collate_fn=_build_collate_fn(processor),
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr))
    tracker = None
    if is_main_process and not args.disable_wandb:
        tracker = init_tracker(
            task_name="train_qwen_planner_lora",
            config={
                "sft_jsonl": args.sft_jsonl,
                "model_name": args.model_name,
                "output_dir": args.output_dir,
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "limit": int(args.limit),
                "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
                "lora_r": int(args.lora_r),
                "lora_alpha": int(args.lora_alpha),
                "lora_dropout": float(args.lora_dropout),
                "target_modules": list(args.target_modules),
                "trainable_params": int(trainable),
                "num_samples": len(dataset),
                "accelerate_enabled": use_accelerate,
                "num_processes": int(accelerator.num_processes if accelerator is not None else 1),
                "ddp_backend": str(args.ddp_backend),
                "gradient_checkpointing": bool(args.gradient_checkpointing),
                "validate_every_steps": int(args.validate_every_steps),
                "validation_jsonl": str(args.validation_jsonl or args.sft_jsonl),
                "validation_limit": int(args.validation_limit),
                "validation_max_new_tokens": int(args.validation_max_new_tokens),
                "validation_extract_latent": bool(args.validation_extract_latent),
            },
        )

    if accelerator is not None:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        device = accelerator.device
    else:
        device = next(model.parameters()).device
    accum = max(1, int(args.gradient_accumulation_steps))
    epochs = max(1, int(args.epochs))
    total_steps = epochs * len(dataloader)
    log_every = max(1, int(args.log_every))
    save_every_steps = max(0, int(args.save_every_steps))
    validate_every_steps = max(0, int(args.validate_every_steps))
    validation_records: list[dict[str, Any]] = []
    if is_main_process and validate_every_steps > 0:
        validation_records = _load_validation_records(
            str(args.validation_jsonl or args.sft_jsonl),
            limit=max(1, int(args.validation_limit)),
        )
        if not validation_records:
            raise RuntimeError(f"No valid validation records found in {args.validation_jsonl or args.sft_jsonl}")
        console.print(f"inline_validation_samples={len(validation_records)}")
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(args.output_dir) / "checkpoints"
    global_step = 0
    resume_checkpoint = _resolve_resume_checkpoint(str(args.resume_from_checkpoint), checkpoint_dir)
    resume_epoch = 0
    resume_batch_idx = -1
    if resume_checkpoint is not None:
        if accelerator is not None:
            accelerator.load_state(str(resume_checkpoint))
        else:
            state = torch.load(resume_checkpoint / "training_state.pt", map_location=device)
            model.load_state_dict(state["model_state_dict"], strict=False)
            optimizer.load_state_dict(state["optimizer_state_dict"])
        trainer_state = _read_trainer_state(resume_checkpoint)
        global_step = int(trainer_state["global_step"])
        resume_epoch = int(trainer_state["epoch"])
        resume_batch_idx = int(trainer_state["batch_idx"])
        if is_main_process:
            console.print(f"Resumed from checkpoint: {resume_checkpoint} at global_step={global_step}")
    optimizer.zero_grad(set_to_none=True)
    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    task_id = progress.add_task("Training Qwen planner LoRA", total=total_steps)
    loss_value = 0.0
    last_validation: dict[str, Any] | None = None
    train_table = _build_train_table(
        epoch=0,
        epochs=epochs,
        global_step=0,
        loss=0.0,
        lr=float(args.lr),
        trainable_params=trainable,
        samples=len(dataset),
        last_validation=last_validation,
    )
    live_context = (
        Live(Group(train_table, progress), console=console, refresh_per_second=4)
        if is_main_process and not args.no_progress
        else None
    )
    if live_context is not None:
        live_context.__enter__()
    try:
        for epoch in range(epochs):
            if epoch < resume_epoch:
                continue
            for batch_idx, batch in enumerate(dataloader):
                if epoch == resume_epoch and batch_idx <= resume_batch_idx:
                    if is_main_process:
                        progress.update(task_id, advance=1)
                    continue
                if accelerator is None:
                    batch = _move_to_device(batch, device)
                    outputs = model(**batch)
                    loss = outputs.loss / float(accum)
                    loss.backward()
                    loss_value = float(loss.item() * accum)
                    if (batch_idx + 1) % accum == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    should_update_progress = True
                else:
                    with accelerator.accumulate(model):
                        outputs = model(**batch)
                        loss = outputs.loss
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    gathered_loss = accelerator.gather(loss.detach()).mean()
                    loss_value = float(gathered_loss.item())
                    should_update_progress = accelerator.is_main_process
                if tracker is not None and global_step % log_every == 0:
                    tracker.log_metrics(
                        {
                            "train/loss": loss_value,
                            "train/epoch": epoch + 1,
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "train/global_step": global_step,
                        },
                        step=global_step,
                    )
                if is_main_process and global_step % log_every == 0:
                    console.print(f"epoch={epoch + 1} step={global_step} loss={loss_value:.6f}")
                global_step += 1
                if should_update_progress:
                    progress.update(task_id, advance=1)
                if live_context is not None:
                    train_table = _build_train_table(
                        epoch=epoch + 1,
                        epochs=epochs,
                        global_step=global_step,
                        loss=loss_value,
                        lr=float(optimizer.param_groups[0]["lr"]),
                        trainable_params=trainable,
                        samples=len(dataset),
                        last_validation=last_validation,
                    )
                    live_context.update(Group(train_table, progress))
                if (
                    validate_every_steps > 0
                    and global_step > 0
                    and global_step % validate_every_steps == 0
                ):
                    if accelerator is not None:
                        accelerator.wait_for_everyone()
                    if is_main_process:
                        model_for_validation = accelerator.unwrap_model(model) if accelerator is not None else model
                        if live_context is not None:
                            live_context.update(Group(train_table, progress))
                        last_validation = _run_inline_validation(
                            model=model_for_validation,
                            processor=processor,
                            records=validation_records,
                            max_new_tokens=int(args.validation_max_new_tokens),
                            extract_latent=bool(args.validation_extract_latent),
                        )
                        console.print(
                            "validation "
                            f"step={global_step} "
                            f"format_rate={float(last_validation.get('format_rate', 0.0)):.4f} "
                            f"top1={float(last_validation.get('action_prior_top1', 0.0)):.4f} "
                            f"top3={float(last_validation.get('action_prior_top3', 0.0)):.4f}"
                        )
                        if tracker is not None:
                            tracker.log_metrics(
                                {
                                    f"validation/{key}": value
                                    for key, value in last_validation.items()
                                    if isinstance(value, (int, float)) and value is not None
                                },
                                step=global_step,
                            )
                        train_table = _build_train_table(
                            epoch=epoch + 1,
                            epochs=epochs,
                            global_step=global_step,
                            loss=loss_value,
                            lr=float(optimizer.param_groups[0]["lr"]),
                            trainable_params=trainable,
                            samples=len(dataset),
                            last_validation=last_validation,
                        )
                        if live_context is not None:
                            live_context.update(Group(train_table, progress))
                    if accelerator is not None:
                        accelerator.wait_for_everyone()
                if (
                    save_every_steps > 0
                    and global_step > 0
                    and global_step % save_every_steps == 0
                ):
                    ckpt_path = checkpoint_dir / f"step_{global_step:08d}"
                    if accelerator is not None:
                        accelerator.wait_for_everyone()
                        accelerator.save_state(str(ckpt_path))
                    elif is_main_process:
                        ckpt_path.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                            },
                            ckpt_path / "training_state.pt",
                        )
                    if is_main_process:
                        _write_trainer_state(
                            ckpt_path,
                            global_step=global_step,
                            epoch=epoch,
                            batch_idx=batch_idx,
                        )
                        console.print(f"checkpoint_saved={ckpt_path}")
    finally:
        if live_context is not None:
            live_context.__exit__(None, None, None)
    if accelerator is None:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    else:
        accelerator.wait_for_everyone()

    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = accelerator.unwrap_model(model) if accelerator is not None else model
        model_to_save.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        final_ckpt = checkpoint_dir / "final"
        final_ckpt.mkdir(parents=True, exist_ok=True)
        _write_trainer_state(final_ckpt, global_step=global_step, epoch=epochs - 1, batch_idx=len(dataloader) - 1)
    if tracker is not None:
        tracker.log_metrics({"train/final_loss": loss_value, "train/total_steps": global_step}, step=global_step)
        tracker.log_artifact_path("qwen-planner-lora", output_dir, artifact_type="model")
        tracker.finish()
    if is_main_process:
        console.print(f"saved_lora_adapter={output_dir}")


if __name__ == "__main__":
    main()
