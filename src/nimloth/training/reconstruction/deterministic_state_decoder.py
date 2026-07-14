"""Train a deterministic spatial image decoder from cached SFT2 states."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from nimloth.training.reconstruction.cfm_sft2 import (
    LoadedStateImageSplit,
    load_state_image_split,
)
from nimloth.wm.reconstruction import WMImageDecoder, WMImageDecoderConfig


def reconstruction_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l1 = torch.nn.functional.l1_loss(prediction, target)
    mse = torch.nn.functional.mse_loss(prediction, target)
    return l1 + 0.5 * mse, l1, mse


@torch.no_grad()
def evaluate_condition_sensitivity(
    decoder: WMImageDecoder,
    split: LoadedStateImageSplit,
    device: torch.device,
    *,
    batch_size: int,
    max_items: int,
) -> dict[str, float]:
    decoder.eval()
    count = split.states.shape[0] if max_items < 0 else min(max_items, split.states.shape[0])
    correct_loss = 0.0
    wrong_loss = 0.0
    l1_sum = 0.0
    mse_sum = 0.0
    total = 0
    for start in range(0, count, batch_size):
        states = split.states[start : start + batch_size].to(
            device=device, dtype=torch.float32
        )
        wrong_states = torch.roll(states, shifts=1, dims=0)
        target = split.images_uint8[start : start + batch_size].to(
            device=device, dtype=torch.float32
        ).div(255.0)
        prediction = decoder(states)
        wrong_prediction = decoder(wrong_states)
        correct_each = (
            torch.nn.functional.l1_loss(prediction, target, reduction="none")
            .flatten(1)
            .mean(1)
            + 0.5
            * torch.nn.functional.mse_loss(prediction, target, reduction="none")
            .flatten(1)
            .mean(1)
        )
        wrong_each = (
            torch.nn.functional.l1_loss(wrong_prediction, target, reduction="none")
            .flatten(1)
            .mean(1)
            + 0.5
            * torch.nn.functional.mse_loss(wrong_prediction, target, reduction="none")
            .flatten(1)
            .mean(1)
        )
        l1_each = (
            torch.nn.functional.l1_loss(prediction, target, reduction="none")
            .flatten(1)
            .mean(1)
        )
        mse_each = (
            torch.nn.functional.mse_loss(prediction, target, reduction="none")
            .flatten(1)
            .mean(1)
        )
        batch = states.shape[0]
        total += batch
        correct_loss += float(correct_each.sum().cpu())
        wrong_loss += float(wrong_each.sum().cpu())
        l1_sum += float(l1_each.sum().cpu())
        mse_sum += float(mse_each.sum().cpu())
    correct = correct_loss / total
    wrong = wrong_loss / total
    mse = mse_sum / total
    return {
        "correct_loss": correct,
        "wrong_condition_loss": wrong,
        "wrong_over_correct": wrong / max(correct, 1e-12),
        "l1": l1_sum / total,
        "mse": mse,
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
        "num_items": total,
    }


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_checkpoint(
    path: Path,
    *,
    decoder: WMImageDecoder,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_loss: float,
    invariants: dict[str, Any],
) -> None:
    _atomic_save(
        {
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "best_loss": float(best_loss),
            "invariants": invariants,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        path,
    )


def _init_wandb(args: argparse.Namespace, metadata: dict[str, Any]):
    if args.no_wandb:
        return None
    import wandb

    id_path = args.output_dir / "wandb_run_id.txt"
    run_id = args.wandb_id
    if run_id is None and args.resume and id_path.is_file():
        run_id = id_path.read_text(encoding="utf-8").strip() or None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=run_id,
        resume="allow" if args.resume else None,
        config=metadata,
        dir=str(args.output_dir),
    )
    id_path.write_text(str(run.id), encoding="utf-8")
    return run


def _select_indices(split: LoadedStateImageSplit, count: int) -> list[int]:
    by_record: dict[str, int] = {}
    for index, row in enumerate(split.rows):
        by_record.setdefault(str(row.get("record_id", index)), index)
    candidates = list(by_record.values())
    if len(candidates) <= count:
        return candidates
    return [
        candidates[round(i * (len(candidates) - 1) / max(count - 1, 1))]
        for i in range(count)
    ]


def _labeled_strip(images: list[Image.Image], labels: list[str]) -> Image.Image:
    label_height = 18
    output = Image.new(
        "RGB",
        (sum(image.width for image in images), max(image.height for image in images) + label_height),
        "white",
    )
    draw = ImageDraw.Draw(output)
    offset = 0
    for image, label in zip(images, labels, strict=True):
        output.paste(image, (offset, label_height))
        draw.text((offset + 2, 2), label, fill="black")
        offset += image.width
    return output


@torch.no_grad()
def _save_samples(
    decoder: WMImageDecoder,
    split: LoadedStateImageSplit,
    output_dir: Path,
    device: torch.device,
    *,
    num_items: int,
) -> Path:
    indices = _select_indices(split, num_items)
    states = split.states[indices].to(device=device, dtype=torch.float32)
    predictions = decoder(states).cpu()
    wrong = decoder(torch.roll(states, shifts=1, dims=0)).cpu()
    strips: list[Image.Image] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for offset, index in enumerate(indices):
        gt = Image.fromarray(split.images_uint8[index].permute(1, 2, 0).numpy(), mode="RGB")
        correct_image = Image.fromarray(
            predictions[offset].mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy(),
            mode="RGB",
        )
        wrong_image = Image.fromarray(
            wrong[offset].mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy(),
            mode="RGB",
        )
        strip = _labeled_strip([gt, correct_image, wrong_image], ["GT", "correct state", "wrong state"])
        strip.save(output_dir / f"sample_{offset:03d}_strip.png")
        strips.append(strip)
    width = max(strip.width for strip in strips)
    height = max(strip.height for strip in strips)
    columns = 2
    contact = Image.new(
        "RGB", (columns * width, math.ceil(len(strips) / columns) * height), "white"
    )
    for index, strip in enumerate(strips):
        contact.paste(strip, ((index % columns) * width, (index // columns) * height))
    path = output_dir / "contact_sheet.png"
    contact.save(path)
    return path


def train_deterministic_decoder(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) if torch.cuda.is_available() else None
    train_split = load_state_image_split(
        args.state_cache_dir / "train",
        image_size=args.image_size,
        max_items=args.max_train_items,
        expected_latent_token_count=args.latent_token_count,
    )
    eval_split = load_state_image_split(
        args.state_cache_dir / args.validation_cache_split,
        image_size=args.image_size,
        max_items=args.max_val_items,
        expected_latent_token_count=args.latent_token_count,
    )
    config = WMImageDecoderConfig(
        emb_dim=int(train_split.states.shape[1]),
        image_size=args.image_size,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
    )
    decoder = WMImageDecoder(config).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    invariants = {
        "decoder_config": config.__dict__,
        "train_fingerprint": train_split.manifest["fingerprint"],
        "eval_fingerprint": eval_split.manifest["fingerprint"],
        "train_items": int(train_split.states.shape[0]),
        "eval_items": int(eval_split.states.shape[0]),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }
    total_steps = args.max_steps
    metadata = {
        "task": "deterministic_sft2_state_image_decoder_probe",
        "state_cache_dir": str(args.state_cache_dir),
        "validation_cache_split": args.validation_cache_split,
        "target": "current_image_0_1",
        "loss": "L1 + 0.5*MSE",
        "invariants": invariants,
        "total_steps": total_steps,
        "trainable": "WMImageDecoder only",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    wandb_run = _init_wandb(args, metadata)
    start_step = 0
    best_loss = float("inf")
    if args.resume:
        checkpoints = sorted(args.output_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            raise FileNotFoundError("--resume requested but no checkpoint exists")
        payload = torch.load(checkpoints[-1], map_location=device, weights_only=False)
        if payload["invariants"] != invariants:
            raise ValueError("deterministic decoder resume invariants mismatch")
        decoder.load_state_dict(payload["decoder"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        best_loss = float(payload["best_loss"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state_all"]])
    log_path = args.output_dir / "train_step_log.csv"
    if start_step == 0 or not log_path.exists():
        with log_path.open("w", newline="") as file:
            csv.writer(file).writerow([
                "time", "step", "train_loss", "eval_correct", "eval_wrong", "wrong_over_correct", "psnr", "best_loss"
            ])
    start_time = time.time()
    last_train = float("nan")
    last_eval: dict[str, float] | None = None
    for step in range(start_step + 1, total_steps + 1):
        indices = torch.randint(0, train_split.states.shape[0], (args.batch_size,))
        states = train_split.states[indices].to(device=device, dtype=torch.float32)
        target = train_split.images_uint8[indices].to(device=device, dtype=torch.float32).div(255.0)
        decoder.train()
        prediction = decoder(states)
        loss, l1, mse = reconstruction_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
        optimizer.step()
        last_train = float(loss.detach().cpu())
        evaluate = step == 1 or step % args.eval_interval == 0 or step == total_steps
        if evaluate:
            last_eval = evaluate_condition_sensitivity(
                decoder,
                eval_split,
                device,
                batch_size=args.eval_batch_size,
                max_items=args.eval_max_items,
            )
            if last_eval["correct_loss"] < best_loss:
                best_loss = last_eval["correct_loss"]
                _save_checkpoint(
                    args.output_dir / "best.pt",
                    decoder=decoder,
                    optimizer=optimizer,
                    step=step,
                    best_loss=best_loss,
                    invariants=invariants,
                )
            with log_path.open("a", newline="") as file:
                csv.writer(file).writerow([
                    time.time(), step, last_train, last_eval["correct_loss"], last_eval["wrong_condition_loss"],
                    last_eval["wrong_over_correct"], last_eval["psnr"], best_loss,
                ])
            if wandb_run is not None:
                wandb_run.log({
                    "detdec/train_loss": last_train,
                    "detdec/train_l1": float(l1.detach().cpu()),
                    "detdec/train_mse": float(mse.detach().cpu()),
                    "detdec/eval_correct": last_eval["correct_loss"],
                    "detdec/eval_wrong": last_eval["wrong_condition_loss"],
                    "detdec/wrong_over_correct": last_eval["wrong_over_correct"],
                    "detdec/psnr": last_eval["psnr"],
                }, step=step)
            print(
                json.dumps(
                    {
                        "step": step,
                        "train": last_train,
                        "eval": last_eval,
                        "best": best_loss,
                        "elapsed": time.time() - start_time,
                    }
                ),
                flush=True,
            )
        elif step % args.log_interval == 0:
            if wandb_run is not None:
                wandb_run.log({"detdec/train_loss": last_train}, step=step)
            print(json.dumps({"step": step, "train": last_train, "elapsed": time.time() - start_time}), flush=True)
        if args.save_interval > 0 and step % args.save_interval == 0:
            _save_checkpoint(
                args.output_dir / f"checkpoint_{step:09d}.pt",
                decoder=decoder,
                optimizer=optimizer,
                step=step,
                best_loss=best_loss,
                invariants=invariants,
            )
    final_path = args.output_dir / f"checkpoint_{total_steps:09d}.pt"
    _save_checkpoint(
        final_path,
        decoder=decoder,
        optimizer=optimizer,
        step=total_steps,
        best_loss=best_loss,
        invariants=invariants,
    )
    best_payload = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    decoder.load_state_dict(best_payload["decoder"], strict=True)
    final_eval = evaluate_condition_sensitivity(
        decoder,
        eval_split,
        device,
        batch_size=args.eval_batch_size,
        max_items=-1,
    )
    contact = _save_samples(
        decoder,
        eval_split,
        args.output_dir / "samples",
        device,
        num_items=args.sample_items,
    )
    if wandb_run is not None:
        import wandb

        wandb_run.log({
            "detdec/final_correct": final_eval["correct_loss"],
            "detdec/final_wrong": final_eval["wrong_condition_loss"],
            "detdec/final_wrong_over_correct": final_eval["wrong_over_correct"],
            "detdec/final_psnr": final_eval["psnr"],
            "detdec/samples": wandb.Image(str(contact)),
        }, step=total_steps + 1)
        wandb_run.finish()
    summary = {
        "status": "completed",
        "total_steps": total_steps,
        "last_train_loss": last_train,
        "last_eval": last_eval,
        "best_loss": best_loss,
        "best_step": int(best_payload["step"]),
        "final_best_checkpoint_eval": final_eval,
        "best_checkpoint": str(args.output_dir / "best.pt"),
        "final_checkpoint": str(final_path),
        "contact_sheet": str(contact),
        "elapsed_sec": time.time() - start_time,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic cached-state image decoder probe")
    parser.add_argument("--state-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latent-token-count", type=int, default=8)
    parser.add_argument("--validation-cache-split", choices=("train", "val"), default="train")
    parser.add_argument("--max-train-items", type=int, default=64)
    parser.add_argument("--max-val-items", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-max-items", type=int, default=64)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--sample-items", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-id", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return train_deterministic_decoder(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
