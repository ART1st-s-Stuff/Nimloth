"""Residual CFM refinement on top of a frozen deterministic state decoder."""

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

from nimloth.cfm import CFMConfig, TokenConditionedFlowUNet
from nimloth.training.reconstruction.cfm_sft2 import (
    LoadedStateImageSplit,
    load_state_image_split,
)
from nimloth.wm.reconstruction import WMImageDecoder, WMImageDecoderConfig


def load_frozen_scaffold(path: Path, device: torch.device) -> WMImageDecoder:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    invariants = payload.get("invariants", {})
    config_data = invariants.get("decoder_config")
    if not isinstance(config_data, dict):
        raise ValueError(f"deterministic scaffold checkpoint has no decoder_config: {path}")
    decoder = WMImageDecoder(WMImageDecoderConfig(**config_data))
    decoder.load_state_dict(payload["decoder"], strict=True)
    decoder.to(device).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder


def biased_flow_loss(
    model: TokenConditionedFlowUNet,
    scaffold: torch.Tensor,
    target_image: torch.Tensor,
    condition: torch.Tensor,
    *,
    reconstruction_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Flow-match image residuals with extra low-time reconstruction pressure."""

    residual_target = target_image - scaffold
    noise = torch.randn_like(residual_target)
    # Squaring uniform samples emphasizes low t, where x_t contains less target
    # information and the state-derived scaffold must carry scene structure.
    time = torch.rand(
        (target_image.shape[0],),
        device=target_image.device,
        dtype=target_image.dtype,
    ).square()
    time_image = time.view(-1, 1, 1, 1)
    interpolated = (1.0 - time_image) * noise + time_image * residual_target
    target_velocity = residual_target - noise
    model_input = torch.cat([interpolated, scaffold], dim=1)
    predicted_velocity = model(model_input, time, condition)
    velocity_mse = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
    predicted_residual_target = interpolated + (1.0 - time_image) * predicted_velocity
    predicted_image = (scaffold + predicted_residual_target).clamp(-1.0, 1.0)
    reconstruction_l1 = torch.nn.functional.l1_loss(predicted_image, target_image)
    loss = velocity_mse + reconstruction_weight * reconstruction_l1
    return loss, {
        "velocity_mse": float(velocity_mse.detach().cpu()),
        "reconstruction_l1": float(reconstruction_l1.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }


@torch.no_grad()
def evaluate_residual_condition_sensitivity(
    model: TokenConditionedFlowUNet,
    scaffold_decoder: WMImageDecoder,
    split: LoadedStateImageSplit,
    device: torch.device,
    *,
    batch_size: int,
    reconstruction_weight: float,
    seed: int,
) -> dict[str, float]:
    model.eval()
    scaffold_decoder.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    correct_sum = 0.0
    wrong_sum = 0.0
    correct_l1_sum = 0.0
    wrong_l1_sum = 0.0
    total = 0
    for start in range(0, split.states.shape[0], batch_size):
        condition = split.states[start : start + batch_size].to(
            device=device, dtype=torch.float32
        )
        wrong_condition = torch.roll(condition, shifts=1, dims=0)
        target = split.images_uint8[start : start + batch_size].to(
            device=device, dtype=torch.float32
        ).div(127.5).sub(1.0)
        scaffold = scaffold_decoder(condition).mul(2.0).sub(1.0)
        wrong_scaffold = scaffold_decoder(wrong_condition).mul(2.0).sub(1.0)
        residual_target = target - scaffold
        wrong_residual_target = target - wrong_scaffold
        batch = condition.shape[0]
        noise = torch.randn(
            residual_target.shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        time = torch.rand(
            (batch,), device=device, dtype=torch.float32, generator=generator
        ).square()
        time_image = time.view(-1, 1, 1, 1)

        def losses(
            state: torch.Tensor,
            base: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            interpolated = (1.0 - time_image) * noise + time_image * residual
            target_velocity = residual - noise
            velocity = model(torch.cat([interpolated, base], dim=1), time, state)
            velocity_mse = (
                torch.nn.functional.mse_loss(
                    velocity, target_velocity, reduction="none"
                )
                .flatten(1)
                .mean(1)
            )
            predicted_residual = interpolated + (1.0 - time_image) * velocity
            predicted_image = (base + predicted_residual).clamp(-1.0, 1.0)
            image_l1 = (
                torch.nn.functional.l1_loss(
                    predicted_image, target, reduction="none"
                )
                .flatten(1)
                .mean(1)
            )
            return velocity_mse + reconstruction_weight * image_l1, image_l1

        correct, correct_l1 = losses(condition, scaffold, residual_target)
        wrong, wrong_l1 = losses(
            wrong_condition, wrong_scaffold, wrong_residual_target
        )
        total += batch
        correct_sum += float(correct.sum().cpu())
        wrong_sum += float(wrong.sum().cpu())
        correct_l1_sum += float(correct_l1.sum().cpu())
        wrong_l1_sum += float(wrong_l1.sum().cpu())
    correct = correct_sum / total
    wrong = wrong_sum / total
    return {
        "correct_loss": correct,
        "wrong_condition_loss": wrong,
        "wrong_over_correct": wrong / max(correct, 1e-12),
        "correct_reconstruction_l1": correct_l1_sum / total,
        "wrong_reconstruction_l1": wrong_l1_sum / total,
        "num_items": total,
    }


@torch.no_grad()
def sample_residual_euler(
    model: TokenConditionedFlowUNet,
    scaffold_decoder: WMImageDecoder,
    condition: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
    *,
    steps: int,
) -> torch.Tensor:
    condition = condition.to(device=device, dtype=torch.float32)
    scaffold = scaffold_decoder(condition).mul(2.0).sub(1.0)
    residual = noise.to(device=device, dtype=torch.float32).clone()
    delta = 1.0 / steps
    for index in range(steps):
        time = torch.full(
            (condition.shape[0],),
            (index + 0.5) / steps,
            device=device,
            dtype=torch.float32,
        )
        residual = residual + delta * model(
            torch.cat([residual, scaffold], dim=1), time, condition
        )
    return (scaffold + residual).clamp(-1.0, 1.0).cpu()


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_checkpoint(
    path: Path,
    *,
    model: TokenConditionedFlowUNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_loss: float,
    invariants: dict[str, Any],
) -> None:
    _atomic_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_loss": best_loss,
            "invariants": invariants,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        },
        path,
    )


def _init_wandb(args: argparse.Namespace, metadata: dict[str, Any]):
    if args.no_wandb:
        return None
    import wandb

    id_path = args.output_dir / "wandb_run_id.txt"
    run_id = id_path.read_text().strip() if args.resume and id_path.is_file() else None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=run_id,
        resume="allow" if args.resume else None,
        config=metadata,
        dir=str(args.output_dir),
    )
    id_path.write_text(str(run.id))
    return run


def _save_samples(
    *,
    model: TokenConditionedFlowUNet,
    scaffold_decoder: WMImageDecoder,
    split: LoadedStateImageSplit,
    output_dir: Path,
    device: torch.device,
    steps: int,
    count: int,
    seed: int,
) -> Path:
    indices = list(range(min(count, split.states.shape[0])))
    condition = split.states[indices].float()
    wrong_condition = torch.roll(condition, shifts=1, dims=0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((len(indices), 3, 128, 128), generator=generator)
    correct = sample_residual_euler(
        model, scaffold_decoder, condition, noise, device, steps=steps
    )
    wrong = sample_residual_euler(
        model, scaffold_decoder, wrong_condition, noise, device, steps=steps
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    strips: list[Image.Image] = []
    for offset, index in enumerate(indices):
        gt = Image.fromarray(split.images_uint8[index].permute(1, 2, 0).numpy(), mode="RGB")
        correct_image = Image.fromarray(
            correct[offset].add(1).mul(127.5).byte().permute(1, 2, 0).numpy(), mode="RGB"
        )
        wrong_image = Image.fromarray(
            wrong[offset].add(1).mul(127.5).byte().permute(1, 2, 0).numpy(), mode="RGB"
        )
        label_height = 18
        strip = Image.new("RGB", (384, 128 + label_height), "white")
        draw = ImageDraw.Draw(strip)
        for column, (image, label) in enumerate(
            zip(
                [gt, correct_image, wrong_image],
                ["GT", "scaffold+residual", "wrong condition"],
                strict=True,
            )
        ):
            strip.paste(image, (128 * column, label_height))
            draw.text((128 * column + 2, 2), label, fill="black")
        strip.save(output_dir / f"sample_{offset:03d}_strip.png")
        strips.append(strip)
    contact = Image.new("RGB", (768, math.ceil(len(strips) / 2) * 146), "white")
    for index, strip in enumerate(strips):
        contact.paste(strip, ((index % 2) * 384, (index // 2) * 146))
    path = output_dir / "contact_sheet.png"
    contact.save(path)
    return path


def train_residual_cfm(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_split = load_state_image_split(
        args.state_cache_dir / "train",
        image_size=128,
        max_items=args.max_train_items,
        expected_latent_token_count=8,
    )
    eval_split = load_state_image_split(
        args.state_cache_dir / args.validation_cache_split,
        image_size=128,
        max_items=args.max_val_items,
        expected_latent_token_count=8,
    )
    scaffold_decoder = load_frozen_scaffold(args.scaffold_checkpoint, device)
    config = CFMConfig(
        image_size=128,
        token_count=1,
        token_dim=int(train_split.states.shape[1]),
        base_channels=64,
        condition_dim=256,
        time_dim=512,
        input_channels=6,
        output_channels=3,
    )
    model = TokenConditionedFlowUNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    invariants = {
        "config": config.to_metadata(),
        "scaffold_checkpoint": str(args.scaffold_checkpoint),
        "train_fingerprint": train_split.manifest["fingerprint"],
        "eval_fingerprint": eval_split.manifest["fingerprint"],
        "train_items": int(train_split.states.shape[0]),
        "eval_items": int(eval_split.states.shape[0]),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "reconstruction_weight": args.reconstruction_weight,
        "seed": args.seed,
    }
    metadata = {
        "task": "deterministic_scaffold_residual_cfm",
        "trainable": "six-channel residual TokenConditionedFlowUNet",
        "frozen": "deterministic state decoder scaffold and SFT2 cache source",
        "time_sampling": "t = uniform(0,1)^2, biased toward low t",
        "loss": "residual velocity MSE + reconstruction_weight * image L1",
        "invariants": invariants,
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
            raise ValueError("residual CFM resume invariants mismatch")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        best_loss = float(payload["best_loss"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state_all"]])
    log_path = args.output_dir / "train_step_log.csv"
    if start_step == 0 or not log_path.is_file():
        with log_path.open("w", newline="") as file:
            csv.writer(file).writerow(
                [
                    "time",
                    "step",
                    "train_loss",
                    "velocity_mse",
                    "reconstruction_l1",
                    "eval_correct",
                    "eval_wrong",
                    "wrong_over_correct",
                    "best_loss",
                ]
            )
    start_time = time.time()
    last_parts: dict[str, float] = {}
    last_eval: dict[str, float] | None = None
    for step in range(start_step + 1, args.max_steps + 1):
        indices = torch.randint(0, train_split.states.shape[0], (args.batch_size,))
        condition = train_split.states[indices].to(device=device, dtype=torch.float32)
        target = train_split.images_uint8[indices].to(device=device, dtype=torch.float32).div(127.5).sub(1.0)
        with torch.no_grad():
            scaffold = scaffold_decoder(condition).mul(2.0).sub(1.0)
        model.train()
        loss, last_parts = biased_flow_loss(
            model,
            scaffold,
            target,
            condition,
            reconstruction_weight=args.reconstruction_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        evaluate = step == 1 or step % args.eval_interval == 0 or step == args.max_steps
        if evaluate:
            last_eval = evaluate_residual_condition_sensitivity(
                model,
                scaffold_decoder,
                eval_split,
                device,
                batch_size=args.eval_batch_size,
                reconstruction_weight=args.reconstruction_weight,
                seed=args.seed + 10_000 + step,
            )
            if last_eval["correct_loss"] < best_loss:
                best_loss = last_eval["correct_loss"]
                _save_checkpoint(
                    args.output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_loss=best_loss,
                    invariants=invariants,
                )
            with log_path.open("a", newline="") as file:
                csv.writer(file).writerow([
                    time.time(), step, last_parts["loss"], last_parts["velocity_mse"], last_parts["reconstruction_l1"],
                    last_eval["correct_loss"],
                    last_eval["wrong_condition_loss"],
                    last_eval["wrong_over_correct"],
                    best_loss,
                ])
            if wandb_run is not None:
                wandb_run.log({
                    "rescfm/train_loss": last_parts["loss"],
                    "rescfm/velocity_mse": last_parts["velocity_mse"],
                    "rescfm/reconstruction_l1": last_parts["reconstruction_l1"],
                    "rescfm/eval_correct": last_eval["correct_loss"],
                    "rescfm/eval_wrong": last_eval["wrong_condition_loss"],
                    "rescfm/wrong_over_correct": last_eval["wrong_over_correct"],
                }, step=step)
            print(
                json.dumps(
                    {
                        "step": step,
                        "train": last_parts,
                        "eval": last_eval,
                        "best": best_loss,
                        "elapsed": time.time() - start_time,
                    }
                ),
                flush=True,
            )
        elif step % args.log_interval == 0:
            print(json.dumps({"step": step, "train": last_parts, "elapsed": time.time() - start_time}), flush=True)
        if args.save_interval > 0 and step % args.save_interval == 0:
            _save_checkpoint(
                args.output_dir / f"checkpoint_{step:09d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                best_loss=best_loss,
                invariants=invariants,
            )
    final_path = args.output_dir / f"checkpoint_{args.max_steps:09d}.pt"
    _save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        step=args.max_steps,
        best_loss=best_loss,
        invariants=invariants,
    )
    best_payload = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"], strict=True)
    final_eval = evaluate_residual_condition_sensitivity(
        model,
        scaffold_decoder,
        eval_split,
        device,
        batch_size=args.eval_batch_size,
        reconstruction_weight=args.reconstruction_weight,
        seed=args.seed + 30_000,
    )
    contact = _save_samples(
        model=model,
        scaffold_decoder=scaffold_decoder,
        split=eval_split,
        output_dir=args.output_dir / "samples_50ode",
        device=device,
        steps=50,
        count=8,
        seed=args.seed + 40_000,
    )
    if wandb_run is not None:
        import wandb

        wandb_run.log({
            "rescfm/final_correct": final_eval["correct_loss"],
            "rescfm/final_wrong": final_eval["wrong_condition_loss"],
            "rescfm/final_wrong_over_correct": final_eval["wrong_over_correct"],
            "rescfm/samples": wandb.Image(str(contact)),
        }, step=args.max_steps + 1)
        wandb_run.finish()
    summary = {
        "status": "completed",
        "best_step": int(best_payload["step"]),
        "best_loss": best_loss,
        "last_train": last_parts,
        "last_eval": last_eval,
        "final_best_eval": final_eval,
        "best_checkpoint": str(args.output_dir / "best.pt"),
        "final_checkpoint": str(final_path),
        "contact_sheet": str(contact),
        "elapsed_sec": time.time() - start_time,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic-scaffold residual CFM diagnostic")
    parser.add_argument("--state-cache-dir", type=Path, required=True)
    parser.add_argument("--scaffold-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-cache-split", choices=("train", "val"), default="train")
    parser.add_argument("--max-train-items", type=int, default=64)
    parser.add_argument("--max-val-items", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return train_residual_cfm(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
