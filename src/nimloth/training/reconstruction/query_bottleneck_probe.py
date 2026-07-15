"""Probe whether cached 8x2048 Query State remains useful through 8x1024.

This is an offline representation-capacity probe. It does not retrain Qwen,
StateProjector, or the world model. A token-wise linear encoder produces the
explicit ``(B, 8, 1024)`` bottleneck; a trainable adapter maps it into the same
proven 16x512 Qwen visual-token target used by the existing 8x2048 baseline.
The established best 8x2048 adapter stays frozen as the positive baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from nimloth.training.reconstruction.state_to_vision_tokens import (
    StateToVisionTokens,
    VisionTokenAdapterConfig,
    _select_indices,
    _strip,
    _tensor_image,
    evaluate_adapter,
    load_aligned_split,
    load_proven_cfm,
    sample_euler_cfg,
    token_loss,
)


class QueryBottleneckAdapter(nn.Module):
    """Token-wise linear 2048→bottleneck encoder plus visual-token adapter."""

    def __init__(
        self,
        *,
        input_tokens: int,
        input_dim: int,
        bottleneck_dim: int,
    ) -> None:
        super().__init__()
        if input_tokens < 1 or input_dim < 1 or bottleneck_dim < 1:
            raise ValueError("input and bottleneck dimensions must be positive")
        self.input_tokens = int(input_tokens)
        self.input_dim = int(input_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.encoder_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Linear(input_dim, bottleneck_dim)
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim)
        self.adapter = StateToVisionTokens(
            VisionTokenAdapterConfig(
                input_tokens=input_tokens,
                input_dim=bottleneck_dim,
            )
        )

    def encode(self, query_hidden: torch.Tensor) -> torch.Tensor:
        expected = (self.input_tokens, self.input_dim)
        if query_hidden.ndim != 3 or tuple(query_hidden.shape[1:]) != expected:
            raise ValueError(
                f"expected query hidden (B, {expected[0]}, {expected[1]}), "
                f"got {tuple(query_hidden.shape)}"
            )
        encoded = self.encoder(self.encoder_norm(query_hidden.float()))
        return self.bottleneck_norm(encoded)

    def forward(self, query_hidden: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.encode(query_hidden))


def initialize_from_baseline(
    bottleneck: QueryBottleneckAdapter,
    baseline: StateToVisionTokens,
) -> int:
    """Copy all shape-compatible downstream weights from the frozen baseline."""

    baseline_state = baseline.state_dict()
    state = bottleneck.adapter.state_dict()
    copied = 0
    for key, value in state.items():
        source = baseline_state.get(key)
        if source is not None and source.shape == value.shape:
            state[key] = source.detach().clone()
            copied += 1
    bottleneck.adapter.load_state_dict(state, strict=True)
    return copied


def load_frozen_baseline(
    checkpoint: Path,
    *,
    input_tokens: int,
    input_dim: int,
    device: torch.device,
) -> tuple[StateToVisionTokens, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "query_adapter" not in payload:
        raise ValueError(f"baseline checkpoint lacks query_adapter: {checkpoint}")
    config = VisionTokenAdapterConfig(input_tokens=input_tokens, input_dim=input_dim)
    baseline = StateToVisionTokens(config)
    baseline.load_state_dict(payload["query_adapter"], strict=True)
    baseline.to(device).eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    return baseline, payload


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint(
    path: Path,
    *,
    model: QueryBottleneckAdapter,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_mse: float,
    invariants: dict[str, Any],
) -> None:
    _atomic_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "best_mse": float(best_mse),
            "invariants": invariants,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
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
        id=run_id or None,
        resume="allow" if args.resume else None,
        config=metadata,
        dir=str(args.output_dir),
    )
    id_path.write_text(str(run.id))
    return run


@torch.no_grad()
def save_samples(
    *,
    baseline: StateToVisionTokens,
    bottleneck: QueryBottleneckAdapter,
    split,
    cfm_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    sample_items: int,
    sample_steps: int,
    cfg_scale: float,
    seed: int,
) -> Path:
    indices = _select_indices(split.rows, sample_items)
    query = split.query_hidden[indices].to(device=device, dtype=torch.float32)
    positive = split.positive_tokens[indices].float()
    conditions = [
        positive,
        torch.roll(positive, 1, 0),
        baseline(query).cpu(),
        bottleneck(query).cpu(),
        bottleneck(torch.roll(query, 1, 0)).cpu(),
    ]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(len(indices), 3, 128, 128, generator=generator)
    cfm = load_proven_cfm(cfm_checkpoint, device)
    generated = [
        sample_euler_cfg(
            cfm,
            condition,
            noise,
            device=device,
            steps=sample_steps,
            cfg_scale=cfg_scale,
        )
        for condition in conditions
    ]
    labels = [
        "GT",
        "Qwen positive",
        "Qwen wrong",
        "Query 8x2048",
        f"Bottleneck 8x{bottleneck.bottleneck_dim}",
        "Bottleneck wrong",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    strips = []
    for offset, index in enumerate(indices):
        with Image.open(split.rows[index]["current_image_path"]) as source:
            gt = source.convert("RGB").resize((128, 128))
        strip = _strip(
            [gt] + [_tensor_image(result[offset]) for result in generated],
            labels,
        )
        strip.save(output_dir / f"sample_{offset:03d}_strip.png")
        strips.append(strip)
    width = max(item.width for item in strips)
    height = max(item.height for item in strips)
    contact = Image.new("RGB", (width, height * len(strips)), "white")
    for index, strip in enumerate(strips):
        contact.paste(strip, (0, index * height))
    path = output_dir / "contact_sheet.png"
    contact.save(path)
    return path


def train(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    train_split = load_aligned_split(
        args.projected_cache_dir / "train",
        args.query_cache_dir / "train",
        args.positive_cache_dir / "train",
        max_items=args.max_train_items,
    )
    val_split = load_aligned_split(
        args.projected_cache_dir / "val",
        args.query_cache_dir / "val",
        args.positive_cache_dir / "val",
        max_items=args.max_val_items,
    )
    input_tokens = int(train_split.query_hidden.shape[1])
    input_dim = int(train_split.query_hidden.shape[2])
    if input_tokens != 8 or input_dim != 2048:
        raise ValueError(
            f"this probe requires cached Query State (8,2048), got {(input_tokens, input_dim)}"
        )
    baseline, baseline_payload = load_frozen_baseline(
        args.baseline_checkpoint,
        input_tokens=input_tokens,
        input_dim=input_dim,
        device=device,
    )
    torch.manual_seed(args.seed + 1)
    model = QueryBottleneckAdapter(
        input_tokens=input_tokens,
        input_dim=input_dim,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)
    copied_keys = initialize_from_baseline(model, baseline)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    invariants = {
        "input_shape": [input_tokens, input_dim],
        "bottleneck_shape": [input_tokens, args.bottleneck_dim],
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "baseline_step": int(baseline_payload.get("step", -1)),
        "cache_fingerprints": {
            key: str(value["fingerprint"])
            for key, value in train_split.manifests.items()
        },
        "train_items": int(train_split.query_hidden.shape[0]),
        "val_items": int(val_split.query_hidden.shape[0]),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }
    metadata = {
        "task": "cached_query_token_bottleneck_sufficiency",
        "invariants": invariants,
        "max_steps": args.max_steps,
        "copied_downstream_keys": copied_keys,
        "interpretation": (
            "A supervised sufficiency probe; matching the frozen 8x2048 baseline "
            "supports 8x1024 sufficiency, while failure does not prove insufficiency."
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    step = 0
    best_mse = float("inf")
    if args.resume:
        checkpoint = args.output_dir / "latest.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError("--resume requested but latest.pt is missing")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["invariants"] != invariants:
            raise ValueError("bottleneck probe resume invariants mismatch")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        step = int(payload["step"])
        best_mse = float(payload["best_mse"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload["cuda_rng_state_all"]]
            )
    baseline_eval = evaluate_adapter(
        baseline,
        val_split.query_hidden,
        val_split.positive_tokens,
        device,
        batch_size=args.eval_batch_size,
        max_items=-1,
    )
    log_path = args.output_dir / "train_step_log.csv"
    if step == 0 or not log_path.is_file():
        with log_path.open("w", newline="") as file:
            csv.writer(file).writerow(
                [
                    "time",
                    "step",
                    "train_loss",
                    "val_mse",
                    "wrong_mse",
                    "val_cos",
                    "baseline_mse",
                    "mse_ratio_to_baseline",
                ]
            )
    wandb_run = _init_wandb(args, metadata)
    generator = torch.Generator(device="cpu")
    started = time.time()
    last_train = float("nan")
    last_eval: dict[str, float] | None = None
    while step < args.max_steps:
        generator.manual_seed(args.seed + step)
        indices = torch.randint(
            0,
            train_split.query_hidden.shape[0],
            (args.batch_size,),
            generator=generator,
        )
        query = train_split.query_hidden[indices].to(
            device=device, dtype=torch.float32
        )
        target = train_split.positive_tokens[indices].to(
            device=device, dtype=torch.float32
        )
        output = model(query)
        loss, mse, cosine = token_loss(output, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        step += 1
        last_train = float(loss.detach().cpu())
        if step % args.log_interval == 0:
            payload = {
                "step": step,
                "train_loss": last_train,
                "train_mse": float(mse.detach().cpu()),
                "train_cos": float(cosine.detach().cpu()),
                "elapsed": time.time() - started,
            }
            print(json.dumps(payload), flush=True)
            if wandb_run is not None:
                wandb_run.log(payload, step=step)
        if step % args.eval_interval == 0 or step == args.max_steps:
            last_eval = evaluate_adapter(
                model,
                val_split.query_hidden,
                val_split.positive_tokens,
                device,
                batch_size=args.eval_batch_size,
                max_items=args.eval_max_items,
            )
            ratio = last_eval["correct_mse"] / baseline_eval["correct_mse"]
            with log_path.open("a", newline="") as file:
                csv.writer(file).writerow(
                    [
                        time.time(),
                        step,
                        last_train,
                        last_eval["correct_mse"],
                        last_eval["wrong_mse"],
                        last_eval["correct_cos"],
                        baseline_eval["correct_mse"],
                        ratio,
                    ]
                )
            if last_eval["correct_mse"] < best_mse:
                best_mse = last_eval["correct_mse"]
                _checkpoint(
                    args.output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_mse=best_mse,
                    invariants=invariants,
                )
            _checkpoint(
                args.output_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                best_mse=best_mse,
                invariants=invariants,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "bottleneck/val_mse": last_eval["correct_mse"],
                        "bottleneck/wrong_mse": last_eval["wrong_mse"],
                        "bottleneck/val_cos": last_eval["correct_cos"],
                        "bottleneck/mse_ratio_to_baseline": ratio,
                        "baseline/val_mse": baseline_eval["correct_mse"],
                        "baseline/val_cos": baseline_eval["correct_cos"],
                    },
                    step=step,
                )
            print(
                json.dumps(
                    {
                        "step": step,
                        "bottleneck_eval": last_eval,
                        "baseline_eval": baseline_eval,
                        "mse_ratio_to_baseline": ratio,
                        "best_mse": best_mse,
                    }
                ),
                flush=True,
            )
    best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    final_eval = evaluate_adapter(
        model,
        val_split.query_hidden,
        val_split.positive_tokens,
        device,
        batch_size=args.eval_batch_size,
        max_items=-1,
    )
    contact = save_samples(
        baseline=baseline,
        bottleneck=model,
        split=val_split,
        cfm_checkpoint=args.cfm_checkpoint,
        output_dir=args.output_dir / "samples",
        device=device,
        sample_items=args.sample_items,
        sample_steps=args.sample_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed + 50000,
    )
    summary = {
        "status": "completed",
        "step": step,
        "best_step": int(best["step"]),
        "bottleneck_shape": [input_tokens, args.bottleneck_dim],
        "baseline": baseline_eval,
        "bottleneck": final_eval,
        "mse_ratio_to_baseline": (
            final_eval["correct_mse"] / baseline_eval["correct_mse"]
        ),
        "cosine_delta_from_baseline": (
            final_eval["correct_cos"] - baseline_eval["correct_cos"]
        ),
        "contact_sheet": str(contact),
        "elapsed_sec": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if wandb_run is not None:
        import wandb

        wandb_run.log(
            {
                "bottleneck/final_mse": final_eval["correct_mse"],
                "bottleneck/final_wrong_mse": final_eval["wrong_mse"],
                "bottleneck/final_cos": final_eval["correct_cos"],
                "bottleneck/final_mse_ratio_to_baseline": summary[
                    "mse_ratio_to_baseline"
                ],
                "bottleneck/contact_sheet": wandb.Image(str(contact)),
            },
            step=step,
        )
        wandb_run.finish()
    print(json.dumps(summary), flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cached 8x1024 Query bottleneck probe")
    parser.add_argument("--projected-cache-dir", type=Path, required=True)
    parser.add_argument("--query-cache-dir", type=Path, required=True)
    parser.add_argument("--positive-cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bottleneck-dim", type=int, default=1024)
    parser.add_argument("--max-train-items", type=int, default=-1)
    parser.add_argument("--max-val-items", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-max-items", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--sample-items", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return train(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
