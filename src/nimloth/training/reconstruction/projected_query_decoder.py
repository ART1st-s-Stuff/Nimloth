"""Decode projected WM states back into Qwen latent-query hidden states.

The decoder is a post-hoc diagnostic. SFT2, Qwen, StateProjector, and the WM
predictor stay frozen. Training uses equal supervision from the true current
projected State and from the one-step WM prediction based on the true previous
State and action.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from nimloth.rcdm.state_cache import RCDMStateCacheDataset
from nimloth.wm.predictor import LatentWMPredictor


@dataclass(frozen=True)
class ProjectedQueryDecoderConfig:
    projected_dim: int = 8192
    hidden_dim: int = 8192
    query_tokens: int = 8
    query_dim: int = 2048

    def __post_init__(self) -> None:
        if min(self.projected_dim, self.hidden_dim, self.query_tokens, self.query_dim) < 1:
            raise ValueError("decoder dimensions must be positive")

    @property
    def query_flat_dim(self) -> int:
        return self.query_tokens * self.query_dim


class ProjectedQueryDecoder(nn.Module):
    """Symmetric MLP counterpart of the 16384→8192→8192 StateProjector."""

    def __init__(self, config: ProjectedQueryDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.projected_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.query_flat_dim),
        )

    def forward(self, projected_state: torch.Tensor) -> torch.Tensor:
        if projected_state.ndim != 2 or projected_state.shape[1] != self.config.projected_dim:
            raise ValueError(
                f"expected projected State (B, {self.config.projected_dim}), "
                f"got {tuple(projected_state.shape)}"
            )
        output = self.net(projected_state.to(dtype=next(self.parameters()).dtype))
        return output.view(output.shape[0], self.config.query_tokens, self.config.query_dim)

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n")
        torch.save(self.state_dict(), path / "decoder.pt")

    @classmethod
    def load_checkpoint(
        cls, path: Path, map_location: str | torch.device = "cpu"
    ) -> "ProjectedQueryDecoder":
        if path.is_dir():
            config = ProjectedQueryDecoderConfig(
                **json.loads((path / "config.json").read_text())
            )
            state = torch.load(path / "decoder.pt", map_location=map_location, weights_only=True)
        else:
            payload = torch.load(path, map_location=map_location, weights_only=False)
            config = ProjectedQueryDecoderConfig(**payload["decoder_config"])
            state = payload["decoder"]
        decoder = cls(config)
        decoder.load_state_dict(state, strict=True)
        return decoder


def joint_decoder_loss(
    clean_output: torch.Tensor,
    predicted_output: torch.Tensor,
    target_query: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    clean_mse = torch.nn.functional.mse_loss(clean_output.float(), target_query.float())
    predicted_mse = torch.nn.functional.mse_loss(predicted_output.float(), target_query.float())
    total = clean_mse + predicted_mse
    return total, {
        "clean_mse": float(clean_mse.detach()),
        "predicted_mse": float(predicted_mse.detach()),
        "total": float(total.detach()),
    }


def build_teacher_forced_pairs(
    projected_rows: Sequence[dict[str, Any]],
    query_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(projected_rows) != len(query_rows):
        raise ValueError("projected/query cache lengths differ")
    pairs: list[dict[str, Any]] = []
    previous_projected: dict[str, Any] | None = None
    for projected, query in zip(projected_rows, query_rows, strict=True):
        for key in ("record_id", "step_index", "action_index"):
            if str(projected.get(key)) != str(query.get(key)):
                raise ValueError(
                    f"projected/query alignment mismatch for {key}: "
                    f"{projected.get(key)!r} != {query.get(key)!r}"
                )
        record_id = str(projected["record_id"])
        step_index = int(projected["step_index"])
        if (
            previous_projected is not None
            and str(previous_projected["record_id"]) == record_id
            and int(previous_projected["step_index"]) + 1 == step_index
        ):
            pairs.append(
                {
                    "record_id": record_id,
                    "step_index": step_index,
                    "previous_action": int(previous_projected["action_index"]),
                    "previous_projected": previous_projected["state_emb"].reshape(-1),
                    "current_projected": projected["state_emb"].reshape(-1),
                    "target_query": query["state_emb"],
                    "current_image_path": str(query.get("current_image_path", "")),
                }
            )
        previous_projected = projected
    return pairs


@dataclass
class TeacherForcedSplit:
    previous_projected: torch.Tensor
    current_projected: torch.Tensor
    previous_actions: torch.Tensor
    target_query: torch.Tensor
    rows: list[dict[str, Any]]
    projected_fingerprint: str
    query_fingerprint: str

    def __len__(self) -> int:
        return self.previous_actions.shape[0]


def validate_cache_lineage(
    projected_manifest: dict[str, Any], query_manifest: dict[str, Any]
) -> None:
    source_fingerprint = projected_manifest.get("source_query_fingerprint")
    query_fingerprint = query_manifest.get("fingerprint")
    if not source_fingerprint:
        raise ValueError("projected cache lacks source_query_fingerprint")
    if str(source_fingerprint) != str(query_fingerprint):
        raise ValueError(
            "projected/query cache lineage mismatch: "
            f"{source_fingerprint} != {query_fingerprint}"
        )


def load_teacher_forced_split(projected_dir: Path, query_dir: Path) -> TeacherForcedSplit:
    projected_manifest = json.loads((projected_dir / "manifest.json").read_text())
    query_manifest = json.loads((query_dir / "manifest.json").read_text())
    if projected_manifest.get("representation") != "projected":
        raise ValueError(f"expected projected cache: {projected_dir}")
    if query_manifest.get("representation") != "qwen_query_hidden":
        raise ValueError(f"expected qwen_query_hidden cache: {query_dir}")
    validate_cache_lineage(projected_manifest, query_manifest)
    projected_dataset = RCDMStateCacheDataset(projected_dir)
    query_dataset = RCDMStateCacheDataset(query_dir)
    pairs = build_teacher_forced_pairs(
        [projected_dataset[index] for index in range(len(projected_dataset))],
        [query_dataset[index] for index in range(len(query_dataset))],
    )
    if not pairs:
        raise ValueError(f"no teacher-forced adjacent pairs in {projected_dir}")
    return TeacherForcedSplit(
        previous_projected=torch.stack([pair["previous_projected"] for pair in pairs]),
        current_projected=torch.stack([pair["current_projected"] for pair in pairs]),
        previous_actions=torch.tensor([pair["previous_action"] for pair in pairs], dtype=torch.long),
        target_query=torch.stack([pair["target_query"] for pair in pairs]),
        rows=[{k: value for k, value in pair.items() if not isinstance(value, torch.Tensor)} for pair in pairs],
        projected_fingerprint=str(projected_manifest["fingerprint"]),
        query_fingerprint=str(query_manifest["fingerprint"]),
    )


@torch.no_grad()
def evaluate_decoder(
    decoder: ProjectedQueryDecoder,
    predictor: LatentWMPredictor,
    split: TeacherForcedSplit,
    device: torch.device,
    batch_size: int,
    max_items: int = -1,
) -> dict[str, float | int]:
    count = len(split) if max_items < 0 else min(len(split), max_items)
    totals = {"clean_mse": 0.0, "predicted_mse": 0.0, "clean_cos": 0.0, "predicted_cos": 0.0}
    decoder.eval()
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        previous = split.previous_projected[start:stop].to(device=device, dtype=torch.float32)
        current = split.current_projected[start:stop].to(device=device, dtype=torch.float32)
        actions = split.previous_actions[start:stop].to(device)
        target = split.target_query[start:stop].to(device=device, dtype=torch.float32)
        predicted_state = predictor(previous, actions)
        clean = decoder(current).float()
        predicted = decoder(predicted_state.float()).float()
        size = stop - start
        totals["clean_mse"] += float(torch.nn.functional.mse_loss(clean, target)) * size
        totals["predicted_mse"] += float(torch.nn.functional.mse_loss(predicted, target)) * size
        totals["clean_cos"] += float(torch.nn.functional.cosine_similarity(clean.flatten(1), target.flatten(1)).mean()) * size
        totals["predicted_cos"] += float(torch.nn.functional.cosine_similarity(predicted.flatten(1), target.flatten(1)).mean()) * size
    return {"num_items": count, **{key: value / count for key, value in totals.items()}}


def _checkpoint(
    path: Path,
    *,
    decoder: ProjectedQueryDecoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_total: float,
    invariants: dict[str, Any],
    shuffle_generator: torch.Generator,
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "decoder_config": asdict(decoder.config),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_total": best_total,
            "invariants": invariants,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng_state": random.getstate(),
            "shuffle_rng_state": shuffle_generator.get_state(),
        },
        tmp,
    )
    tmp.replace(path)


def train(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_split = load_teacher_forced_split(args.projected_cache_dir / "train", args.query_cache_dir / "train")
    val_split = load_teacher_forced_split(args.projected_cache_dir / "val", args.query_cache_dir / "val")
    query_shape = tuple(train_split.target_query.shape[1:])
    if query_shape != tuple(val_split.target_query.shape[1:]) or len(query_shape) != 2:
        raise ValueError(f"invalid query shapes: train={query_shape}, val={tuple(val_split.target_query.shape[1:])}")
    config = ProjectedQueryDecoderConfig(
        projected_dim=int(train_split.current_projected.shape[1]),
        hidden_dim=args.hidden_dim,
        query_tokens=int(query_shape[0]),
        query_dim=int(query_shape[1]),
    )
    decoder = ProjectedQueryDecoder(config).to(device)
    predictor = LatentWMPredictor.load_checkpoint(
        args.wm_checkpoint,
        map_location=device,
        history_size_override=args.wm_history_size_override,
    ).to(device).eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    invariants = {
        "decoder_config": asdict(config),
        "loss": "clean_mse + teacher_forced_predicted_mse (1:1)",
        "protocol": "predict current from true previous projected State and previous action",
        "train_projected_fingerprint": train_split.projected_fingerprint,
        "train_query_fingerprint": train_split.query_fingerprint,
        "val_projected_fingerprint": val_split.projected_fingerprint,
        "val_query_fingerprint": val_split.query_fingerprint,
        "train_pairs": len(train_split),
        "val_pairs": len(val_split),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "wm_checkpoint": str(args.wm_checkpoint),
        "wm_history_size_override": args.wm_history_size_override,
    }
    metadata = {
        "task": "projected_wm_state_to_qwen_query_latent_decoder",
        "trainable_modules": "ProjectedQueryDecoder only",
        "frozen_modules": "SFT2 Qwen, StateProjector, LatentWMPredictor, CFM",
        "split_semantics": "strict train adjacent pairs for training; disjoint val adjacent pairs for validation",
        "invariants": invariants,
        "epochs": args.epochs,
        "wandb": {"project": args.wandb_project, "run_name": args.wandb_run_name, "enabled": not args.no_wandb},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shuffle_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    start_epoch = 0
    global_step = 0
    best_total = math.inf
    if args.resume_checkpoint is not None:
        payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if payload["invariants"] != invariants:
            raise ValueError("decoder resume invariants mismatch")
        decoder.load_state_dict(payload["decoder"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        start_epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        best_total = float(payload["best_total"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        random.setstate(payload["python_rng_state"])
        shuffle_generator.set_state(payload["shuffle_rng_state"])
    wandb_run = None
    if not args.no_wandb:
        import wandb
        run_id_path = args.output_dir / "wandb_run_id.txt"
        run_id = run_id_path.read_text().strip() if run_id_path.is_file() else None
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=run_id,
            resume="allow" if run_id else None,
            config=metadata,
            dir=str(args.output_dir),
        )
        run_id_path.write_text(wandb_run.id)
    log_path = args.output_dir / "train_step_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="") as file:
            csv.writer(file).writerow(["time", "epoch", "global_step", "train_total", "train_clean_mse", "train_predicted_mse", "val_clean_mse", "val_predicted_mse", "val_clean_cos", "val_predicted_cos"])
    for epoch in range(start_epoch, args.epochs):
        decoder.train()
        order = torch.randperm(len(train_split), generator=shuffle_generator)
        last_metrics: dict[str, float] = {}
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            previous = train_split.previous_projected[indices].to(device=device, dtype=torch.float32)
            current = train_split.current_projected[indices].to(device=device, dtype=torch.float32)
            actions = train_split.previous_actions[indices].to(device)
            target = train_split.target_query[indices].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                predicted_state = predictor(previous, actions)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                clean_output = decoder(current)
                predicted_output = decoder(predicted_state.float())
            loss, last_metrics = joint_decoder_loss(clean_output, predicted_output, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite decoder loss at step {global_step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1
        val = evaluate_decoder(decoder, predictor, val_split, device, args.eval_batch_size, args.eval_max_items)
        val_total = float(val["clean_mse"]) + float(val["predicted_mse"])
        if val_total < best_total:
            best_total = val_total
            _checkpoint(args.output_dir / "best.pt", decoder=decoder, optimizer=optimizer, epoch=epoch + 1, global_step=global_step, best_total=best_total, invariants=invariants, shuffle_generator=shuffle_generator)
        _checkpoint(args.output_dir / "latest.pt", decoder=decoder, optimizer=optimizer, epoch=epoch + 1, global_step=global_step, best_total=best_total, invariants=invariants, shuffle_generator=shuffle_generator)
        with log_path.open("a", newline="") as file:
            csv.writer(file).writerow([time.time(), epoch + 1, global_step, last_metrics["total"], last_metrics["clean_mse"], last_metrics["predicted_mse"], val["clean_mse"], val["predicted_mse"], val["clean_cos"], val["predicted_cos"]])
        payload = {"epoch": epoch + 1, "global_step": global_step, "train": last_metrics, "val": val, "best_total": best_total}
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/total": last_metrics["total"],
                "train/clean_mse": last_metrics["clean_mse"],
                "train/predicted_mse": last_metrics["predicted_mse"],
                "val/clean_mse": val["clean_mse"],
                "val/predicted_mse": val["predicted_mse"],
                "val/clean_cos": val["clean_cos"],
                "val/predicted_cos": val["predicted_cos"],
            }, step=global_step)
        print(json.dumps(payload), flush=True)
    best = ProjectedQueryDecoder.load_checkpoint(args.output_dir / "best.pt", map_location=device).to(device)
    final_val = evaluate_decoder(best, predictor, val_split, device, args.eval_batch_size, -1)
    (args.output_dir / "final_metrics.json").write_text(json.dumps({"best_total": best_total, "best_full_val": final_val}, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.log({f"full_val/{key}": value for key, value in final_val.items()}, step=global_step + 1)
        wandb_run.finish()
    print(json.dumps({"status": "completed", "best_full_val": final_val}), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train projected-State to query-latent Decoder")
    parser.add_argument("--projected-cache-dir", type=Path, required=True)
    parser.add_argument("--query-cache-dir", type=Path, required=True)
    parser.add_argument("--wm-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-max-items", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--wm-history-size-override", type=int, choices=[1], default=1)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return train(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
