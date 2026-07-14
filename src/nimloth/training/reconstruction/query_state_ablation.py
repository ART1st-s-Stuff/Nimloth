"""Controlled reconstruction probe for projected versus preprojection query states.

Both branches use the same token-conditioned image-decoder body and see the
same optimization examples.  The only intended representation difference is
one projected 1024-d token versus the k Qwen latent-query hidden vectors before
``StateProjector``.  This is a diagnostic; it never updates Qwen or SFT2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import nn

from nimloth.rcdm.state_cache import RCDMStateCacheDataset
from nimloth.training.reconstruction.cfm_sft2 import _load_image_uint8


@dataclass(frozen=True)
class TokenImageDecoderConfig:
    condition_dim: int
    condition_tokens: int
    image_size: int = 128
    patch_size: int = 8
    hidden_dim: int = 256
    depth: int = 4
    heads: int = 8
    mlp_ratio: int = 4

    def __post_init__(self) -> None:
        if self.condition_dim < 1 or self.condition_tokens < 1:
            raise ValueError("condition dimensions and token count must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")


class _ConditionedPatchBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.patch_cross_norm = nn.LayerNorm(hidden_dim)
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.patch_self_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, patches: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        normalized_condition = self.condition_norm(condition)
        cross, _ = self.cross_attention(
            self.patch_cross_norm(patches),
            normalized_condition,
            normalized_condition,
            need_weights=False,
        )
        patches = patches + cross
        normalized_patches = self.patch_self_norm(patches)
        attended, _ = self.self_attention(
            normalized_patches,
            normalized_patches,
            normalized_patches,
            need_weights=False,
        )
        patches = patches + attended
        return patches + self.mlp(self.mlp_norm(patches))


class TokenConditionedImageDecoder(nn.Module):
    """Decode one or more condition tokens with a shared spatial architecture."""

    def __init__(self, config: TokenImageDecoderConfig) -> None:
        super().__init__()
        self.config = config
        grid = config.image_size // config.patch_size
        self.num_patches = grid * grid
        self.input_norm = nn.LayerNorm(config.condition_dim)
        self.condition_projection = nn.Linear(config.condition_dim, config.hidden_dim)
        self.condition_position = nn.Parameter(
            torch.randn(1, config.condition_tokens, config.hidden_dim) * 0.02
        )
        self.patch_position = nn.Parameter(
            torch.randn(1, self.num_patches, config.hidden_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                _ConditionedPatchBlock(config.hidden_dim, config.heads, config.mlp_ratio)
                for _ in range(config.depth)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.patch_head = nn.Linear(
            config.hidden_dim,
            config.patch_size * config.patch_size * 3,
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim == 2 and self.config.condition_tokens == 1:
            condition = condition[:, None, :]
        expected = (
            condition.shape[0] if condition.ndim >= 1 else -1,
            self.config.condition_tokens,
            self.config.condition_dim,
        )
        if condition.ndim != 3 or tuple(condition.shape[1:]) != expected[1:]:
            raise ValueError(
                "condition must have shape "
                f"(B, {self.config.condition_tokens}, {self.config.condition_dim}), "
                f"got {tuple(condition.shape)}"
            )
        embedded = self.condition_projection(self.input_norm(condition.float()))
        embedded = embedded + self.condition_position
        # The initial patch stream already depends on the condition.  Cross
        # attention then retains access to all k query vectors at every block.
        patches = embedded.mean(dim=1, keepdim=True) + self.patch_position
        for block in self.blocks:
            patches = block(patches, embedded)
        patches = self.patch_head(self.output_norm(patches))
        cfg = self.config
        grid = cfg.image_size // cfg.patch_size
        images = patches.view(
            patches.shape[0], grid, grid, cfg.patch_size, cfg.patch_size, 3
        )
        images = images.permute(0, 5, 1, 3, 2, 4).reshape(
            patches.shape[0], 3, cfg.image_size, cfg.image_size
        )
        return torch.sigmoid(images)


@dataclass
class PairedStateImageSplit:
    projected: torch.Tensor
    query_hidden: torch.Tensor
    images_uint8: torch.Tensor
    rows: list[dict[str, Any]]
    projected_manifest: dict[str, Any]
    query_manifest: dict[str, Any]


def _manifest(cache_dir: Path) -> dict[str, Any]:
    return json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))


def _representation(manifest: dict[str, Any]) -> str:
    return str(manifest.get("representation", "projected"))


def load_paired_state_image_split(
    projected_cache_dir: Path,
    query_cache_dir: Path,
    *,
    image_size: int,
    max_items: int = -1,
    expected_query_tokens: int = 8,
) -> PairedStateImageSplit:
    """Load aligned representations and resize each target image only once."""

    projected_manifest = _manifest(projected_cache_dir)
    query_manifest = _manifest(query_cache_dir)
    if _representation(projected_manifest) != "projected":
        raise ValueError("projected cache does not contain projected states")
    if _representation(query_manifest) != "qwen_query_hidden":
        raise ValueError("query cache does not contain qwen_query_hidden states")
    query_shape = tuple(int(dim) for dim in query_manifest.get("state_shape", []))
    if len(query_shape) != 2 or query_shape[0] != expected_query_tokens:
        raise ValueError(
            f"query cache state_shape must be ({expected_query_tokens}, H), got {query_shape}"
        )

    projected_dataset = RCDMStateCacheDataset(projected_cache_dir)
    query_dataset = RCDMStateCacheDataset(query_cache_dir)
    if len(projected_dataset) != len(query_dataset):
        raise ValueError(
            f"cache item count mismatch: projected={len(projected_dataset)}, query={len(query_dataset)}"
        )
    count = len(projected_dataset) if max_items < 0 else min(max_items, len(projected_dataset))
    if count < 2:
        raise ValueError(f"paired split needs at least two items, got {count}")

    projected_dim = int(projected_manifest["cond_dim"])
    projected = torch.empty((count, projected_dim), dtype=torch.float16)
    query_hidden = torch.empty((count, *query_shape), dtype=torch.float16)
    images = torch.empty((count, 3, image_size, image_size), dtype=torch.uint8)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        projected_item = projected_dataset[index]
        query_item = query_dataset[index]
        for key in ("id", "record_id", "step_index", "current_image_path"):
            if str(projected_item.get(key, "")) != str(query_item.get(key, "")):
                raise ValueError(
                    f"cache row alignment mismatch at index={index}, key={key}: "
                    f"{projected_item.get(key)!r} != {query_item.get(key)!r}"
                )
        projected[index].copy_(projected_item["state_emb"].reshape(-1).to(torch.float16))
        query_hidden[index].copy_(query_item["state_emb"].to(torch.float16))
        image_path = str(projected_item["current_image_path"])
        images[index].copy_(_load_image_uint8(image_path, image_size))
        rows.append(
            {
                "id": str(projected_item["id"]),
                "record_id": str(projected_item.get("record_id", "")),
                "step_index": int(projected_item.get("step_index", -1)),
                "current_image_path": image_path,
            }
        )
        if (index + 1) % 5000 == 0:
            print(json.dumps({"paired_preload": str(query_cache_dir), "items": index + 1, "total": count}), flush=True)
    return PairedStateImageSplit(
        projected,
        query_hidden,
        images,
        rows,
        projected_manifest,
        query_manifest,
    )


def reconstruction_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l1 = torch.nn.functional.l1_loss(prediction, target)
    mse = torch.nn.functional.mse_loss(prediction, target)
    return l1 + 0.5 * mse, l1, mse


@torch.no_grad()
def evaluate_condition_sensitivity(
    decoder: TokenConditionedImageDecoder,
    states: torch.Tensor,
    images_uint8: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    max_items: int,
) -> dict[str, float]:
    decoder.eval()
    count = states.shape[0] if max_items < 0 else min(max_items, states.shape[0])
    wrong_indices = torch.roll(torch.arange(count), shifts=1)
    totals = {"correct": 0.0, "wrong": 0.0, "l1": 0.0, "mse": 0.0, "delta": 0.0}
    total = 0
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        condition = states[start:end].to(device=device, dtype=torch.float32)
        wrong_condition = states[wrong_indices[start:end]].to(device=device, dtype=torch.float32)
        target = images_uint8[start:end].to(device=device, dtype=torch.float32).div(255.0)
        prediction = decoder(condition)
        wrong_prediction = decoder(wrong_condition)
        correct_each = (
            torch.nn.functional.l1_loss(prediction, target, reduction="none").flatten(1).mean(1)
            + 0.5 * torch.nn.functional.mse_loss(prediction, target, reduction="none").flatten(1).mean(1)
        )
        wrong_each = (
            torch.nn.functional.l1_loss(wrong_prediction, target, reduction="none").flatten(1).mean(1)
            + 0.5 * torch.nn.functional.mse_loss(wrong_prediction, target, reduction="none").flatten(1).mean(1)
        )
        l1_each = torch.nn.functional.l1_loss(prediction, target, reduction="none").flatten(1).mean(1)
        mse_each = torch.nn.functional.mse_loss(prediction, target, reduction="none").flatten(1).mean(1)
        delta_each = torch.nn.functional.l1_loss(prediction, wrong_prediction, reduction="none").flatten(1).mean(1)
        totals["correct"] += float(correct_each.sum().cpu())
        totals["wrong"] += float(wrong_each.sum().cpu())
        totals["l1"] += float(l1_each.sum().cpu())
        totals["mse"] += float(mse_each.sum().cpu())
        totals["delta"] += float(delta_each.sum().cpu())
        total += end - start
    correct = totals["correct"] / total
    mse = totals["mse"] / total
    return {
        "correct_loss": correct,
        "wrong_condition_loss": totals["wrong"] / total,
        "wrong_over_correct": (totals["wrong"] / total) / max(correct, 1e-12),
        "l1": totals["l1"] / total,
        "mse": mse,
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
        "correct_wrong_output_l1": totals["delta"] / total,
        "num_items": total,
    }


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _invariants(
    args: argparse.Namespace,
    train_split: PairedStateImageSplit,
    projected_config: TokenImageDecoderConfig,
    query_config: TokenImageDecoderConfig,
) -> dict[str, Any]:
    return {
        "projected_config": asdict(projected_config),
        "query_config": asdict(query_config),
        "projected_train_fingerprint": str(train_split.projected_manifest["fingerprint"]),
        "query_train_fingerprint": str(train_split.query_manifest["fingerprint"]),
        "train_items": int(train_split.projected.shape[0]),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
    }


def _save_checkpoint(
    path: Path,
    *,
    projected_decoder: TokenConditionedImageDecoder,
    query_decoder: TokenConditionedImageDecoder,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_loss: dict[str, float],
    invariants: dict[str, Any],
) -> None:
    _atomic_save(
        {
            "projected_decoder": projected_decoder.state_dict(),
            "query_decoder": query_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "best_loss": {key: float(value) for key, value in best_loss.items()},
            "invariants": invariants,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def _restore_checkpoint(
    path: Path,
    *,
    projected_decoder: TokenConditionedImageDecoder,
    query_decoder: TokenConditionedImageDecoder,
    optimizer: torch.optim.Optimizer,
    invariants: dict[str, Any],
) -> tuple[int, dict[str, float]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("invariants") != invariants:
        raise ValueError("query-state ablation resume invariants mismatch")
    projected_decoder.load_state_dict(checkpoint["projected_decoder"], strict=True)
    query_decoder.load_state_dict(checkpoint["query_decoder"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])
    return int(checkpoint["step"]), {
        key: float(value) for key, value in checkpoint["best_loss"].items()
    }


def _latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("checkpoint_*.pt"))
    return candidates[-1] if candidates else None


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


def _select_indices(rows: list[dict[str, Any]], count: int) -> list[int]:
    by_record: dict[str, int] = {}
    for index, row in enumerate(rows):
        by_record.setdefault(str(row.get("record_id", index)), index)
    candidates = list(by_record.values())
    if len(candidates) <= count:
        return candidates
    return [candidates[round(i * (len(candidates) - 1) / max(count - 1, 1))] for i in range(count)]


def _image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def _labeled_strip(images: list[Image.Image], labels: list[str]) -> Image.Image:
    label_height = 18
    output = Image.new("RGB", (sum(image.width for image in images), images[0].height + label_height), "white")
    draw = ImageDraw.Draw(output)
    offset = 0
    for image, label in zip(images, labels, strict=True):
        output.paste(image, (offset, label_height))
        draw.text((offset + 2, 2), label, fill="black")
        offset += image.width
    return output


@torch.no_grad()
def _save_samples(
    projected_decoder: TokenConditionedImageDecoder,
    query_decoder: TokenConditionedImageDecoder,
    split: PairedStateImageSplit,
    output_dir: Path,
    device: torch.device,
    *,
    count: int,
) -> Path:
    indices = _select_indices(split.rows, count)
    projected = split.projected[indices].to(device=device, dtype=torch.float32)
    query = split.query_hidden[indices].to(device=device, dtype=torch.float32)
    projected_correct = projected_decoder(projected)
    projected_wrong = projected_decoder(torch.roll(projected, shifts=1, dims=0))
    query_correct = query_decoder(query)
    query_wrong = query_decoder(torch.roll(query, shifts=1, dims=0))
    output_dir.mkdir(parents=True, exist_ok=True)
    strips: list[Image.Image] = []
    labels = ["GT", "projected", "projected wrong", "query hidden", "query wrong"]
    for offset, index in enumerate(indices):
        strip = _labeled_strip(
            [
                Image.fromarray(split.images_uint8[index].permute(1, 2, 0).numpy(), mode="RGB"),
                _image(projected_correct[offset]),
                _image(projected_wrong[offset]),
                _image(query_correct[offset]),
                _image(query_wrong[offset]),
            ],
            labels,
        )
        strip.save(output_dir / f"sample_{offset:03d}_strip.png")
        strips.append(strip)
    columns = 1
    width = max(strip.width for strip in strips)
    height = max(strip.height for strip in strips)
    contact = Image.new("RGB", (columns * width, len(strips) * height), "white")
    for index, strip in enumerate(strips):
        contact.paste(strip, (0, index * height))
    path = output_dir / "contact_sheet.png"
    contact.save(path)
    return path


def train_query_state_ablation(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    train_split = load_paired_state_image_split(
        args.projected_cache_dir / "train",
        args.query_cache_dir / "train",
        image_size=args.image_size,
        max_items=args.max_train_items,
        expected_query_tokens=args.query_token_count,
    )
    val_split = load_paired_state_image_split(
        args.projected_cache_dir / args.validation_cache_split,
        args.query_cache_dir / args.validation_cache_split,
        image_size=args.image_size,
        max_items=args.max_val_items,
        expected_query_tokens=args.query_token_count,
    )
    projected_config = TokenImageDecoderConfig(
        condition_dim=train_split.projected.shape[-1],
        condition_tokens=1,
        image_size=args.image_size,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
    )
    query_config = TokenImageDecoderConfig(
        condition_dim=train_split.query_hidden.shape[-1],
        condition_tokens=train_split.query_hidden.shape[1],
        image_size=args.image_size,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
    )
    # Reset the seed before each construction so all shape-compatible body
    # parameters begin identically; only condition adapters differ in shape.
    torch.manual_seed(args.seed + 1)
    projected_decoder = TokenConditionedImageDecoder(projected_config).to(device)
    torch.manual_seed(args.seed + 1)
    query_decoder = TokenConditionedImageDecoder(query_config).to(device)
    projected_state = projected_decoder.state_dict()
    query_state = query_decoder.state_dict()
    for key, value in projected_state.items():
        if key in query_state and query_state[key].shape == value.shape:
            query_state[key] = value.detach().clone()
    query_decoder.load_state_dict(query_state, strict=True)
    optimizer = torch.optim.AdamW(
        list(projected_decoder.parameters()) + list(query_decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    invariants = _invariants(args, train_split, projected_config, query_config)
    metadata = {
        "task": "projected_vs_preprojection_query_state_reconstruction",
        "invariants": invariants,
        "validation_cache_split": args.validation_cache_split,
        "max_steps": args.max_steps,
        "eval_max_items": args.eval_max_items,
        "frozen": "Qwen, latent query states, StateProjector, SFT2",
        "trainable": "paired token-conditioned image decoders",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    step = 0
    best_loss = {"projected": float("inf"), "query_hidden": float("inf")}
    if args.resume:
        checkpoint = _latest_checkpoint(args.output_dir)
        if checkpoint is None:
            raise FileNotFoundError(f"--resume requested but no checkpoint exists in {args.output_dir}")
        step, best_loss = _restore_checkpoint(
            checkpoint,
            projected_decoder=projected_decoder,
            query_decoder=query_decoder,
            optimizer=optimizer,
            invariants=invariants,
        )
    log_path = args.output_dir / "train_step_log.csv"
    if step == 0 or not log_path.is_file():
        with log_path.open("w", newline="") as file:
            csv.writer(file).writerow(
                [
                    "time", "step", "projected_train", "query_train",
                    "projected_eval", "projected_wrong", "projected_ratio",
                    "query_eval", "query_wrong", "query_ratio", "best_loss",
                ]
            )
    wandb_run = _init_wandb(args, metadata)
    generator = torch.Generator(device="cpu")
    started = time.time()
    last_train = {"projected": float("nan"), "query": float("nan")}
    last_eval: dict[str, Any] | None = None

    while step < args.max_steps:
        # Counter-based sampling makes the data sequence resume exactly from
        # the checkpoint step without storing a second RNG state.
        generator.manual_seed(args.seed + step)
        indices = torch.randint(
            0,
            train_split.projected.shape[0],
            (args.batch_size,),
            generator=generator,
        )
        target = train_split.images_uint8[indices].to(device=device, dtype=torch.float32).div(255.0)
        projected_condition = train_split.projected[indices].to(device=device, dtype=torch.float32)
        query_condition = train_split.query_hidden[indices].to(device=device, dtype=torch.float32)
        projected_prediction = projected_decoder(projected_condition)
        query_prediction = query_decoder(query_condition)
        projected_loss, projected_l1, projected_mse = reconstruction_loss(projected_prediction, target)
        query_loss, query_l1, query_mse = reconstruction_loss(query_prediction, target)
        optimizer.zero_grad(set_to_none=True)
        (projected_loss + query_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            list(projected_decoder.parameters()) + list(query_decoder.parameters()),
            args.grad_clip,
        )
        optimizer.step()
        step += 1
        last_train = {"projected": float(projected_loss.detach().cpu()), "query": float(query_loss.detach().cpu())}

        should_eval = args.eval_interval > 0 and (step % args.eval_interval == 0 or step == args.max_steps)
        if should_eval:
            projected_eval = evaluate_condition_sensitivity(
                projected_decoder,
                val_split.projected,
                val_split.images_uint8,
                device,
                batch_size=args.eval_batch_size,
                max_items=args.eval_max_items,
            )
            query_eval = evaluate_condition_sensitivity(
                query_decoder,
                val_split.query_hidden,
                val_split.images_uint8,
                device,
                batch_size=args.eval_batch_size,
                max_items=args.eval_max_items,
            )
            last_eval = {"projected": projected_eval, "query_hidden": query_eval}
            if projected_eval["correct_loss"] < best_loss["projected"]:
                best_loss["projected"] = projected_eval["correct_loss"]
                _save_checkpoint(
                    args.output_dir / "best_projected.pt",
                    projected_decoder=projected_decoder,
                    query_decoder=query_decoder,
                    optimizer=optimizer,
                    step=step,
                    best_loss=best_loss,
                    invariants=invariants,
                )
            if query_eval["correct_loss"] < best_loss["query_hidden"]:
                best_loss["query_hidden"] = query_eval["correct_loss"]
                _save_checkpoint(
                    args.output_dir / "best_query_hidden.pt",
                    projected_decoder=projected_decoder,
                    query_decoder=query_decoder,
                    optimizer=optimizer,
                    step=step,
                    best_loss=best_loss,
                    invariants=invariants,
                )
            with log_path.open("a", newline="") as file:
                csv.writer(file).writerow(
                    [
                        time.time(), step, last_train["projected"], last_train["query"],
                        projected_eval["correct_loss"], projected_eval["wrong_condition_loss"], projected_eval["wrong_over_correct"],
                        query_eval["correct_loss"], query_eval["wrong_condition_loss"], query_eval["wrong_over_correct"], json.dumps(best_loss, sort_keys=True),
                    ]
                )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "projected/eval_correct": projected_eval["correct_loss"],
                        "projected/eval_wrong": projected_eval["wrong_condition_loss"],
                        "projected/wrong_over_correct": projected_eval["wrong_over_correct"],
                        "projected/psnr": projected_eval["psnr"],
                        "projected/output_delta": projected_eval["correct_wrong_output_l1"],
                        "query/eval_correct": query_eval["correct_loss"],
                        "query/eval_wrong": query_eval["wrong_condition_loss"],
                        "query/wrong_over_correct": query_eval["wrong_over_correct"],
                        "query/psnr": query_eval["psnr"],
                        "query/output_delta": query_eval["correct_wrong_output_l1"],
                    },
                    step=step,
                )
            print(json.dumps({"step": step, "eval": last_eval, "best": best_loss, "elapsed": time.time() - started}), flush=True)

        if step % args.log_interval == 0:
            payload = {
                "step": step,
                "train": {
                    "projected": last_train["projected"],
                    "query_hidden": last_train["query"],
                    "projected_l1": float(projected_l1.detach().cpu()),
                    "projected_mse": float(projected_mse.detach().cpu()),
                    "query_l1": float(query_l1.detach().cpu()),
                    "query_mse": float(query_mse.detach().cpu()),
                },
                "elapsed": time.time() - started,
            }
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "projected/train_loss": last_train["projected"],
                        "query/train_loss": last_train["query"],
                    },
                    step=step,
                )
            print(json.dumps(payload), flush=True)

        if args.save_interval > 0 and step % args.save_interval == 0:
            _save_checkpoint(
                args.output_dir / f"checkpoint_{step:09d}.pt",
                projected_decoder=projected_decoder,
                query_decoder=query_decoder,
                optimizer=optimizer,
                step=step,
                best_loss=best_loss,
                invariants=invariants,
            )

    final_checkpoint = args.output_dir / f"checkpoint_{step:09d}.pt"
    _save_checkpoint(
        final_checkpoint,
        projected_decoder=projected_decoder,
        query_decoder=query_decoder,
        optimizer=optimizer,
        step=step,
        best_loss=best_loss,
        invariants=invariants,
    )
    best_projected_checkpoint = args.output_dir / "best_projected.pt"
    best_query_checkpoint = args.output_dir / "best_query_hidden.pt"
    best_projected_payload = torch.load(best_projected_checkpoint, map_location=device, weights_only=False)
    best_query_payload = torch.load(best_query_checkpoint, map_location=device, weights_only=False)
    projected_decoder.load_state_dict(best_projected_payload["projected_decoder"], strict=True)
    query_decoder.load_state_dict(best_query_payload["query_decoder"], strict=True)
    final_eval = {
        "projected": evaluate_condition_sensitivity(
            projected_decoder, val_split.projected, val_split.images_uint8, device,
            batch_size=args.eval_batch_size, max_items=-1,
        ),
        "query_hidden": evaluate_condition_sensitivity(
            query_decoder, val_split.query_hidden, val_split.images_uint8, device,
            batch_size=args.eval_batch_size, max_items=-1,
        ),
    }
    contact_sheet = _save_samples(
        projected_decoder,
        query_decoder,
        val_split,
        args.output_dir / "samples",
        device,
        count=args.sample_items,
    )
    summary = {
        "status": "completed",
        "step": step,
        "best_steps": {
            "projected": int(best_projected_payload["step"]),
            "query_hidden": int(best_query_payload["step"]),
        },
        "best_losses": best_loss,
        "last_train": last_train,
        "last_eval": last_eval,
        "final_full_val": final_eval,
        "best_checkpoints": {
            "projected": str(best_projected_checkpoint),
            "query_hidden": str(best_query_checkpoint),
        },
        "final_checkpoint": str(final_checkpoint),
        "contact_sheet": str(contact_sheet),
        "elapsed_sec": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.log(
            {
                "projected/final_correct": final_eval["projected"]["correct_loss"],
                "projected/final_wrong": final_eval["projected"]["wrong_condition_loss"],
                "projected/final_ratio": final_eval["projected"]["wrong_over_correct"],
                "query/final_correct": final_eval["query_hidden"]["correct_loss"],
                "query/final_wrong": final_eval["query_hidden"]["wrong_condition_loss"],
                "query/final_ratio": final_eval["query_hidden"]["wrong_over_correct"],
                "comparison/contact_sheet": __import__("wandb").Image(str(contact_sheet)),
            },
            step=step,
        )
        wandb_run.finish()
    print(json.dumps(summary), flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare projected and preprojection k-query reconstruction")
    parser.add_argument("--projected-cache-dir", type=Path, required=True)
    parser.add_argument("--query-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-cache-split", default="val")
    parser.add_argument("--query-token-count", type=int, default=8)
    parser.add_argument("--max-train-items", type=int, default=-1)
    parser.add_argument("--max-val-items", type=int, default=-1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=18560)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-max-items", type=int, default=1024)
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
    return train_query_state_ablation(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
