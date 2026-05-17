"""Joint fine-tune EB-Nav WM dynamics and semantic value head."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.train_eb_nav_value_head_predicted import (  # noqa: E402
    NUM_PATCHES,
    QWEN_VISUAL_DIM,
    SemanticWMValueHead,
    build_visual_encoder,
    encode_many,
    pearson_r,
    resolve_repo_path,
)
from src.train.train_wm_joint import CustomJointSequenceDataset, _joint_collate_fn  # noqa: E402
from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402
from src.wm.predictor.lewm import LeWMWorldModel  # noqa: E402


@dataclass
class EncodedWMBatch:
    z_history: torch.Tensor
    action_history: torch.Tensor
    z_future: torch.Tensor
    gt_action_future: torch.Tensor
    rewards: torch.Tensor | None = None


@dataclass
class EncodedValueBatch:
    semantic: torch.Tensor
    z_current: torch.Tensor
    z_next_pred: torch.Tensor
    action: torch.Tensor
    value: torch.Tensor


def build_trainable_wm_from_checkpoint(
    path: Path,
    device: torch.device,
    visual_dim: int = QWEN_VISUAL_DIM,
    latent_dim: int | None = None,
) -> LeWMWorldModel:
    ckpt = torch.load(path, map_location="cpu")
    if latent_dim is None:
        latent_dim = NUM_PATCHES * int(visual_dim)
    wm = LeWMWorldModel(
        latent_dim=int(latent_dim),
        action_dim=8,
        hidden_dim=512,
        history_len=4,
        num_patches=NUM_PATCHES,
        token_dim=int(visual_dim),
        num_layers=6,
        num_heads=16,
        dim_head=64,
        mlp_ratio=4.0,
        dropout=0.1,
        emb_dropout=0.0,
        sigreg_enabled=False,
        sigreg_latent_dim=int(visual_dim),
        reward_enabled=False,
        image_decoder_enabled=False,
        ensemble_size=1,
        predict_delta=True,
        delta_scale=1.0,
        zero_init_delta_head=True,
    ).to(device)
    state = ckpt.get("wm_state", ckpt)
    current = wm.state_dict()
    filtered = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
    skipped_shape = [k for k, v in state.items() if k in current and tuple(v.shape) != tuple(current[k].shape)]
    missing, unexpected = wm.load_state_dict(filtered, strict=False)
    if skipped_shape:
        print(f"skipped_wm_shape_keys={skipped_shape[:8]}", flush=True)
    if unexpected:
        print(f"unexpected_wm_keys={unexpected[:8]}", flush=True)
    if missing:
        print(f"missing_wm_keys={missing[:8]}", flush=True)
    wm.train()
    for param in wm.parameters():
        param.requires_grad = True
    return wm


def freeze_qwen(adapter: QwenVLMAdapter) -> None:
    adapter._ensure_model()
    if adapter._model is None:
        raise RuntimeError(f"Qwen init failed: {adapter.init_error}")
    adapter._model.eval()
    for param in adapter._model.parameters():
        param.requires_grad = False


def make_subset(manifest: str, images_base_dir: str, *, max_samples: int) -> Subset:
    dataset = CustomJointSequenceDataset(
        manifest,
        images_base_dir,
        history_len=4,
        temporal_stride=3,
        action_dim=8,
        max_samples=0,
        require_prompt=False,
    )
    limit = int(max_samples)
    indices = list(range(len(dataset)))[: limit or len(dataset)]
    return Subset(dataset, indices)


def encode_wm_batch(raw: dict[str, Any], visual_encoder: Any, device: torch.device, visual_dim: int) -> EncodedWMBatch:
    history_images: list[list[str]] = raw["history_images"]
    future_images: list[list[str]] = raw["future_images"]
    batch_size = len(history_images)
    history_len = len(history_images[0])
    future_len = len(future_images[0])
    flat_hist = [path for seq in history_images for path in seq]
    flat_future = [path for seq in future_images for path in seq]
    z_history = encode_many(visual_encoder, flat_hist, None, device).reshape(batch_size, history_len, NUM_PATCHES, visual_dim)
    z_future = encode_many(visual_encoder, flat_future, None, device).reshape(batch_size, future_len, NUM_PATCHES, visual_dim)
    rewards = raw.get("future_rewards")
    return EncodedWMBatch(
        z_history=z_history,
        action_history=raw["history_actions"].float().to(device),
        z_future=z_future,
        gt_action_future=raw["future_actions"].float().to(device),
        rewards=None if rewards is None else rewards.float().to(device),
    )


def wm_rollout_loss(
    wm: LeWMWorldModel,
    batch: EncodedWMBatch,
    *,
    free_run_start: int,
    detach_rollout: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_z = batch.z_history
    teacher_action = batch.action_history.clone()
    losses: list[torch.Tensor] = []
    mse_values: list[torch.Tensor] = []
    copy_values: list[torch.Tensor] = []
    delta_cos_values: list[torch.Tensor] = []
    horizon = int(batch.z_future.size(1))
    for step_idx in range(horizon):
        teacher_action[:, -1, :] = batch.gt_action_future[:, step_idx, :]
        pred_z = wm.predict_next(teacher_z, teacher_action)
        target_z = batch.z_future[:, step_idx, :, :].detach()
        losses.append(F.mse_loss(pred_z, target_z))
        last_z = teacher_z[:, -1, :, :]
        mse_values.append(F.mse_loss(pred_z, target_z).detach())
        copy_values.append(F.mse_loss(last_z, target_z).detach())
        pred_delta = pred_z - last_z
        target_delta = target_z - last_z
        delta_cos_values.append(F.cosine_similarity(pred_delta.flatten(1), target_delta.flatten(1), dim=1).mean().detach())
        use_free_running = horizon > 1 and (step_idx + 1) >= int(free_run_start)
        next_teacher = pred_z if use_free_running else target_z
        if use_free_running and detach_rollout:
            next_teacher = next_teacher.detach()
        teacher_z = torch.cat([teacher_z[:, 1:, ...], next_teacher.unsqueeze(1)], dim=1)
        if step_idx < horizon - 1:
            teacher_action = torch.cat(
                [teacher_action[:, 1:, :], batch.gt_action_future[:, step_idx, :].unsqueeze(1)],
                dim=1,
            )
    wm_mse = torch.stack(mse_values).mean()
    copy_mse = torch.stack(copy_values).mean()
    return torch.stack(losses).mean(), {
        "wm_mse": float(wm_mse.item()),
        "copy_mse": float(copy_mse.item()),
        "wm_margin": float((wm_mse - copy_mse).item()),
        "delta_cos": float(torch.stack(delta_cos_values).mean().item()),
    }


def encode_value_batch(
    raw: dict[str, Any],
    *,
    visual_encoder: Any,
    semantic_encoder: QwenLLMLatentEncoder,
    wm: LeWMWorldModel,
    device: torch.device,
    visual_dim: int,
    gamma: float,
) -> EncodedValueBatch:
    history_images: list[list[str]] = raw["history_images"]
    future_images: list[list[str]] = raw["future_images"]
    batch_size = len(history_images)
    prompts = [str(x or "") for x in raw.get("prompts", raw.get("instructions", [""] * batch_size))]
    hist_actions = raw["history_actions"].float().to(device)
    fut_actions = raw["future_actions"].float().to(device)
    rewards = raw.get("future_rewards")
    if rewards is None:
        raise RuntimeError("future_rewards are required for value training")
    rewards_t = rewards.float().to(device)
    discounts = torch.tensor([float(gamma) ** i for i in range(rewards_t.size(1))], dtype=torch.float32, device=device)
    value = (rewards_t * discounts.unsqueeze(0)).sum(dim=1)

    flat_hist = [path for seq in history_images for path in seq]
    z_hist = encode_many(visual_encoder, flat_hist, None, device).reshape(batch_size, len(history_images[0]), NUM_PATCHES, visual_dim)
    semantic = encode_many(
        semantic_encoder,
        [seq[-1] for seq in history_images],
        prompts,
        device,
        expected_flat_dim=3584,
    ).reshape(batch_size, 3584)
    teacher_action = hist_actions.clone()
    teacher_action[:, -1, :] = fut_actions[:, 0, :]
    z_next_pred = wm.predict_next(z_hist, teacher_action)
    return EncodedValueBatch(
        semantic=semantic,
        z_current=z_hist[:, -1],
        z_next_pred=z_next_pred,
        action=fut_actions[:, 0, :],
        value=value,
    )


def eval_value(
    *,
    loader: DataLoader,
    visual_encoder: Any,
    semantic_encoder: QwenLLMLatentEncoder,
    wm: LeWMWorldModel,
    head: SemanticWMValueHead,
    device: torch.device,
    visual_dim: int,
    gamma: float,
) -> dict[str, float | None]:
    wm.eval(); head.eval()
    preds: list[float] = []
    targets: list[float] = []
    loss_sum = 0.0
    mse_sum = 0.0
    n = 0
    with torch.no_grad():
        for raw in loader:
            batch = encode_value_batch(raw, visual_encoder=visual_encoder, semantic_encoder=semantic_encoder, wm=wm, device=device, visual_dim=visual_dim, gamma=gamma)
            pred = head(batch.semantic, batch.z_current, batch.z_next_pred, batch.action)
            loss = F.smooth_l1_loss(pred, batch.value)
            mse = F.mse_loss(pred, batch.value)
            count = int(pred.numel())
            loss_sum += float(loss.item()) * count
            mse_sum += float(mse.item()) * count
            n += count
            preds.extend(float(x) for x in pred.detach().cpu())
            targets.extend(float(x) for x in batch.value.detach().cpu())
    wm.train(); head.train()
    return {
        "value_loss": loss_sum / max(1, n),
        "value_mse": mse_sum / max(1, n),
        "value_pearson": pearson_r(preds, targets),
        "value_n": float(n),
    }


def eval_wm(
    *,
    loader: DataLoader,
    visual_encoder: Any,
    wm: LeWMWorldModel,
    device: torch.device,
    visual_dim: int,
    free_run_start: int,
    detach_rollout: bool,
) -> dict[str, float]:
    wm.eval()
    sums = {"wm_loss": 0.0, "wm_mse": 0.0, "copy_mse": 0.0, "wm_margin": 0.0, "delta_cos": 0.0}
    n = 0
    with torch.no_grad():
        for raw in loader:
            batch = encode_wm_batch(raw, visual_encoder, device, visual_dim)
            loss, metrics = wm_rollout_loss(wm, batch, free_run_start=free_run_start, detach_rollout=detach_rollout)
            count = int(batch.z_history.size(0))
            sums["wm_loss"] += float(loss.item()) * count
            for key, value in metrics.items():
                sums[key] += float(value) * count
            n += count
    wm.train()
    return {key: value / max(1, n) for key, value in sums.items()} | {"wm_n": float(n)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wm-train-manifest", required=True)
    p.add_argument("--value-train-manifest", required=True)
    p.add_argument("--value-test-manifest", required=True)
    p.add_argument("--wm-original-test-manifest", default="")
    p.add_argument("--wm-rollout-test-manifest", default="")
    p.add_argument("--images-base-dir", default=".")
    p.add_argument("--wm-checkpoint", required=True)
    p.add_argument("--init-value-head", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--max-wm-samples", type=int, default=4096)
    p.add_argument("--max-value-samples", type=int, default=4096)
    p.add_argument("--test-max-samples", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--steps-per-epoch", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--value-weight", type=float, default=0.05)
    p.add_argument("--wm-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--free-run-start", type=int, default=1)
    p.add_argument("--detach-rollout", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every-epoch", action="store_true")
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-dtype", default="auto")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--visual-encoder", choices=["qwen", "dino"], default="qwen")
    p.add_argument("--dino-model-name", default="dinov2_vits14")
    p.add_argument("--dino-image-size", type=int, default=224)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if str(args.cuda_device) in {"", "-1", "cpu"} else f"cuda:{args.cuda_device}")
    out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "dev" / f"{datetime.now():%Y%m%d_%H%M%S}_joint_wm_value"
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    wm_train_ds = make_subset(args.wm_train_manifest, args.images_base_dir, max_samples=args.max_wm_samples)
    value_train_ds = make_subset(args.value_train_manifest, args.images_base_dir, max_samples=args.max_value_samples)
    value_test_ds = make_subset(args.value_test_manifest, args.images_base_dir, max_samples=args.test_max_samples)
    wm_original_test_ds = make_subset(args.wm_original_test_manifest, args.images_base_dir, max_samples=args.test_max_samples) if args.wm_original_test_manifest else None
    wm_rollout_test_ds = make_subset(args.wm_rollout_test_manifest, args.images_base_dir, max_samples=args.test_max_samples) if args.wm_rollout_test_manifest else None

    adapter = QwenVLMAdapter(
        model_name=args.model_name,
        latent_dim=NUM_PATCHES * QWEN_VISUAL_DIM,
        enabled=True,
        fallback_enabled=False,
        device_map=None if str(args.device_map).lower() in {"", "none"} else args.device_map,
        model_dtype=args.model_dtype,
    )
    freeze_qwen(adapter)
    visual_encoder, visual_dim, visual_latent_dim = build_visual_encoder(args, adapter)
    semantic_encoder = QwenLLMLatentEncoder(QWEN_VISUAL_DIM, name="qwen_llm_semantic", model_name=args.model_name, qwen_adapter=adapter, use_vision_only=False, visual_pooling="last", cache_latents=True)

    wm = build_trainable_wm_from_checkpoint(resolve_repo_path(args.wm_checkpoint), device, visual_dim=visual_dim, latent_dim=visual_latent_dim)
    head = SemanticWMValueHead(semantic_dim=QWEN_VISUAL_DIM, visual_dim=visual_dim, action_dim=8, hidden=int(args.hidden)).to(device)
    if args.init_value_head:
        state = torch.load(resolve_repo_path(args.init_value_head), map_location="cpu")
        head.load_state_dict(state.get("head_state", state), strict=True)

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in wm.parameters() if p.requires_grad], "lr": float(args.wm_lr)},
            {"params": [p for p in head.parameters() if p.requires_grad], "lr": float(args.head_lr)},
        ],
        weight_decay=float(args.weight_decay),
    )

    wm_loader = DataLoader(wm_train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=_joint_collate_fn)
    value_loader = DataLoader(value_train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=_joint_collate_fn)
    value_test_loader = DataLoader(value_test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_joint_collate_fn)
    wm_original_test_loader = DataLoader(wm_original_test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_joint_collate_fn) if wm_original_test_ds is not None else None
    wm_rollout_test_loader = DataLoader(wm_rollout_test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_joint_collate_fn) if wm_rollout_test_ds is not None else None

    log_fields = [
        "epoch", "train_total", "train_wm_loss", "train_value_loss", "train_value_mse", "train_value_pearson",
        "test_value_loss", "test_value_mse", "test_value_pearson",
        "orig_wm_mse", "orig_copy_mse", "orig_wm_margin", "orig_delta_cos",
        "rollout_wm_mse", "rollout_copy_mse", "rollout_wm_margin", "rollout_delta_cos",
    ]
    best = {"test_value_mse": float("inf"), "epoch": -1}
    with (out / "train_log.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        wm_iter = cycle(wm_loader)
        value_iter = cycle(value_loader)
        for epoch in range(1, int(args.epochs) + 1):
            wm.train(); head.train()
            total_sum = wm_sum = value_sum = value_mse_sum = 0.0
            preds: list[float] = []
            targets: list[float] = []
            train_n = 0
            for _ in range(int(args.steps_per_epoch)):
                wm_raw = next(wm_iter)
                value_raw = next(value_iter)
                wm_batch = encode_wm_batch(wm_raw, visual_encoder, device, visual_dim)
                wm_loss, _ = wm_rollout_loss(wm, wm_batch, free_run_start=int(args.free_run_start), detach_rollout=bool(args.detach_rollout))
                value_batch = encode_value_batch(value_raw, visual_encoder=visual_encoder, semantic_encoder=semantic_encoder, wm=wm, device=device, visual_dim=visual_dim, gamma=float(args.gamma))
                value_pred = head(value_batch.semantic, value_batch.z_current, value_batch.z_next_pred, value_batch.action)
                value_loss = F.smooth_l1_loss(value_pred, value_batch.value)
                value_mse = F.mse_loss(value_pred, value_batch.value)
                total_loss = wm_loss + float(args.value_weight) * value_loss
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(wm.parameters()) + list(head.parameters()), float(args.grad_clip))
                optimizer.step()
                count = int(value_pred.numel())
                total_sum += float(total_loss.item()) * count
                wm_sum += float(wm_loss.item()) * count
                value_sum += float(value_loss.item()) * count
                value_mse_sum += float(value_mse.item()) * count
                train_n += count
                preds.extend(float(x) for x in value_pred.detach().cpu())
                targets.extend(float(x) for x in value_batch.value.detach().cpu())

            value_eval = eval_value(loader=value_test_loader, visual_encoder=visual_encoder, semantic_encoder=semantic_encoder, wm=wm, head=head, device=device, visual_dim=visual_dim, gamma=float(args.gamma))
            orig_eval = eval_wm(loader=wm_original_test_loader, visual_encoder=visual_encoder, wm=wm, device=device, visual_dim=visual_dim, free_run_start=int(args.free_run_start), detach_rollout=bool(args.detach_rollout)) if wm_original_test_loader is not None else {}
            rollout_eval = eval_wm(loader=wm_rollout_test_loader, visual_encoder=visual_encoder, wm=wm, device=device, visual_dim=visual_dim, free_run_start=int(args.free_run_start), detach_rollout=bool(args.detach_rollout)) if wm_rollout_test_loader is not None else {}
            row = {
                "epoch": epoch,
                "train_total": total_sum / max(1, train_n),
                "train_wm_loss": wm_sum / max(1, train_n),
                "train_value_loss": value_sum / max(1, train_n),
                "train_value_mse": value_mse_sum / max(1, train_n),
                "train_value_pearson": pearson_r(preds, targets),
                "test_value_loss": value_eval["value_loss"],
                "test_value_mse": value_eval["value_mse"],
                "test_value_pearson": value_eval["value_pearson"],
                "orig_wm_mse": orig_eval.get("wm_mse"),
                "orig_copy_mse": orig_eval.get("copy_mse"),
                "orig_wm_margin": orig_eval.get("wm_margin"),
                "orig_delta_cos": orig_eval.get("delta_cos"),
                "rollout_wm_mse": rollout_eval.get("wm_mse"),
                "rollout_copy_mse": rollout_eval.get("copy_mse"),
                "rollout_wm_margin": rollout_eval.get("wm_margin"),
                "rollout_delta_cos": rollout_eval.get("delta_cos"),
            }
            writer.writerow(row); f.flush(); print(json.dumps(row), flush=True)
            if float(row["test_value_mse"]) < float(best["test_value_mse"]):
                best = {"test_value_mse": float(row["test_value_mse"]), "epoch": epoch}
                torch.save({"wm_state": wm.state_dict(), "head_state": head.state_dict(), "args": vars(args), "best": best, "visual_dim": visual_dim, "visual_encoder": args.visual_encoder}, out / "best_joint_wm_value.pt")

    torch.save({"wm_state": wm.state_dict(), "head_state": head.state_dict(), "args": vars(args), "best": best, "visual_dim": visual_dim, "visual_encoder": args.visual_encoder}, out / "final_joint_wm_value.pt")
    (out / "summary.json").write_text(json.dumps({"best": best, "args": vars(args), "visual_dim": visual_dim, "visual_encoder": args.visual_encoder}, indent=2))


if __name__ == "__main__":
    main()
