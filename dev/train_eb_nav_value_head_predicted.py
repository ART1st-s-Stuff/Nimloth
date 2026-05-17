"""Train semantic-conditioned EB-Nav value head on GT or WM-predicted visual latents."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.train_wm_joint import CustomJointSequenceDataset, _joint_collate_fn  # noqa: E402
from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402
from src.wm.encoder.dino import DinoV2MiniEncoder  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402
from src.wm.predictor.lewm import LeWMWorldModel  # noqa: E402


NUM_PATCHES = 16
QWEN_VISUAL_DIM = 3584
DINO_VISUAL_DIM = 384


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return None if den_x == 0.0 or den_y == 0.0 else num / (den_x * den_y)


@dataclass
class ValueBatch:
    semantic: torch.Tensor
    z_current: torch.Tensor
    z_next: torch.Tensor
    action: torch.Tensor
    value: torch.Tensor


class SemanticWMValueHead(nn.Module):
    def __init__(self, *, semantic_dim: int, visual_dim: int, action_dim: int, hidden: int = 512) -> None:
        super().__init__()
        self.semantic_proj = nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, hidden), nn.GELU())
        self.visual_proj = nn.Sequential(nn.LayerNorm(visual_dim * 3), nn.Linear(visual_dim * 3, hidden), nn.GELU())
        self.action_proj = nn.Sequential(nn.Linear(action_dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.head = nn.Sequential(
            nn.LayerNorm(hidden + hidden + 64),
            nn.Linear(hidden + hidden + 64, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, semantic: torch.Tensor, z_current: torch.Tensor, z_next: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        cur = z_current.mean(dim=1)
        nxt = z_next.mean(dim=1)
        delta = nxt - cur
        sem = self.semantic_proj(semantic)
        vis = self.visual_proj(torch.cat([cur, nxt, delta], dim=-1))
        act = self.action_proj(action)
        return self.head(torch.cat([sem, vis, act], dim=-1)).squeeze(-1)


def _infer_wm_ensemble_size(state: dict[str, torch.Tensor]) -> int:
    max_idx = -1
    for key in state.keys():
        if key.startswith("wm_core.ensemble_transformers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                max_idx = max(max_idx, int(parts[2]))
        elif key.startswith("ensemble_transformers."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
    return max(1, max_idx + 2)


def build_wm_from_checkpoint(
    path: Path,
    device: torch.device,
    *,
    visual_dim: int = QWEN_VISUAL_DIM,
    latent_dim: int | None = None,
) -> LeWMWorldModel:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("wm_state", ckpt)
    if latent_dim is None:
        latent_dim = NUM_PATCHES * int(visual_dim)
    ensemble_size = _infer_wm_ensemble_size(state)
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
        ensemble_size=ensemble_size,
        predict_delta=True,
        delta_scale=1.0,
        zero_init_delta_head=True,
    ).to(device)
    missing, unexpected = wm.load_state_dict(state, strict=False)
    if unexpected:
        print(f"unexpected_wm_keys={unexpected[:8]}")
    if missing:
        print(f"missing_wm_keys={missing[:8]}")
    print(f"loaded_wm_ensemble_size={ensemble_size}", flush=True)
    wm.eval()
    for param in wm.parameters():
        param.requires_grad = False
    return wm


def freeze_qwen(adapter: QwenVLMAdapter) -> None:
    adapter._ensure_model()
    if adapter._model is None:
        raise RuntimeError(f"Qwen init failed: {adapter.init_error}")
    adapter._model.eval()
    for param in adapter._model.parameters():
        param.requires_grad = False


@torch.no_grad()
def encode_many(
    encoder: Any,
    paths: list[str],
    prompts: list[str] | None,
    device: torch.device,
    *,
    expected_flat_dim: int | None = None,
) -> torch.Tensor:
    out = []
    prompt_list: list[str | None] = list(prompts) if prompts is not None else ([None] * len(paths))
    for path, prompt in zip(paths, prompt_list):
        if hasattr(encoder, "encode_image_path_with_prompt"):
            z = encoder.encode_image_path_with_prompt(path, prompt_override=prompt).z.detach().float()
        else:
            z = encoder.encode_image_path(path).z.detach().float()
        if expected_flat_dim is not None:
            z = z.reshape(-1)
            if z.numel() < expected_flat_dim:
                raise RuntimeError(f"latent for {path} has {z.numel()} values, expected at least {expected_flat_dim}")
            z = z[:expected_flat_dim]
        elif z.dim() == 1:
            z = z.unsqueeze(0)
        out.append(z.to(device))
    return torch.stack(out, dim=0)


def build_visual_encoder(args: argparse.Namespace, adapter: QwenVLMAdapter | None = None) -> tuple[Any, int, int]:
    if args.visual_encoder == "dino":
        encoder = DinoV2MiniEncoder(
            latent_dim=NUM_PATCHES * DINO_VISUAL_DIM,
            freeze_backbone=True,
            image_size=int(args.dino_image_size),
            patch_size=56,
            num_patches=NUM_PATCHES,
            model_name=args.dino_model_name,
            token_strategy="patch_tokens",
        )
        return encoder, DINO_VISUAL_DIM, NUM_PATCHES * DINO_VISUAL_DIM
    if adapter is None:
        raise ValueError("Qwen visual encoder requires adapter")
    encoder = QwenLLMLatentEncoder(
        NUM_PATCHES * QWEN_VISUAL_DIM,
        name="qwen_visual_tokens",
        model_name=args.model_name,
        qwen_adapter=adapter,
        use_vision_only=True,
        visual_pooling="tokens",
        visual_num_tokens=NUM_PATCHES,
        cache_latents=True,
    )
    return encoder, QWEN_VISUAL_DIM, NUM_PATCHES * QWEN_VISUAL_DIM


def make_value_collate(
    *,
    visual_encoder: Any,
    semantic_encoder: QwenLLMLatentEncoder,
    wm: LeWMWorldModel | None,
    device: torch.device,
    visual_dim: int,
    gamma: float,
    next_latent_source: str,
    mixed_pred_prob: float,
    rng: random.Random,
) -> Any:
    def collate(items: list[dict[str, Any]]) -> ValueBatch:
        raw = cast(dict[str, Any], _joint_collate_fn(items))
        history_images: list[list[str]] = raw["history_images"]  # type: ignore[assignment]
        future_images: list[list[str]] = raw["future_images"]  # type: ignore[assignment]
        prompts = [str(x or "") for x in raw.get("prompts", raw.get("instructions", [""] * len(items)))]
        hist_actions = cast(torch.Tensor, raw["history_actions"]).float().to(device)
        fut_actions = cast(torch.Tensor, raw["future_actions"]).float().to(device)
        rewards = cast(torch.Tensor | None, raw.get("future_rewards"))
        if rewards is None:
            raise RuntimeError("future_rewards are required for value training")
        rewards_t = rewards.float().to(device)
        discounts = torch.tensor([gamma ** i for i in range(rewards_t.size(1))], dtype=torch.float32, device=device)
        value = (rewards_t * discounts.unsqueeze(0)).sum(dim=1)

        flat_hist = [path for seq in history_images for path in seq]
        z_hist = encode_many(visual_encoder, flat_hist, None, device).reshape(len(items), len(history_images[0]), NUM_PATCHES, visual_dim)
        z_current = z_hist[:, -1]
        z_gt_next = encode_many(visual_encoder, [seq[0] for seq in future_images], None, device).reshape(len(items), NUM_PATCHES, visual_dim)
        semantic = encode_many(
            semantic_encoder,
            [seq[-1] for seq in history_images],
            prompts,
            device,
            expected_flat_dim=3584,
        ).reshape(len(items), 3584)

        source = next_latent_source
        if source == "mixed":
            source = "pred" if rng.random() < mixed_pred_prob else "gt"
        if source == "pred":
            if wm is None:
                raise RuntimeError("--wm-checkpoint is required for predicted next latents")
            teacher_action = hist_actions.clone()
            teacher_action[:, -1, :] = fut_actions[:, 0, :]
            z_next = wm.predict_next(z_hist, teacher_action).detach()
        elif source == "gt":
            z_next = z_gt_next
        else:
            raise ValueError(f"unknown next_latent_source={next_latent_source}")
        return ValueBatch(
            semantic=semantic,
            z_current=z_current,
            z_next=z_next,
            action=fut_actions[:, 0, :],
            value=value,
        )
    return collate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--images-base-dir", default=".")
    p.add_argument("--wm-checkpoint", default="")
    p.add_argument("--next-latent-source", choices=["pred", "gt", "mixed"], default="pred")
    p.add_argument("--mixed-pred-prob", type=float, default=0.8)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--max-samples", type=int, default=8192)
    p.add_argument("--test-max-samples", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--output-dir", default="")
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
    rng = random.Random(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if str(args.cuda_device) in {"", "-1", "cpu"} else f"cuda:{args.cuda_device}")
    out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "dev" / f"{datetime.now():%Y%m%d_%H%M%S}_eb_nav_value_head_pred"
    out.mkdir(parents=True, exist_ok=True)

    train_ds_full = CustomJointSequenceDataset(args.train_manifest, args.images_base_dir, history_len=4, temporal_stride=3, action_dim=8, max_samples=0, require_prompt=False)
    test_ds_full = CustomJointSequenceDataset(args.test_manifest, args.images_base_dir, history_len=4, temporal_stride=3, action_dim=8, max_samples=0, require_prompt=False)
    train_idx = list(range(len(train_ds_full)))[: max(0, int(args.max_samples)) or len(train_ds_full)]
    test_idx = list(range(len(test_ds_full)))[: max(0, int(args.test_max_samples)) or len(test_ds_full)]
    train_ds = Subset(train_ds_full, train_idx)
    test_ds = Subset(test_ds_full, test_idx)

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

    wm = None
    if args.next_latent_source in {"pred", "mixed"}:
        if not args.wm_checkpoint:
            raise ValueError("--wm-checkpoint is required for pred/mixed next latent source")
        wm = build_wm_from_checkpoint(resolve_repo_path(args.wm_checkpoint), device=device, visual_dim=visual_dim, latent_dim=visual_latent_dim)

    collate = make_value_collate(visual_encoder=visual_encoder, semantic_encoder=semantic_encoder, wm=wm, device=device, visual_dim=visual_dim, gamma=float(args.gamma), next_latent_source=args.next_latent_source, mixed_pred_prob=float(args.mixed_pred_prob), rng=rng)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate)

    head = SemanticWMValueHead(semantic_dim=QWEN_VISUAL_DIM, visual_dim=visual_dim, action_dim=8, hidden=int(args.hidden)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.SmoothL1Loss()
    log_path = out / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "test_loss", "test_mse", "test_pearson", "num_train", "num_test"])
        writer.writeheader()
        best = {"test_mse": float("inf"), "epoch": -1}
        for epoch in range(1, int(args.epochs) + 1):
            head.train(); train_sum = 0.0; train_n = 0
            for batch in train_loader:
                pred = head(batch.semantic, batch.z_current, batch.z_next, batch.action)
                loss = loss_fn(pred, batch.value)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                train_sum += float(loss.item()) * int(pred.numel()); train_n += int(pred.numel())
            head.eval(); preds=[]; targets=[]; test_loss=0.0; test_mse=0.0; test_n=0
            with torch.no_grad():
                for batch in test_loader:
                    pred = head(batch.semantic, batch.z_current, batch.z_next, batch.action)
                    loss = loss_fn(pred, batch.value)
                    mse = nn.functional.mse_loss(pred, batch.value)
                    test_loss += float(loss.item()) * int(pred.numel())
                    test_mse += float(mse.item()) * int(pred.numel())
                    test_n += int(pred.numel())
                    preds.extend(float(x) for x in pred.detach().cpu())
                    targets.extend(float(x) for x in batch.value.detach().cpu())
            row = {
                "epoch": epoch,
                "train_loss": train_sum / max(1, train_n),
                "test_loss": test_loss / max(1, test_n),
                "test_mse": test_mse / max(1, test_n),
                "test_pearson": pearson_r(preds, targets),
                "num_train": train_n,
                "num_test": test_n,
            }
            writer.writerow(row); f.flush(); print(json.dumps(row), flush=True)
            if row["test_mse"] < best["test_mse"]:
                best = {"test_mse": float(row["test_mse"]), "epoch": epoch}
                torch.save({"head_state": head.state_dict(), "args": vars(args), "best": best, "visual_dim": visual_dim, "visual_encoder": args.visual_encoder}, out / "best_value_head.pt")
    (out / "summary.json").write_text(json.dumps({"best": best, "args": vars(args), "visual_dim": visual_dim, "visual_encoder": args.visual_encoder}, indent=2))


if __name__ == "__main__":
    main()
