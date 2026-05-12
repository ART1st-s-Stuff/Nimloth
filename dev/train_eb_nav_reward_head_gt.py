"""Head-only GT transition reward training for EB-Nav (frozen Qwen vision, token latents)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.eb_nav_dataset import EBNavSequenceDataset, encode_eb_nav_action  # noqa: E402
from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def binary_sign_accuracy(preds: list[float], targets: list[float]) -> float | None:
    if not preds or len(preds) != len(targets):
        return None
    ok = 0
    for p, t in zip(preds, targets):
        sp = 1 if p >= 0.0 else -1
        st = 1 if t >= 0.0 else -1
        if sp == st:
            ok += 1
    return ok / len(preds)


_SEMANTIC_TABLE: torch.Tensor | None = None


def semantic_bow_embedding(text: str, *, dim: int, num_buckets: int, seed: int) -> torch.Tensor:
    """Deterministic bag-of-words embedding without loading a text encoder."""
    global _SEMANTIC_TABLE
    if _SEMANTIC_TABLE is None or _SEMANTIC_TABLE.shape != (num_buckets, dim):
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        _SEMANTIC_TABLE = torch.randn(num_buckets, dim, generator=gen, dtype=torch.float32)
    words = str(text or "").lower().replace("\n", " ").split()
    if not words:
        return torch.zeros(dim, dtype=torch.float32)
    acc = torch.zeros(dim, dtype=torch.float32)
    for w in words:
        if not w:
            continue
        digest = hashlib.sha1(w.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % int(num_buckets)
        acc += _SEMANTIC_TABLE[idx]
    return acc / max(1.0, math.sqrt(float(len(words))))


@dataclass
class BatchPack:
    z_t: torch.Tensor
    z_next: torch.Tensor
    delta: torch.Tensor
    action_oh: torch.Tensor
    sem: torch.Tensor
    reward: torch.Tensor


class TransitionRewardHead(nn.Module):
    """Pooled visual tokens + action MLP + semantic BoW -> scalar reward."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_tokens: int,
        action_dim: int,
        semantic_dim: int,
        action_hidden: int = 64,
        mlp_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        pooled = 3 * self.token_dim
        self.action_mlp = nn.Sequential(
            nn.Linear(int(action_dim), int(action_hidden)),
            nn.GELU(),
            nn.Linear(int(action_hidden), int(action_hidden)),
        )
        in_dim = pooled + int(action_hidden) + int(semantic_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(mlp_hidden)),
            nn.GELU(),
            nn.Linear(int(mlp_hidden), int(mlp_hidden)),
            nn.GELU(),
            nn.Linear(int(mlp_hidden), 1),
        )

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor, delta: torch.Tensor, action_oh: torch.Tensor, sem: torch.Tensor) -> torch.Tensor:
        # z_* : [B, P, D]
        def pool(x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=1)

        v_t = pool(z_t)
        v_next = pool(z_next)
        v_delta = pool(delta)
        v_cat = torch.cat([v_t, v_next, v_delta], dim=-1)
        a = self.action_mlp(action_oh)
        h = torch.cat([v_cat, a, sem], dim=-1)
        return self.head(h).squeeze(-1)


def freeze_qwen_visual_model(adapter: QwenVLMAdapter) -> None:
    adapter._ensure_model()
    model = adapter._model
    if model is None:
        raise RuntimeError(f"Qwen init failed: {adapter.init_error}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


@torch.no_grad()
def encode_visual_tokens(
    encoder: QwenLLMLatentEncoder,
    path: str,
    *,
    device: torch.device,
    extra_cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    if path in extra_cache:
        return extra_cache[path]
    out = encoder.encode_image_path_with_prompt(image_path=path, prompt_override=None)
    z = out.z
    if z.dim() == 1:
        z = z.unsqueeze(0)
    z = z.detach().float().to(device=device, non_blocking=True)
    extra_cache[path] = z
    return z


def build_indices(dataset_len: int, max_samples: int | None, rng: random.Random) -> list[int]:
    idx = list(range(dataset_len))
    rng.shuffle(idx)
    if max_samples is not None and max_samples > 0:
        idx = idx[: int(max_samples)]
    return idx


def collate_builder(
    *,
    encoder: QwenLLMLatentEncoder,
    action_dim: int,
    semantic_dim: int,
    semantic_buckets: int,
    seed: int,
    device: torch.device,
    latent_cache: dict[str, torch.Tensor],
) -> Any:
    def collate(batch: list[dict[str, Any]]) -> BatchPack:
        src_paths: list[str] = []
        tgt_paths: list[str] = []
        action_ids: list[int] = []
        rewards: list[float] = []
        texts: list[str] = []
        for item in batch:
            hist = item["history_images"]
            fut = item["future_images"]
            fa_ids = item.get("future_action_ids", [])
            fr = item.get("future_rewards", [])
            if not hist or not fut or not fa_ids or not fr:
                continue
            src_paths.append(str(hist[-1]))
            tgt_paths.append(str(fut[0]))
            action_ids.append(int(fa_ids[0]))
            rewards.append(float(fr[0]))
            instr = str(item.get("instruction", ""))
            prompt = str(item.get("prompt", ""))
            texts.append(f"{instr}\n{prompt}")

        if not src_paths:
            raise RuntimeError("Empty batch after filtering; check dataset paths.")

        z_t_list: list[torch.Tensor] = []
        z_n_list: list[torch.Tensor] = []
        for sp, tp in zip(src_paths, tgt_paths):
            z_t_list.append(encode_visual_tokens(encoder, sp, device=device, extra_cache=latent_cache))
            z_n_list.append(encode_visual_tokens(encoder, tp, device=device, extra_cache=latent_cache))

        z_t = torch.stack(z_t_list, dim=0)
        z_next = torch.stack(z_n_list, dim=0)
        delta = z_next - z_t

        act_rows = [encode_eb_nav_action(aid, action_dim) for aid in action_ids]
        action_oh = torch.tensor(act_rows, dtype=torch.float32, device=device)
        sem = torch.stack(
            [semantic_bow_embedding(t, dim=semantic_dim, num_buckets=semantic_buckets, seed=seed).to(device) for t in texts],
            dim=0,
        )
        reward = torch.tensor(rewards, dtype=torch.float32, device=device)
        return BatchPack(z_t=z_t, z_next=z_next, delta=delta, action_oh=action_oh, sem=sem, reward=reward)

    return collate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train semantic-conditioned transition reward head on frozen Qwen visual tokens.")
    p.add_argument("--dataset-path", default="datasets/EB-Nav/eb-nav_dataset_single_step.json")
    p.add_argument("--images-base-dir", default="datasets/EB-Nav")
    p.add_argument("--reward-cache-path", default="datasets/EB-Nav/phase2_reward_cache.jsonl")
    p.add_argument("--max-samples", type=int, default=0, help="Max training samples (0 = all after shuffle).")
    p.add_argument("--test-max-samples", type=int, default=0, help="Max test samples (0 = infer 20%% of train cap or 256).")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--output-dir", default="")
    p.add_argument("--cuda-device", default="0", help="CUDA device index or -1 for CPU.")
    p.add_argument("--visual-num-tokens", type=int, default=16)
    p.add_argument("--token-dim", type=int, default=3584)
    p.add_argument("--history-len", type=int, default=4)
    p.add_argument("--temporal-stride", type=int, default=1)
    p.add_argument("--action-dim", type=int, default=8)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-dtype", default="auto")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--semantic-dim", type=int, default=64)
    p.add_argument("--semantic-buckets", type=int, default=2048)
    p.add_argument("--mlp-hidden", type=int, default=512)
    p.add_argument("--action-hidden", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"])
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb regardless of wandb-mode.")
    p.add_argument("--use-fallback", action="store_true", help="Use adapter fallback latents (no GPU model load). For dry runs only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds_path = resolve_repo_path(args.dataset_path)
    img_base = resolve_repo_path(args.images_base_dir)
    reward_cache = resolve_repo_path(args.reward_cache_path) if args.reward_cache_path else None

    full_dataset = EBNavSequenceDataset(
        json_path=str(ds_path),
        images_base_dir=str(img_base),
        reward_cache_path=str(reward_cache) if reward_cache and reward_cache.exists() else str(reward_cache) if reward_cache else None,
        history_len=args.history_len,
        temporal_stride=args.temporal_stride,
        action_dim=args.action_dim,
    )

    all_idx = list(range(len(full_dataset)))
    rng.shuffle(all_idx)
    max_train = int(args.max_samples) if int(args.max_samples) > 0 else len(all_idx)
    train_n = min(max_train, len(all_idx))
    train_idx = all_idx[:train_n]

    if int(args.test_max_samples) > 0:
        test_n = int(args.test_max_samples)
    else:
        test_n = max(64, min(256, max(1, train_n // 5)))
    remaining = [i for i in all_idx if i not in set(train_idx)]
    rng.shuffle(remaining)
    test_idx = remaining[: min(test_n, len(remaining))]

    train_ds = Subset(full_dataset, train_idx)
    test_ds = Subset(full_dataset, test_idx)

    out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "dev" / f"{datetime.now():%Y%m%d_%H%M%S}_eb_nav_reward_head_gt"
    out.mkdir(parents=True, exist_ok=True)

    cuda = str(args.cuda_device).strip()
    if cuda in {"", "-1", "cpu"}:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cuda}" if cuda.isdigit() else cuda)

    qwen_adapter = QwenVLMAdapter(
        model_name=args.model_name,
        latent_dim=int(args.visual_num_tokens) * int(args.token_dim),
        enabled=not bool(args.use_fallback),
        fallback_enabled=bool(args.use_fallback),
        device_map=None if str(args.device_map).lower() in {"none", ""} else args.device_map,
        model_dtype=args.model_dtype,
    )
    if not args.use_fallback:
        freeze_qwen_visual_model(qwen_adapter)

    vision_encoder = QwenLLMLatentEncoder(
        latent_dim=int(args.visual_num_tokens) * int(args.token_dim),
        name="qwen_visual_frozen",
        model_name=args.model_name,
        qwen_adapter=qwen_adapter,
        use_vision_only=True,
        visual_pooling="tokens",
        visual_num_tokens=int(args.visual_num_tokens),
        cache_latents=True,
        llm_backbone_trainable=False,
    )

    latent_cache: dict[str, torch.Tensor] = {}
    collate_fn = collate_builder(
        encoder=vision_encoder,
        action_dim=args.action_dim,
        semantic_dim=args.semantic_dim,
        semantic_buckets=args.semantic_buckets,
        seed=args.seed,
        device=device,
        latent_cache=latent_cache,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_fn,
        drop_last=False,
    )

    head = TransitionRewardHead(
        token_dim=int(args.token_dim),
        num_tokens=int(args.visual_num_tokens),
        action_dim=args.action_dim,
        semantic_dim=args.semantic_dim,
        action_hidden=args.action_hidden,
        mlp_hidden=args.mlp_hidden,
    ).to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    use_wandb = (not args.no_wandb) and args.wandb_mode not in {"disabled", ""}
    wb = None
    if use_wandb:
        try:
            import wandb

            wb = wandb.init(project="eb-nav-reward-head-gt", mode=None if args.wandb_mode == "online" else args.wandb_mode)
        except Exception:
            wb = None

    log_csv = out / "train_log.csv"
    with open(log_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "test_mse", "test_sign_acc", "test_pearson", "num_train", "num_test"],
        )
        w.writeheader()

    best = {"test_mse": float("inf"), "epoch": -1}

    for epoch in range(int(args.epochs)):
        head.train()
        ep_loss = 0.0
        ep_n = 0
        for batch in train_loader:
            pred = head(batch.z_t, batch.z_next, batch.delta, batch.action_oh, batch.sem)
            loss = nn.functional.mse_loss(pred, batch.reward)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * pred.numel()
            ep_n += int(pred.numel())
        train_loss = ep_loss / max(1, ep_n)

        head.eval()
        test_preds: list[float] = []
        test_targets: list[float] = []
        test_loss_sum = 0.0
        test_count = 0
        with torch.no_grad():
            for batch in test_loader:
                pred = head(batch.z_t, batch.z_next, batch.delta, batch.action_oh, batch.sem)
                loss = nn.functional.mse_loss(pred, batch.reward, reduction="sum")
                test_loss_sum += float(loss.item())
                test_count += int(pred.numel())
                test_preds.extend(pred.detach().cpu().tolist())
                test_targets.extend(batch.reward.detach().cpu().tolist())
        test_mse = test_loss_sum / max(1, test_count)
        sign_acc = binary_sign_accuracy(test_preds, test_targets)
        pr = pearson_r(test_preds, test_targets)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_mse": test_mse,
            "test_sign_acc": sign_acc,
            "test_pearson": pr,
            "num_train": len(train_ds),
            "num_test": len(test_ds),
        }
        with open(log_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writerow(row)

        if test_mse < best["test_mse"]:
            best = {"test_mse": test_mse, "epoch": epoch + 1}

        if wb is not None:
            try:
                wb.log({k: v for k, v in row.items() if k != "epoch" and v is not None}, step=epoch + 1)
            except Exception:
                pass

    metrics = {
        "train_indices": len(train_idx),
        "test_indices": len(test_idx),
        "latent_cache_unique_paths": len(latent_cache),
        "encoder_cache_size": len(getattr(vision_encoder, "_latent_cache", {})),
        "best_test_mse": best["test_mse"],
        "best_epoch": best["epoch"],
        "args": vars(args),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    if wb is not None:
        try:
            wb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
