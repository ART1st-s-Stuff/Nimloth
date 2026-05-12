"""GT-latent 8-action Q/ranking head training for EB-Nav (frozen Qwen vision, token latents)."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.train_eb_nav_reward_head_gt import (  # noqa: E402
    freeze_qwen_visual_model,
    pearson_r,
    resolve_repo_path,
    semantic_bow_embedding,
)
from src.data.eb_nav_dataset import EBNavSequenceDataset, encode_eb_nav_action  # noqa: E402
from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402


@dataclass
class QBatchPack:
    z_t: torch.Tensor
    z_next: torch.Tensor
    delta: torch.Tensor
    action_oh: torch.Tensor
    sem: torch.Tensor
    pos_action: torch.Tensor
    has_reward: torch.Tensor
    reward: torch.Tensor
    fallback_all_missing: torch.Tensor


class TransitionQHead(nn.Module):
    """Scores each transition candidate with semantic conditioning."""

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
        # z_* : [B, A, P, D], action_oh: [B, A, Adim], sem: [B, S]
        batch, actions, _, _ = z_t.shape

        def pool(x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=2)

        v_t = pool(z_t)
        v_next = pool(z_next)
        v_delta = pool(delta)
        v_cat = torch.cat([v_t, v_next, v_delta], dim=-1)
        a = self.action_mlp(action_oh)
        sem_expand = sem.unsqueeze(1).expand(batch, actions, sem.shape[-1])
        h = torch.cat([v_cat, a, sem_expand], dim=-1)
        out = self.head(h.view(batch * actions, -1)).view(batch, actions)
        return out


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


def build_candidate_index(dataset: EBNavSequenceDataset, action_dim: int) -> tuple[dict[str, dict[int, list[str]]], dict[int, list[str]], list[str]]:
    by_src: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    by_action: dict[int, list[str]] = defaultdict(list)
    all_targets: list[str] = []
    for item in dataset.sequences:
        hist = item.get("history_images", [])
        fut = item.get("future_images", [])
        fa_ids = item.get("future_action_ids", [])
        if not hist or not fut or not fa_ids:
            continue
        src = str(hist[-1])
        tgt = str(fut[0])
        aid = int(fa_ids[0])
        if 0 <= aid < int(action_dim):
            by_src[src][aid].append(tgt)
            by_action[aid].append(tgt)
        all_targets.append(tgt)
    return by_src, by_action, all_targets


def sample_target_for_action(
    *,
    rng: random.Random,
    src_path: str,
    action_id: int,
    src_index: dict[str, dict[int, list[str]]],
    action_index: dict[int, list[str]],
    all_targets: list[str],
    positive_fallback: str,
) -> tuple[str, bool]:
    src_action_paths = src_index.get(src_path, {}).get(int(action_id), [])
    if src_action_paths:
        return str(rng.choice(src_action_paths)), True
    if int(action_id) in action_index and action_index[int(action_id)]:
        return str(rng.choice(action_index[int(action_id)])), False
    if all_targets:
        return str(rng.choice(all_targets)), False
    return str(positive_fallback), False


def collate_builder(
    *,
    encoder: QwenLLMLatentEncoder,
    action_dim: int,
    semantic_dim: int,
    semantic_buckets: int,
    seed: int,
    device: torch.device,
    latent_cache: dict[str, torch.Tensor],
    src_index: dict[str, dict[int, list[str]]],
    action_index: dict[int, list[str]],
    all_targets: list[str],
    epoch_rng: random.Random,
) -> Any:
    def collate(batch: list[dict[str, Any]]) -> QBatchPack:
        z_t_rows: list[torch.Tensor] = []
        z_next_rows: list[torch.Tensor] = []
        action_rows: list[torch.Tensor] = []
        sem_rows: list[torch.Tensor] = []
        pos_actions: list[int] = []
        rewards: list[float] = []
        has_rewards: list[float] = []
        fallback_flags: list[float] = []

        for item in batch:
            hist = item.get("history_images", [])
            fut = item.get("future_images", [])
            fa_ids = item.get("future_action_ids", [])
            fr = item.get("future_rewards", [])
            if not hist or not fa_ids:
                continue

            src_path = str(hist[-1])
            pos_action = int(fa_ids[0])
            if not (0 <= pos_action < int(action_dim)):
                continue

            pos_target = str(fut[0]) if fut else ""
            if not pos_target:
                pos_target, _ = sample_target_for_action(
                    rng=epoch_rng,
                    src_path=src_path,
                    action_id=pos_action,
                    src_index=src_index,
                    action_index=action_index,
                    all_targets=all_targets,
                    positive_fallback=src_path,
                )

            candidate_paths: list[str] = [src_path] * int(action_dim)
            matched_from_src = 0
            for aid in range(int(action_dim)):
                if aid == pos_action:
                    candidate_paths[aid] = pos_target
                    if src_index.get(src_path, {}).get(aid):
                        matched_from_src += 1
                    continue
                candidate, matched = sample_target_for_action(
                    rng=epoch_rng,
                    src_path=src_path,
                    action_id=aid,
                    src_index=src_index,
                    action_index=action_index,
                    all_targets=all_targets,
                    positive_fallback=pos_target,
                )
                candidate_paths[aid] = candidate
                if matched:
                    matched_from_src += 1

            fallback_flags.append(1.0 if matched_from_src == 0 else 0.0)

            z_t = encode_visual_tokens(encoder, src_path, device=device, extra_cache=latent_cache)
            z_t_rep = torch.stack([z_t for _ in range(int(action_dim))], dim=0)
            z_n = torch.stack(
                [encode_visual_tokens(encoder, cp, device=device, extra_cache=latent_cache) for cp in candidate_paths],
                dim=0,
            )
            z_t_rows.append(z_t_rep)
            z_next_rows.append(z_n)

            act_rows = torch.tensor([encode_eb_nav_action(aid, action_dim) for aid in range(int(action_dim))], dtype=torch.float32, device=device)
            action_rows.append(act_rows)

            instr = str(item.get("instruction", ""))
            prompt = str(item.get("prompt", ""))
            sem_rows.append(semantic_bow_embedding(f"{instr}\n{prompt}", dim=semantic_dim, num_buckets=semantic_buckets, seed=seed).to(device))

            pos_actions.append(pos_action)
            if fr:
                rewards.append(float(fr[0]))
                has_rewards.append(1.0)
            else:
                rewards.append(0.0)
                has_rewards.append(0.0)

        if not z_t_rows:
            raise RuntimeError("Empty batch after filtering; check dataset paths.")

        z_t_batch = torch.stack(z_t_rows, dim=0)
        z_next_batch = torch.stack(z_next_rows, dim=0)
        delta_batch = z_next_batch - z_t_batch
        action_batch = torch.stack(action_rows, dim=0)
        sem_batch = torch.stack(sem_rows, dim=0)
        pos_action_batch = torch.tensor(pos_actions, dtype=torch.long, device=device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=device)
        has_reward_batch = torch.tensor(has_rewards, dtype=torch.float32, device=device)
        fallback_batch = torch.tensor(fallback_flags, dtype=torch.float32, device=device)
        return QBatchPack(
            z_t=z_t_batch,
            z_next=z_next_batch,
            delta=delta_batch,
            action_oh=action_batch,
            sem=sem_batch,
            pos_action=pos_action_batch,
            has_reward=has_reward_batch,
            reward=reward_batch,
            fallback_all_missing=fallback_batch,
        )

    return collate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train semantic-conditioned GT-latent Q/ranking head on frozen Qwen visual tokens.")
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
    p.add_argument("--pairwise-loss", default="bce", choices=["bce", "margin"])
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--reward-reg-weight", type=float, default=0.0)
    p.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"])
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb regardless of wandb-mode.")
    p.add_argument("--use-fallback", action="store_true", help="Use adapter fallback latents (no GPU model load). For dry runs only.")
    return p.parse_args()


def compute_pairwise_metrics(scores: torch.Tensor, pos_action: torch.Tensor) -> tuple[float, float, float, float, float, list[int], list[int]]:
    # scores: [B, A], pos_action: [B]
    batch, actions = scores.shape
    row_idx = torch.arange(batch, device=scores.device)
    pos_scores = scores[row_idx, pos_action]
    top1 = scores.argmax(dim=1)
    top1_acc = float((top1 == pos_action).float().mean().item()) if batch > 0 else 0.0

    pair_ok = 0.0
    pair_total = 0.0
    neg_means: list[float] = []
    for i in range(batch):
        pa = int(pos_action[i].item())
        ps = float(pos_scores[i].item())
        neg_scores = [float(scores[i, a].item()) for a in range(actions) if a != pa]
        if neg_scores:
            pair_ok += sum(1.0 for ns in neg_scores if ps > ns)
            pair_total += float(len(neg_scores))
            neg_means.append(sum(neg_scores) / len(neg_scores))
        else:
            neg_means.append(0.0)
    pair_acc = pair_ok / pair_total if pair_total > 0 else 0.0

    mean_pos = float(pos_scores.mean().item()) if batch > 0 else 0.0
    mean_neg = float(sum(neg_means) / len(neg_means)) if neg_means else 0.0
    score_gap = mean_pos - mean_neg

    per_action_counts = [0 for _ in range(actions)]
    per_action_correct = [0 for _ in range(actions)]
    for i in range(batch):
        pa = int(pos_action[i].item())
        per_action_counts[pa] += 1
        if int(top1[i].item()) == pa:
            per_action_correct[pa] += 1

    return top1_acc, pair_acc, mean_pos, mean_neg, score_gap, per_action_counts, per_action_correct


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

    src_index, action_index, all_targets = build_candidate_index(full_dataset, int(args.action_dim))

    all_idx = list(range(len(full_dataset)))
    rng.shuffle(all_idx)
    max_train = int(args.max_samples) if int(args.max_samples) > 0 else len(all_idx)
    train_n = min(max_train, len(all_idx))
    train_idx = all_idx[:train_n]

    if int(args.test_max_samples) > 0:
        test_n = int(args.test_max_samples)
    else:
        test_n = max(64, min(256, max(1, train_n // 5)))
    train_idx_set = set(train_idx)
    remaining = [i for i in all_idx if i not in train_idx_set]
    rng.shuffle(remaining)
    test_idx = remaining[: min(test_n, len(remaining))]

    train_ds = Subset(full_dataset, train_idx)
    test_ds = Subset(full_dataset, test_idx)

    out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "dev" / f"{datetime.now():%Y%m%d_%H%M%S}_eb_nav_q_head_gt"
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
        src_index=src_index,
        action_index=action_index,
        all_targets=all_targets,
        epoch_rng=rng,
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

    head = TransitionQHead(
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

            wb = wandb.init(project="eb-nav-q-head-gt", mode=None if args.wandb_mode == "online" else args.wandb_mode)
        except Exception:
            wb = None

    log_csv = out / "train_log.csv"
    with open(log_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "test_top1_action_acc",
                "test_pairwise_rank_acc",
                "test_mean_pos_score",
                "test_mean_neg_score",
                "test_score_gap",
                "test_reward_mse",
                "test_reward_pearson",
                "num_train",
                "num_test",
            ],
        )
        w.writeheader()

    best = {"top1": -1.0, "epoch": -1}
    last_eval: dict[str, Any] = {}

    for epoch in range(int(args.epochs)):
        head.train()
        ep_loss = 0.0
        ep_n = 0
        for batch in train_loader:
            scores = head(batch.z_t, batch.z_next, batch.delta, batch.action_oh, batch.sem)
            row_idx = torch.arange(scores.shape[0], device=scores.device)
            pos = scores[row_idx, batch.pos_action]
            loss_rank = torch.tensor(0.0, device=scores.device)
            neg_terms = 0
            for aid in range(scores.shape[1]):
                mask = batch.pos_action != aid
                if not torch.any(mask):
                    continue
                neg = scores[mask, aid]
                pos_masked = pos[mask]
                if args.pairwise_loss == "margin":
                    term = torch.relu(float(args.margin) - (pos_masked - neg))
                else:
                    term = nn.functional.softplus(-(pos_masked - neg))
                loss_rank = loss_rank + term.sum()
                neg_terms += int(term.numel())
            if neg_terms > 0:
                loss_rank = loss_rank / float(neg_terms)

            loss = loss_rank
            if float(args.reward_reg_weight) > 0.0:
                valid = batch.has_reward > 0.5
                if torch.any(valid):
                    reward_loss = nn.functional.mse_loss(pos[valid], batch.reward[valid])
                    loss = loss + float(args.reward_reg_weight) * reward_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * int(scores.shape[0])
            ep_n += int(scores.shape[0])
        train_loss = ep_loss / max(1, ep_n)

        head.eval()
        top1_sum = 0.0
        pair_sum = 0.0
        mean_pos_sum = 0.0
        mean_neg_sum = 0.0
        gap_sum = 0.0
        batch_count = 0
        reward_preds: list[float] = []
        reward_targets: list[float] = []
        per_action_counts = [0 for _ in range(int(args.action_dim))]
        per_action_correct = [0 for _ in range(int(args.action_dim))]
        fallback_all_missing = 0
        with torch.no_grad():
            for batch in test_loader:
                scores = head(batch.z_t, batch.z_next, batch.delta, batch.action_oh, batch.sem)
                top1, pair_acc, mean_pos, mean_neg, gap, pa_counts, pa_correct = compute_pairwise_metrics(scores, batch.pos_action)
                top1_sum += top1
                pair_sum += pair_acc
                mean_pos_sum += mean_pos
                mean_neg_sum += mean_neg
                gap_sum += gap
                batch_count += 1
                for aid in range(int(args.action_dim)):
                    per_action_counts[aid] += pa_counts[aid]
                    per_action_correct[aid] += pa_correct[aid]
                fallback_all_missing += int((batch.fallback_all_missing > 0.5).sum().item())

                row_idx = torch.arange(scores.shape[0], device=scores.device)
                pos_scores = scores[row_idx, batch.pos_action]
                valid = batch.has_reward > 0.5
                if torch.any(valid):
                    reward_preds.extend(pos_scores[valid].detach().cpu().tolist())
                    reward_targets.extend(batch.reward[valid].detach().cpu().tolist())

        test_top1 = top1_sum / max(1, batch_count)
        test_pair = pair_sum / max(1, batch_count)
        test_mean_pos = mean_pos_sum / max(1, batch_count)
        test_mean_neg = mean_neg_sum / max(1, batch_count)
        test_gap = gap_sum / max(1, batch_count)
        if reward_preds:
            pred_t = torch.tensor(reward_preds)
            tgt_t = torch.tensor(reward_targets)
            test_reward_mse = float(nn.functional.mse_loss(pred_t, tgt_t).item())
            test_reward_pearson = pearson_r(reward_preds, reward_targets)
        else:
            test_reward_mse = None
            test_reward_pearson = None

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_top1_action_acc": test_top1,
            "test_pairwise_rank_acc": test_pair,
            "test_mean_pos_score": test_mean_pos,
            "test_mean_neg_score": test_mean_neg,
            "test_score_gap": test_gap,
            "test_reward_mse": test_reward_mse,
            "test_reward_pearson": test_reward_pearson,
            "num_train": len(train_ds),
            "num_test": len(test_ds),
        }
        with open(log_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writerow(row)

        if test_top1 > best["top1"]:
            best = {"top1": test_top1, "epoch": epoch + 1}

        pa_acc = {
            str(aid): (float(per_action_correct[aid]) / float(per_action_counts[aid]) if per_action_counts[aid] > 0 else None)
            for aid in range(int(args.action_dim))
        }
        last_eval = {
            "top1_action_acc": test_top1,
            "pairwise_rank_acc": test_pair,
            "mean_pos_score": test_mean_pos,
            "mean_neg_score": test_mean_neg,
            "score_gap": test_gap,
            "per_action_counts": {str(aid): int(per_action_counts[aid]) for aid in range(int(args.action_dim))},
            "per_action_accuracy": pa_acc,
            "fallback_all_missing_count": fallback_all_missing,
            "reward_mse": test_reward_mse,
            "reward_pearson": test_reward_pearson,
        }

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
        "best_top1_action_acc": best["top1"],
        "best_epoch": best["epoch"],
        "final_eval": last_eval,
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
