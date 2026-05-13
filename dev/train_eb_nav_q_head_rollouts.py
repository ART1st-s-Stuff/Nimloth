"""Train/eval lightweight Q/ranking head on collected rollout transitions JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.train_eb_nav_reward_head_gt import (  # noqa: E402
    freeze_qwen_visual_model,
    pearson_r,
    resolve_repo_path,
    semantic_bow_embedding,
)
from src.data.eb_nav_dataset import encode_eb_nav_action  # noqa: E402
from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402


@dataclass
class RolloutTransition:
    image_t: str
    image_next: str
    sampled_action_id: int
    reward: float | None
    instruction: str
    policy: str


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
    fallback_by_action_count: torch.Tensor
    fallback_global_count: torch.Tensor


class TransitionQHead(nn.Module):
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
        return self.head(h.view(batch * actions, -1)).view(batch, actions)


@torch.no_grad()
def encode_visual_tokens(encoder: QwenLLMLatentEncoder, path: str, *, device: torch.device, extra_cache: dict[str, torch.Tensor]) -> torch.Tensor:
    if path in extra_cache:
        return extra_cache[path]
    out = encoder.encode_image_path_with_prompt(image_path=path, prompt_override=None)
    z = out.z
    if z.dim() == 1:
        z = z.unsqueeze(0)
    z = z.detach().float().to(device=device, non_blocking=True)
    extra_cache[path] = z
    return z


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train fallback-ranking Q-head on rollout transitions JSONL.")
    p.add_argument("--transitions-jsonl", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=128, help="Max train samples after deterministic shuffle (0 = all).")
    p.add_argument("--test-max-samples", type=int, default=64, help="Max test samples after train split (0 = all remaining).")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--cuda-device", default="0", help="CUDA device index or -1 for CPU.")
    p.add_argument("--visual-num-tokens", type=int, default=16)
    p.add_argument("--token-dim", type=int, default=3584)
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
    p.add_argument("--use-fallback", action="store_true", help="Use adapter fallback latents (dry runs only).")
    return p.parse_args()


def _to_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def load_transitions(paths: list[str], action_dim: int) -> list[RolloutTransition]:
    rows: list[RolloutTransition] = []
    for raw_path in paths:
        path = resolve_repo_path(raw_path)
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                action_id = _to_int(item.get("sampled_action_id"), default=None)
                image_t = str(item.get("image_t", "")).strip()
                image_next = str(item.get("image_next", "")).strip()
                if action_id is None or not (0 <= int(action_id) < int(action_dim)):
                    continue
                if not image_t or not image_next:
                    continue
                path_t = resolve_repo_path(image_t)
                path_n = resolve_repo_path(image_next)
                if not path_t.is_file() or not path_n.is_file():
                    continue
                rows.append(
                    RolloutTransition(
                        image_t=str(path_t),
                        image_next=str(path_n),
                        sampled_action_id=int(action_id),
                        reward=_to_float(item.get("reward"), default=None),
                        instruction=str(item.get("instruction", "")),
                        policy=str(item.get("policy", "unknown")),
                    )
                )
    return rows


def split_samples(rows: list[RolloutTransition], *, seed: int, max_samples: int, test_max_samples: int) -> tuple[list[RolloutTransition], list[RolloutTransition]]:
    idx = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    shuffled = [rows[i] for i in idx]

    train_n = len(shuffled) if int(max_samples) <= 0 else min(len(shuffled), int(max_samples))
    rem = shuffled[train_n:]
    if int(test_max_samples) <= 0:
        test_n = len(rem)
    else:
        test_n = min(len(rem), int(test_max_samples))
    return shuffled[:train_n], rem[:test_n]


def build_candidate_indexes(rows: list[RolloutTransition], action_dim: int) -> tuple[dict[int, list[str]], list[str], Counter[int]]:
    by_action: dict[int, list[str]] = defaultdict(list)
    all_next: list[str] = []
    counts: Counter[int] = Counter()
    for row in rows:
        aid = int(row.sampled_action_id)
        if 0 <= aid < int(action_dim):
            by_action[aid].append(str(row.image_next))
            counts[aid] += 1
            all_next.append(str(row.image_next))
    return by_action, all_next, counts


def sample_negative_next(
    *,
    action_id: int,
    by_action: dict[int, list[str]],
    all_next: list[str],
    rng: random.Random,
    pos_next: str,
) -> tuple[str, str]:
    candidates = by_action.get(int(action_id), [])
    if candidates:
        return str(rng.choice(candidates)), "action"
    if all_next:
        return str(rng.choice(all_next)), "global"
    return pos_next, "none"


def collate_builder(
    *,
    encoder: QwenLLMLatentEncoder,
    action_dim: int,
    semantic_dim: int,
    semantic_buckets: int,
    seed: int,
    device: torch.device,
    latent_cache: dict[str, torch.Tensor],
    by_action: dict[int, list[str]],
    all_next: list[str],
    epoch_rng: random.Random,
) -> Any:
    def collate(batch: list[RolloutTransition]) -> QBatchPack:
        z_t_rows: list[torch.Tensor] = []
        z_next_rows: list[torch.Tensor] = []
        action_rows: list[torch.Tensor] = []
        sem_rows: list[torch.Tensor] = []
        pos_actions: list[int] = []
        rewards: list[float] = []
        has_rewards: list[float] = []
        fallback_by_action_count: list[float] = []
        fallback_global_count: list[float] = []

        for row in batch:
            pos_action = int(row.sampled_action_id)
            if not (0 <= pos_action < int(action_dim)):
                continue

            candidate_paths: list[str] = [str(row.image_next)] * int(action_dim)
            fb_action = 0
            fb_global = 0
            for aid in range(int(action_dim)):
                if aid == pos_action:
                    candidate_paths[aid] = str(row.image_next)
                    continue
                sampled_path, source = sample_negative_next(
                    action_id=aid,
                    by_action=by_action,
                    all_next=all_next,
                    rng=epoch_rng,
                    pos_next=str(row.image_next),
                )
                candidate_paths[aid] = sampled_path
                if source == "action":
                    pass
                elif source == "global":
                    fb_action += 1
                    fb_global += 1
                else:
                    fb_action += 1
                    fb_global += 1

            z_t = encode_visual_tokens(encoder, str(row.image_t), device=device, extra_cache=latent_cache)
            z_t_rep = torch.stack([z_t for _ in range(int(action_dim))], dim=0)
            z_n = torch.stack([encode_visual_tokens(encoder, cp, device=device, extra_cache=latent_cache) for cp in candidate_paths], dim=0)
            z_t_rows.append(z_t_rep)
            z_next_rows.append(z_n)

            act_rows = torch.tensor([encode_eb_nav_action(aid, action_dim) for aid in range(int(action_dim))], dtype=torch.float32, device=device)
            action_rows.append(act_rows)
            sem_rows.append(
                semantic_bow_embedding(f"{row.instruction}\n{row.policy}", dim=semantic_dim, num_buckets=semantic_buckets, seed=seed).to(device)
            )
            pos_actions.append(pos_action)

            if row.reward is None:
                rewards.append(0.0)
                has_rewards.append(0.0)
            else:
                rewards.append(float(row.reward))
                has_rewards.append(1.0)

            fallback_by_action_count.append(float(fb_action))
            fallback_global_count.append(float(fb_global))

        if not z_t_rows:
            raise RuntimeError("Empty batch after filtering; check transitions paths.")

        z_t_batch = torch.stack(z_t_rows, dim=0)
        z_next_batch = torch.stack(z_next_rows, dim=0)
        delta_batch = z_next_batch - z_t_batch
        action_batch = torch.stack(action_rows, dim=0)
        sem_batch = torch.stack(sem_rows, dim=0)
        pos_action_batch = torch.tensor(pos_actions, dtype=torch.long, device=device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=device)
        has_reward_batch = torch.tensor(has_rewards, dtype=torch.float32, device=device)
        fb_action_batch = torch.tensor(fallback_by_action_count, dtype=torch.float32, device=device)
        fb_global_batch = torch.tensor(fallback_global_count, dtype=torch.float32, device=device)
        return QBatchPack(
            z_t=z_t_batch,
            z_next=z_next_batch,
            delta=delta_batch,
            action_oh=action_batch,
            sem=sem_batch,
            pos_action=pos_action_batch,
            has_reward=has_reward_batch,
            reward=reward_batch,
            fallback_by_action_count=fb_action_batch,
            fallback_global_count=fb_global_batch,
        )

    return collate


def compute_eval_metrics(scores: torch.Tensor, pos_action: torch.Tensor, action_dim: int) -> tuple[dict[str, Any], torch.Tensor]:
    batch = int(scores.shape[0])
    row_idx = torch.arange(batch, device=scores.device)
    top1 = scores.argmax(dim=1)
    pos_scores = scores[row_idx, pos_action]

    top1_action_acc = float((top1 == pos_action).float().mean().item()) if batch > 0 else 0.0

    pair_ok = 0.0
    pair_total = 0.0
    neg_means: list[float] = []
    for i in range(batch):
        pa = int(pos_action[i].item())
        ps = float(pos_scores[i].item())
        negs = [float(scores[i, a].item()) for a in range(int(action_dim)) if a != pa]
        if negs:
            pair_ok += sum(1.0 for v in negs if ps > v)
            pair_total += float(len(negs))
            neg_means.append(sum(negs) / len(negs))
        else:
            neg_means.append(0.0)
    pair_acc = pair_ok / pair_total if pair_total > 0 else 0.0

    mean_pos = float(pos_scores.mean().item()) if batch > 0 else 0.0
    mean_neg = float(sum(neg_means) / len(neg_means)) if neg_means else 0.0
    score_gap = mean_pos - mean_neg

    per_action_counts = [0 for _ in range(int(action_dim))]
    per_action_correct = [0 for _ in range(int(action_dim))]
    pred_counts = [0 for _ in range(int(action_dim))]
    for i in range(batch):
        true_a = int(pos_action[i].item())
        pred_a = int(top1[i].item())
        per_action_counts[true_a] += 1
        if pred_a == true_a:
            per_action_correct[true_a] += 1
        pred_counts[pred_a] += 1

    macro_vals = [
        float(per_action_correct[a]) / float(per_action_counts[a])
        for a in range(int(action_dim))
        if per_action_counts[a] > 0
    ]
    macro_action_acc = float(sum(macro_vals) / len(macro_vals)) if macro_vals else 0.0

    return (
        {
            "top1_action_acc": top1_action_acc,
            "macro_action_acc": macro_action_acc,
            "pairwise_rank_acc": pair_acc,
            "score_gap": score_gap,
            "mean_pos_score": mean_pos,
            "mean_neg_score": mean_neg,
            "per_action_counts": per_action_counts,
            "per_action_correct": per_action_correct,
            "pred_counts": pred_counts,
        },
        top1,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    rows = load_transitions(args.transitions_jsonl, int(args.action_dim))
    if not rows:
        raise RuntimeError("No valid transitions loaded with existing image_t/image_next and sampled_action_id.")

    train_rows, test_rows = split_samples(rows, seed=int(args.seed), max_samples=int(args.max_samples), test_max_samples=int(args.test_max_samples))
    if not train_rows:
        raise RuntimeError("Train split is empty after filtering/splitting.")
    if not test_rows:
        raise RuntimeError("Test split is empty; increase input rows or lower --max-samples.")

    by_action, all_next, all_counts = build_candidate_indexes(train_rows, int(args.action_dim))

    out = resolve_repo_path(args.output_dir)
    if out.suffix:
        out = out.parent
    if str(args.output_dir).strip() == "":
        out = REPO_ROOT / "outputs" / "dev" / f"{datetime.now():%Y%m%d_%H%M%S}_eb_nav_q_head_rollouts"
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

    encoder = QwenLLMLatentEncoder(
        latent_dim=int(args.visual_num_tokens) * int(args.token_dim),
        name="qwen_visual_frozen_rollout_q",
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
        encoder=encoder,
        action_dim=int(args.action_dim),
        semantic_dim=int(args.semantic_dim),
        semantic_buckets=int(args.semantic_buckets),
        seed=int(args.seed),
        device=device,
        latent_cache=latent_cache,
        by_action=by_action,
        all_next=all_next,
        epoch_rng=rng,
    )

    train_loader = DataLoader(train_rows, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers), collate_fn=collate_fn)
    test_loader = DataLoader(test_rows, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), collate_fn=collate_fn)

    head = TransitionQHead(
        token_dim=int(args.token_dim),
        num_tokens=int(args.visual_num_tokens),
        action_dim=int(args.action_dim),
        semantic_dim=int(args.semantic_dim),
        action_hidden=int(args.action_hidden),
        mlp_hidden=int(args.mlp_hidden),
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    log_csv = out / "train_log.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "test_top1_action_acc",
                "test_macro_action_acc",
                "test_pairwise_rank_acc",
                "test_score_gap",
                "test_reward_mse",
                "test_reward_pearson",
                "test_fallback_by_action_count",
                "test_fallback_global_count",
                "num_train",
                "num_test",
            ],
        ).writeheader()

    best = {"top1": -1.0, "epoch": -1}
    last_eval: dict[str, Any] = {}
    for epoch in range(int(args.epochs)):
        head.train()
        train_loss_sum = 0.0
        train_n = 0
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
            train_loss_sum += float(loss.item()) * int(scores.shape[0])
            train_n += int(scores.shape[0])
        train_loss = train_loss_sum / max(1, train_n)

        head.eval()
        top1_sum = 0.0
        macro_sum = 0.0
        pair_sum = 0.0
        gap_sum = 0.0
        batches = 0
        pred_counts_total = [0 for _ in range(int(args.action_dim))]
        true_counts_total = [0 for _ in range(int(args.action_dim))]
        true_correct_total = [0 for _ in range(int(args.action_dim))]
        reward_preds: list[float] = []
        reward_targets: list[float] = []
        fb_action_total = 0.0
        fb_global_total = 0.0

        with torch.no_grad():
            for batch in test_loader:
                scores = head(batch.z_t, batch.z_next, batch.delta, batch.action_oh, batch.sem)
                m, _ = compute_eval_metrics(scores, batch.pos_action, int(args.action_dim))
                top1_sum += float(m["top1_action_acc"])
                macro_sum += float(m["macro_action_acc"])
                pair_sum += float(m["pairwise_rank_acc"])
                gap_sum += float(m["score_gap"])
                batches += 1

                pa_counts = m["per_action_counts"]
                pa_correct = m["per_action_correct"]
                pred_counts = m["pred_counts"]
                for aid in range(int(args.action_dim)):
                    true_counts_total[aid] += int(pa_counts[aid])
                    true_correct_total[aid] += int(pa_correct[aid])
                    pred_counts_total[aid] += int(pred_counts[aid])

                row_idx = torch.arange(scores.shape[0], device=scores.device)
                pos_scores = scores[row_idx, batch.pos_action]
                valid = batch.has_reward > 0.5
                if torch.any(valid):
                    reward_preds.extend(pos_scores[valid].detach().cpu().tolist())
                    reward_targets.extend(batch.reward[valid].detach().cpu().tolist())

                fb_action_total += float(batch.fallback_by_action_count.sum().item())
                fb_global_total += float(batch.fallback_global_count.sum().item())

        test_top1 = top1_sum / max(1, batches)
        test_macro = macro_sum / max(1, batches)
        test_pair = pair_sum / max(1, batches)
        test_gap = gap_sum / max(1, batches)
        reward_mse: float | None
        reward_corr: float | None
        if reward_preds:
            pred_t = torch.tensor(reward_preds)
            tgt_t = torch.tensor(reward_targets)
            reward_mse = float(nn.functional.mse_loss(pred_t, tgt_t).item())
            reward_corr = pearson_r(reward_preds, reward_targets)
        else:
            reward_mse = None
            reward_corr = None

        if test_top1 > best["top1"]:
            best = {"top1": test_top1, "epoch": epoch + 1}

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_top1_action_acc": test_top1,
            "test_macro_action_acc": test_macro,
            "test_pairwise_rank_acc": test_pair,
            "test_score_gap": test_gap,
            "test_reward_mse": reward_mse,
            "test_reward_pearson": reward_corr,
            "test_fallback_by_action_count": fb_action_total,
            "test_fallback_global_count": fb_global_total,
            "num_train": len(train_rows),
            "num_test": len(test_rows),
        }
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)

        per_action_accuracy = {
            str(a): (float(true_correct_total[a]) / float(true_counts_total[a]) if true_counts_total[a] > 0 else None)
            for a in range(int(args.action_dim))
        }
        pred_dist_total = sum(pred_counts_total)
        pred_action_distribution = {
            str(a): (float(pred_counts_total[a]) / float(pred_dist_total) if pred_dist_total > 0 else 0.0)
            for a in range(int(args.action_dim))
        }
        last_eval = {
            "top1_action_acc": test_top1,
            "macro_action_acc": test_macro,
            "pairwise_rank_acc": test_pair,
            "score_gap": test_gap,
            "per_action_counts": {str(a): int(true_counts_total[a]) for a in range(int(args.action_dim))},
            "per_action_accuracy": per_action_accuracy,
            "pred_action_distribution": pred_action_distribution,
            "fallback_by_action_count": int(fb_action_total),
            "fallback_global_count": int(fb_global_total),
            "reward_mse": reward_mse,
            "reward_pearson": reward_corr,
        }

    metrics = {
        "num_loaded_rows": len(rows),
        "num_train": len(train_rows),
        "num_test": len(test_rows),
        "train_action_distribution": {str(k): int(v) for k, v in sorted(all_counts.items())},
        "latent_cache_unique_paths": len(latent_cache),
        "encoder_cache_size": len(getattr(encoder, "_latent_cache", {})),
        "best_top1_action_acc": best["top1"],
        "best_epoch": best["epoch"],
        "final_eval": last_eval,
        "args": vars(args),
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
