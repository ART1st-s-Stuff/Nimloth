"""8-action counterfactual value-ranking eval for SemanticWMValueHead checkpoints.

For each heldout state, keep the visual/semantic history fixed, enumerate action ids 0..7,
use WM to predict next visual latent for each action, score with the value head, and compare
argmax/rank against the logged/expert future_action_ids[0]. This is a proxy ranking eval;
other actions do not have causal ground-truth returns in the current manifests.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.train_eb_nav_joint_wm_value import build_trainable_wm_from_checkpoint, freeze_qwen, make_subset  # noqa: E402
from dev.train_eb_nav_value_head_predicted import SemanticWMValueHead, encode_many, pearson_r, resolve_repo_path  # noqa: E402
from src.train.train_wm_joint import _joint_collate_fn  # noqa: E402
from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402


def one_hot(ids: torch.Tensor, dim: int = 8) -> torch.Tensor:
    out = torch.zeros(ids.numel(), dim, dtype=torch.float32, device=ids.device)
    out.scatter_(1, ids.reshape(-1, 1), 1.0)
    return out


def discounted_return(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    discounts = torch.tensor([float(gamma) ** i for i in range(rewards.size(1))], dtype=torch.float32, device=rewards.device)
    return (rewards.float() * discounts.unsqueeze(0)).sum(dim=1)


@torch.no_grad()
def eval_ranking(
    *,
    loader: DataLoader,
    visual_encoder: QwenLLMLatentEncoder,
    semantic_encoder: QwenLLMLatentEncoder,
    wm: torch.nn.Module,
    head: SemanticWMValueHead,
    device: torch.device,
    gamma: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    wm.eval(); head.eval()
    n = 0
    top1 = 0
    ranks: list[int] = []
    rr: list[float] = []
    pairwise_fracs: list[float] = []
    gaps: list[float] = []
    logged_scores: list[float] = []
    target_values: list[float] = []
    by_action: dict[int, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "top1": 0.0, "rank_sum": 0.0, "mrr_sum": 0.0, "pairwise_sum": 0.0, "gap_sum": 0.0})
    rows: list[dict[str, Any]] = []

    for raw in loader:
        history_images: list[list[str]] = raw["history_images"]
        batch_size = len(history_images)
        prompts = [str(x or "") for x in raw.get("prompts", raw.get("instructions", [""] * batch_size))]
        hist_actions = raw["history_actions"].float().to(device)
        if "future_action_ids" not in raw:
            raise RuntimeError("future_action_ids required for ranking eval")
        logged_ids = raw["future_action_ids"][:, 0].long().to(device)
        rewards = raw.get("future_rewards")
        if rewards is None:
            raise RuntimeError("future_rewards required for logged-score/return correlation")
        values = discounted_return(rewards.float().to(device), gamma)

        flat_hist = [path for seq in history_images for path in seq]
        z_hist = encode_many(visual_encoder, flat_hist, None, device).reshape(batch_size, len(history_images[0]), 16, 3584)
        z_current = z_hist[:, -1]
        semantic = encode_many(
            semantic_encoder,
            [seq[-1] for seq in history_images],
            prompts,
            device,
            expected_flat_dim=3584,
        ).reshape(batch_size, 3584)

        score_cols: list[torch.Tensor] = []
        for action_id in range(8):
            teacher_action = hist_actions.clone()
            ids = torch.full((batch_size,), action_id, dtype=torch.long, device=device)
            action_vec = one_hot(ids, 8)
            teacher_action[:, -1, :] = action_vec
            z_next = wm.predict_next(z_hist, teacher_action)
            score_cols.append(head(semantic, z_current, z_next, action_vec))
        scores = torch.stack(score_cols, dim=1)  # B, 8
        pred_ids = torch.argmax(scores, dim=1)

        for i in range(batch_size):
            logged = int(logged_ids[i].item())
            pred = int(pred_ids[i].item())
            sorted_ids = torch.argsort(scores[i], descending=True)
            rank = int((sorted_ids == logged).nonzero(as_tuple=False)[0].item()) + 1
            logged_score = float(scores[i, logged].item())
            other_scores = [float(scores[i, j].item()) for j in range(8) if j != logged]
            max_other = max(other_scores)
            pairwise = sum(1 for s in other_scores if logged_score > s) / 7.0
            gap = logged_score - max_other
            target = float(values[i].item())
            ok = 1 if pred == logged else 0

            n += 1; top1 += ok
            ranks.append(rank); rr.append(1.0 / rank); pairwise_fracs.append(pairwise); gaps.append(gap)
            logged_scores.append(logged_score); target_values.append(target)
            st = by_action[logged]
            st["n"] += 1; st["top1"] += ok; st["rank_sum"] += rank; st["mrr_sum"] += 1.0 / rank; st["pairwise_sum"] += pairwise; st["gap_sum"] += gap
            rows.append({
                "idx": n - 1,
                "logged_action": logged,
                "pred_action": pred,
                "top1": ok,
                "rank": rank,
                "mrr": 1.0 / rank,
                "pairwise_logged_gt_other": pairwise,
                "score_gap_logged_vs_max_other": gap,
                "logged_score": logged_score,
                "target_return": target,
                **{f"score_a{a}": float(scores[i, a].item()) for a in range(8)},
            })

    per_action = {}
    for a in range(8):
        st = by_action[a]
        if st["n"] <= 0:
            per_action[str(a)] = {"n": 0}
        else:
            c = st["n"]
            per_action[str(a)] = {
                "n": int(c),
                "top1_acc": st["top1"] / c,
                "mean_rank": st["rank_sum"] / c,
                "mrr": st["mrr_sum"] / c,
                "pairwise_logged_gt_other": st["pairwise_sum"] / c,
                "score_gap_logged_vs_max_other": st["gap_sum"] / c,
            }
    macro_top1 = sum(v.get("top1_acc", 0.0) for v in per_action.values() if v.get("n", 0) > 0) / max(1, sum(1 for v in per_action.values() if v.get("n", 0) > 0))
    metrics = {
        "n": n,
        "top1_acc": top1 / max(1, n),
        "macro_top1_acc": macro_top1,
        "mean_rank": sum(ranks) / max(1, len(ranks)),
        "mrr": sum(rr) / max(1, len(rr)),
        "pairwise_logged_gt_other": sum(pairwise_fracs) / max(1, len(pairwise_fracs)),
        "score_gap_logged_vs_max_other": sum(gaps) / max(1, len(gaps)),
        "logged_score_return_pearson": pearson_r(logged_scores, target_values),
        "per_action": per_action,
    }
    return metrics, rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--images-base-dir", default=".")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split-name", default="test")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-dtype", default="auto")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--write-per-sample", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if str(args.cuda_device) in {"", "-1", "cpu"} else f"cuda:{args.cuda_device}")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    ckpt_path = resolve_repo_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    wm = build_trainable_wm_from_checkpoint(ckpt_path, device)
    if "wm_state" in ckpt:
        wm.load_state_dict(ckpt["wm_state"], strict=False)
    wm.eval()
    if "head_state" not in ckpt:
        raise RuntimeError(f"checkpoint lacks head_state: {ckpt_path}")
    head = SemanticWMValueHead(semantic_dim=3584, visual_dim=3584, action_dim=8, hidden=int(args.hidden)).to(device)
    head.load_state_dict(ckpt["head_state"], strict=True)
    head.eval()

    adapter = QwenVLMAdapter(
        model_name=args.model_name,
        latent_dim=57344,
        enabled=True,
        fallback_enabled=False,
        device_map=None if str(args.device_map).lower() in {"", "none"} else args.device_map,
        model_dtype=args.model_dtype,
    )
    freeze_qwen(adapter)
    visual_encoder = QwenLLMLatentEncoder(57344, name="qwen_visual_tokens", model_name=args.model_name, qwen_adapter=adapter, use_vision_only=True, visual_pooling="tokens", visual_num_tokens=16, cache_latents=True)
    semantic_encoder = QwenLLMLatentEncoder(3584, name="qwen_llm_semantic", model_name=args.model_name, qwen_adapter=adapter, use_vision_only=False, visual_pooling="last", cache_latents=True)

    ds = make_subset(args.manifest, args.images_base_dir, max_samples=int(args.max_samples))
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_joint_collate_fn)
    metrics, rows = eval_ranking(loader=loader, visual_encoder=visual_encoder, semantic_encoder=semantic_encoder, wm=wm, head=head, device=device, gamma=float(args.gamma))
    summary = {"checkpoint": str(ckpt_path), "manifest": args.manifest, "split": args.split_name, "metrics": metrics}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.write_per_sample:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with (out / "per_sample.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
