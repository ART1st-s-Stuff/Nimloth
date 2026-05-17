"""Train frozen-Qwen/frozen-WM semantic action-ranking head for EB-Nav.

For each state, enumerate 8 actions, predict next visual latent with frozen WM, score each
candidate with SemanticWMValueHead, and train CE against logged/expert future_action_ids[0].
Optionally keeps scalar return regression on the logged action as an auxiliary loss.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.eval_eb_nav_value_action_ranking import eval_ranking, one_hot, discounted_return  # noqa: E402
from dev.train_eb_nav_value_head_predicted import (  # noqa: E402
    SemanticWMValueHead,
    build_wm_from_checkpoint,
    encode_many,
    resolve_repo_path,
)
from dev.train_eb_nav_joint_wm_value import freeze_qwen, make_subset  # noqa: E402
from src.train.train_wm_joint import _joint_collate_fn  # noqa: E402
from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402


def load_head_init(head: SemanticWMValueHead, path: str) -> None:
    if not path:
        return
    ckpt = torch.load(resolve_repo_path(path), map_location="cpu")
    state = ckpt.get("head_state", ckpt)
    missing, unexpected = head.load_state_dict(state, strict=False)
    print(f"loaded_head_init={path} missing={missing[:5]} unexpected={unexpected[:5]}", flush=True)


def subset(manifest: str, images_base_dir: str, max_samples: int) -> Subset:
    ds = make_subset(manifest, images_base_dir, max_samples=max_samples)
    return ds


def encode_scores(
    raw: dict[str, Any],
    *,
    visual_encoder: QwenLLMLatentEncoder,
    semantic_encoder: QwenLLMLatentEncoder,
    wm: torch.nn.Module,
    head: SemanticWMValueHead,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history_images: list[list[str]] = raw["history_images"]
    bsz = len(history_images)
    prompts = [str(x or "") for x in raw.get("prompts", raw.get("instructions", [""] * bsz))]
    hist_actions = raw["history_actions"].float().to(device)
    if "future_action_ids" not in raw:
        raise RuntimeError("future_action_ids required")
    labels = raw["future_action_ids"][:, 0].long().to(device).clamp(0, 7)
    rewards = raw.get("future_rewards")
    if rewards is None:
        raise RuntimeError("future_rewards required")
    target_return = discounted_return(rewards.float().to(device), float(ARGS.gamma))

    flat_hist = [p for seq in history_images for p in seq]
    z_hist = encode_many(visual_encoder, flat_hist, None, device).reshape(bsz, len(history_images[0]), 16, 3584)
    z_current = z_hist[:, -1]
    semantic = encode_many(
        semantic_encoder,
        [seq[-1] for seq in history_images],
        prompts,
        device,
        expected_flat_dim=3584,
    ).reshape(bsz, 3584)

    cols: list[torch.Tensor] = []
    for action_id in range(8):
        ids = torch.full((bsz,), action_id, dtype=torch.long, device=device)
        action_vec = one_hot(ids, 8)
        teacher_action = hist_actions.clone()
        teacher_action[:, -1, :] = action_vec
        with torch.no_grad():
            z_next = wm.predict_next(z_hist, teacher_action)
        cols.append(head(semantic, z_current, z_next, action_vec))
    return torch.stack(cols, dim=1), labels, target_return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--rollout-test-manifest", default="")
    p.add_argument("--images-base-dir", default=".")
    p.add_argument("--wm-checkpoint", required=True)
    p.add_argument("--head-init", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=8192)
    p.add_argument("--test-max-samples", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--value-aux-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-dtype", default="auto")
    p.add_argument("--device-map", default="auto")
    return p.parse_args()


ARGS: argparse.Namespace


def main() -> None:
    global ARGS
    ARGS = parse_args()
    random.seed(int(ARGS.seed)); torch.manual_seed(int(ARGS.seed))
    device = torch.device("cpu" if str(ARGS.cuda_device) in {"", "-1", "cpu"} else f"cuda:{ARGS.cuda_device}")
    out = Path(ARGS.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(ARGS), indent=2))

    adapter = QwenVLMAdapter(
        model_name=ARGS.model_name,
        latent_dim=57344,
        enabled=True,
        fallback_enabled=False,
        device_map=None if str(ARGS.device_map).lower() in {"", "none"} else ARGS.device_map,
        model_dtype=ARGS.model_dtype,
    )
    freeze_qwen(adapter)
    visual_encoder = QwenLLMLatentEncoder(57344, name="qwen_visual_tokens", model_name=ARGS.model_name, qwen_adapter=adapter, use_vision_only=True, visual_pooling="tokens", visual_num_tokens=16, cache_latents=True)
    semantic_encoder = QwenLLMLatentEncoder(3584, name="qwen_llm_semantic", model_name=ARGS.model_name, qwen_adapter=adapter, use_vision_only=False, visual_pooling="last", cache_latents=True)
    wm = build_wm_from_checkpoint(resolve_repo_path(ARGS.wm_checkpoint), device)
    wm.eval()
    head = SemanticWMValueHead(semantic_dim=3584, visual_dim=3584, action_dim=8, hidden=512).to(device)
    load_head_init(head, ARGS.head_init)
    opt = torch.optim.AdamW(head.parameters(), lr=float(ARGS.lr), weight_decay=float(ARGS.weight_decay))

    train_ds = subset(ARGS.train_manifest, ARGS.images_base_dir, int(ARGS.max_samples))
    test_ds = subset(ARGS.test_manifest, ARGS.images_base_dir, int(ARGS.test_max_samples))
    train_loader = DataLoader(train_ds, batch_size=int(ARGS.batch_size), shuffle=True, num_workers=0, collate_fn=_joint_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=int(ARGS.batch_size), shuffle=False, num_workers=0, collate_fn=_joint_collate_fn)
    rollout_loader = None
    if ARGS.rollout_test_manifest:
        rollout_ds = subset(ARGS.rollout_test_manifest, ARGS.images_base_dir, int(ARGS.test_max_samples))
        rollout_loader = DataLoader(rollout_ds, batch_size=int(ARGS.batch_size), shuffle=False, num_workers=0, collate_fn=_joint_collate_fn)

    rows=[]; best=None
    for epoch in range(1, int(ARGS.epochs)+1):
        head.train(); loss_sum=ce_sum=aux_sum=0.0; n=0; correct=0
        for raw in train_loader:
            logits, labels, returns = encode_scores(raw, visual_encoder=visual_encoder, semantic_encoder=semantic_encoder, wm=wm, head=head, device=device)
            ce = F.cross_entropy(logits, labels)
            logged_score = logits.gather(1, labels[:, None]).squeeze(1)
            aux = F.smooth_l1_loss(logged_score, returns)
            loss = ce + float(ARGS.value_aux_weight) * aux
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            b = int(labels.numel())
            loss_sum += float(loss.item())*b; ce_sum += float(ce.item())*b; aux_sum += float(aux.item())*b; n += b
            correct += int((logits.argmax(dim=1) == labels).sum().item())
        train_metrics = {"train_loss": loss_sum/max(1,n), "train_ce": ce_sum/max(1,n), "train_value_aux": aux_sum/max(1,n), "train_top1": correct/max(1,n), "train_n": n}
        test_metrics, _ = eval_ranking(loader=test_loader, visual_encoder=visual_encoder, semantic_encoder=semantic_encoder, wm=wm, head=head, device=device, gamma=float(ARGS.gamma))
        row = {"epoch": epoch, **train_metrics, **{f"test_{k}": v for k,v in test_metrics.items() if k != "per_action"}}
        if rollout_loader is not None:
            rm, _ = eval_ranking(loader=rollout_loader, visual_encoder=visual_encoder, semantic_encoder=semantic_encoder, wm=wm, head=head, device=device, gamma=float(ARGS.gamma))
            row.update({f"rollout_{k}": v for k,v in rm.items() if k != "per_action"})
        rows.append(row)
        print(json.dumps(row), flush=True)
        score = float(test_metrics.get("mrr", 0.0))
        if best is None or score > float(best["test_mrr"]):
            best = {"epoch": epoch, "test_mrr": score, "test_top1": test_metrics.get("top1_acc"), "test_macro_top1": test_metrics.get("macro_top1_acc")}
            torch.save({"wm_state": wm.state_dict(), "head_state": head.state_dict(), "args": vars(ARGS), "best": best}, out / "best_action_ranking_head.pt")
    torch.save({"wm_state": wm.state_dict(), "head_state": head.state_dict(), "args": vars(ARGS), "best": best}, out / "final_action_ranking_head.pt")
    (out / "summary.json").write_text(json.dumps({"best": best, "last": rows[-1] if rows else None}, indent=2))
    with (out / "train_log.csv").open("w", newline="") as f:
        fields = sorted({k for r in rows for k in r.keys()})
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
