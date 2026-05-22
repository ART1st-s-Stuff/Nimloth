"""Train a multi-head EB-Nav value ensemble with bootstrap + randomized priors.

This is a faster replacement for launching N independent value-head seed jobs: the
frozen encoder and frozen WM are run once per batch, then K value heads are trained
in-process.  The checkpoint keeps the trainable heads and frozen prior heads so
inference can compute calibrated mean/std on predicted next-state values.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.train_eb_nav_value_head_predicted import (  # noqa: E402
    QWEN_VISUAL_DIM,
    SemanticWMValueHead,
    build_visual_encoder,
    build_wm_from_checkpoint,
    freeze_qwen,
    make_value_collate,
    pearson_r,
    resolve_repo_path,
)
from dev.training_observability import add_observability_args, emit_metrics, init_wandb  # noqa: E402
from src.train.train_wm_joint import CustomJointSequenceDataset  # noqa: E402
from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402


class RPEValueEnsemble(nn.Module):
    """K value heads plus K frozen randomized prior heads."""

    def __init__(self, *, ensemble_size: int, semantic_dim: int, visual_dim: int, action_dim: int = 8, hidden: int = 512, prior_scale: float = 0.1) -> None:
        super().__init__()
        self.ensemble_size = int(ensemble_size)
        self.semantic_dim = int(semantic_dim)
        self.visual_dim = int(visual_dim)
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)
        self.prior_scale = float(prior_scale)
        self.heads = nn.ModuleList([
            SemanticWMValueHead(semantic_dim=semantic_dim, visual_dim=visual_dim, action_dim=action_dim, hidden=hidden)
            for _ in range(self.ensemble_size)
        ])
        self.priors = nn.ModuleList([
            SemanticWMValueHead(semantic_dim=semantic_dim, visual_dim=visual_dim, action_dim=action_dim, hidden=hidden)
            for _ in range(self.ensemble_size)
        ])
        for prior in self.priors:
            prior.eval()
            for p in prior.parameters():
                p.requires_grad = False

    def forward_members(self, semantic: torch.Tensor, z_current: torch.Tensor, z_next: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        outs = []
        for head, prior in zip(self.heads, self.priors):
            y = head(semantic, z_current, z_next, action)
            if self.prior_scale:
                with torch.no_grad():
                    p = prior(semantic, z_current, z_next, action)
                y = y + float(self.prior_scale) * p
            outs.append(y)
        return torch.stack(outs, dim=0)  # [K, B]

    def forward(self, semantic: torch.Tensor, z_current: torch.Tensor, z_next: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward_members(semantic, z_current, z_next, action).mean(dim=0)


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, *, margin: float, min_delta: float, max_pairs: int, rng: random.Random) -> torch.Tensor:
    n = int(pred.numel())
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(float(target[i] - target[j])) >= min_delta]
    if not pairs:
        return pred.new_tensor(0.0)
    if len(pairs) > max_pairs:
        pairs = rng.sample(pairs, int(max_pairs))
    ii = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=pred.device)
    jj = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=pred.device)
    sign = torch.sign(target[ii] - target[jj])
    diff = pred[ii] - pred[jj]
    return torch.relu(float(margin) - sign * diff).mean()


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
    p.add_argument("--ensemble-size", type=int, default=8)
    p.add_argument("--bootstrap-prob", type=float, default=0.8)
    p.add_argument("--prior-scale", type=float, default=0.1)
    p.add_argument("--rank-loss-weight", type=float, default=0.2)
    p.add_argument("--rank-margin", type=float, default=0.02)
    p.add_argument("--rank-min-delta", type=float, default=0.01)
    p.add_argument("--max-rank-pairs", type=int, default=256)
    p.add_argument("--output-dir", default="")
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-dtype", default="auto")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--visual-encoder", choices=["qwen", "dino"], default="qwen")
    p.add_argument("--dino-model-name", default="dinov2_vits14")
    p.add_argument("--dino-image-size", type=int, default=224)
    add_observability_args(p, default_project="flower", default_run_name="qwen-value-rpe")
    return p.parse_args()


def _make_loaders(args: argparse.Namespace, collate: Any) -> tuple[DataLoader, DataLoader, int, int]:
    train_ds_full = CustomJointSequenceDataset(args.train_manifest, args.images_base_dir, history_len=4, temporal_stride=3, action_dim=8, max_samples=0, require_prompt=False)
    test_ds_full = CustomJointSequenceDataset(args.test_manifest, args.images_base_dir, history_len=4, temporal_stride=3, action_dim=8, max_samples=0, require_prompt=False)
    train_idx = list(range(len(train_ds_full)))[: max(0, int(args.max_samples)) or len(train_ds_full)]
    test_idx = list(range(len(test_ds_full)))[: max(0, int(args.test_max_samples)) or len(test_ds_full)]
    train_loader = DataLoader(Subset(train_ds_full, train_idx), batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(Subset(test_ds_full, test_idx), batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate)
    return train_loader, test_loader, len(train_idx), len(test_idx)


def save_checkpoint(out: Path, model: RPEValueEnsemble, args: argparse.Namespace, *, best: dict[str, Any], visual_dim: int, visual_encoder: str) -> None:
    payload = {
        "ensemble_state": model.state_dict(),
        "head_states": [h.state_dict() for h in model.heads],
        "prior_states": [p.state_dict() for p in model.priors],
        "args": vars(args),
        "best": best,
        "visual_dim": int(visual_dim),
        "visual_encoder": visual_encoder,
        "semantic_dim": QWEN_VISUAL_DIM,
        "ensemble_size": int(model.ensemble_size),
        "prior_scale": float(model.prior_scale),
        "checkpoint_format": "rpe_value_ensemble_v1",
    }
    torch.save(payload, out / "best_value_ensemble.pt")
    # Compatibility export: trainable head only. The full RPE behavior requires best_value_ensemble.pt.
    for i, head in enumerate(model.heads):
        hd = out / f"head_{i}"
        hd.mkdir(parents=True, exist_ok=True)
        torch.save({
            "head_state": head.state_dict(),
            "args": vars(args),
            "best": best,
            "visual_dim": int(visual_dim),
            "visual_encoder": visual_encoder,
            "semantic_dim": QWEN_VISUAL_DIM,
            "ensemble_member": i,
            "note": "Compatibility export contains trainable head only; use best_value_ensemble.pt for randomized prior outputs.",
        }, hd / "best_value_head.pt")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if str(args.cuda_device) in {"", "-1", "cpu"} else f"cuda:{args.cuda_device}")
    out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "dev" / f"{datetime.now():%Y%m%d_%H%M%S}_value_rpe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    wandb_run = init_wandb(args, task_name="value_rpe", config=vars(args) | {"output_dir": str(out)}, output_dir=out)

    adapter = QwenVLMAdapter(model_name=args.model_name, latent_dim=16 * QWEN_VISUAL_DIM, enabled=True, fallback_enabled=False, device_map=None if str(args.device_map).lower() in {"", "none"} else args.device_map, model_dtype=args.model_dtype)
    freeze_qwen(adapter)
    visual_encoder, visual_dim, visual_latent_dim = build_visual_encoder(args, adapter)
    semantic_encoder = QwenLLMLatentEncoder(QWEN_VISUAL_DIM, name="qwen_llm_semantic", model_name=args.model_name, qwen_adapter=adapter, use_vision_only=False, visual_pooling="last", cache_latents=True)
    wm = None
    if args.next_latent_source in {"pred", "mixed"}:
        if not args.wm_checkpoint:
            raise ValueError("--wm-checkpoint is required for pred/mixed next latent source")
        wm = build_wm_from_checkpoint(resolve_repo_path(args.wm_checkpoint), device=device, visual_dim=visual_dim, latent_dim=visual_latent_dim)
    collate = make_value_collate(visual_encoder=visual_encoder, semantic_encoder=semantic_encoder, wm=wm, device=device, visual_dim=visual_dim, gamma=float(args.gamma), next_latent_source=args.next_latent_source, mixed_pred_prob=float(args.mixed_pred_prob), rng=rng)
    train_loader, test_loader, num_train_items, num_test_items = _make_loaders(args, collate)

    model = RPEValueEnsemble(ensemble_size=int(args.ensemble_size), semantic_dim=QWEN_VISUAL_DIM, visual_dim=visual_dim, action_dim=8, hidden=int(args.hidden), prior_scale=float(args.prior_scale)).to(device)
    opt = torch.optim.AdamW(model.heads.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    reg_loss_fn = nn.SmoothL1Loss(reduction="none")
    best = {"test_mse": float("inf"), "epoch": -1}
    step_log_fields = ["global_step", "epoch", "batch_idx", "train_loss", "train_reg_loss", "train_rank_loss", "batch_size", "seen"]
    log_fields = ["epoch", "train_loss", "train_reg_loss", "train_rank_loss", "test_loss", "test_mse", "test_pearson", "test_ensemble_std", "num_train", "num_test"]
    step_log = (out / "train_step_log.csv").open("w", newline="", encoding="utf-8")
    step_writer = csv.DictWriter(step_log, fieldnames=step_log_fields); step_writer.writeheader(); step_log.flush()
    global_step = 0
    with (out / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields); writer.writeheader()
        try:
          for epoch in range(1, int(args.epochs) + 1):
            model.train(); train_sum = train_reg_sum = train_rank_sum = 0.0; train_n = 0
            for batch_idx, batch in enumerate(train_loader, start=1):
                preds = model.forward_members(batch.semantic, batch.z_current, batch.z_next, batch.action)  # [K,B]
                target = batch.value
                reg_each = reg_loss_fn(preds, target.unsqueeze(0).expand_as(preds))
                mask = (torch.rand_like(reg_each) < float(args.bootstrap_prob)).float()
                # Ensure every sample contributes at least through ensemble on tiny batches.
                denom = mask.sum().clamp_min(1.0)
                reg_loss = (reg_each * mask).sum() / denom
                rank_losses = []
                for k in range(preds.size(0)):
                    rank_losses.append(pairwise_rank_loss(preds[k], target, margin=float(args.rank_margin), min_delta=float(args.rank_min_delta), max_pairs=int(args.max_rank_pairs), rng=rng))
                rank_loss = torch.stack(rank_losses).mean() if rank_losses else preds.new_tensor(0.0)
                loss = reg_loss + float(args.rank_loss_weight) * rank_loss
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                bsz = int(target.numel())
                train_sum += float(loss.item()) * bsz; train_reg_sum += float(reg_loss.item()) * bsz; train_rank_sum += float(rank_loss.item()) * bsz; train_n += bsz
                global_step += 1
                step_row = {"global_step": global_step, "epoch": epoch, "batch_idx": batch_idx, "train_loss": float(loss.item()), "train_reg_loss": float(reg_loss.item()), "train_rank_loss": float(rank_loss.item()), "batch_size": bsz, "seen": train_n}
                step_writer.writerow(step_row); step_log.flush()
                if int(args.log_every_steps) <= 1 or global_step % int(args.log_every_steps) == 0:
                    emit_metrics(step_row, wandb_run=wandb_run, step=global_step, prefix="train_step/")
            model.eval(); preds_mean=[]; targets=[]; test_loss=0.0; test_mse=0.0; test_std_sum=0.0; test_n=0
            with torch.no_grad():
                for batch in test_loader:
                    members = model.forward_members(batch.semantic, batch.z_current, batch.z_next, batch.action)
                    mean = members.mean(dim=0)
                    std = members.std(dim=0, unbiased=False)
                    loss_vec = reg_loss_fn(mean, batch.value)
                    mse = nn.functional.mse_loss(mean, batch.value, reduction="none")
                    n = int(batch.value.numel())
                    test_loss += float(loss_vec.mean().item()) * n; test_mse += float(mse.mean().item()) * n; test_std_sum += float(std.mean().item()) * n; test_n += n
                    preds_mean.extend(float(x) for x in mean.detach().cpu()); targets.extend(float(x) for x in batch.value.detach().cpu())
            row = {
                "epoch": epoch,
                "train_loss": train_sum / max(1, train_n),
                "train_reg_loss": train_reg_sum / max(1, train_n),
                "train_rank_loss": train_rank_sum / max(1, train_n),
                "test_loss": test_loss / max(1, test_n),
                "test_mse": test_mse / max(1, test_n),
                "test_pearson": pearson_r(preds_mean, targets),
                "test_ensemble_std": test_std_sum / max(1, test_n),
                "num_train": train_n,
                "num_test": test_n,
            }
            writer.writerow(row); f.flush(); emit_metrics(row | {"global_step": global_step}, wandb_run=wandb_run, step=global_step, prefix="epoch/")
            if float(row["test_mse"]) < float(best["test_mse"]):
                best = {"test_mse": float(row["test_mse"]), "epoch": epoch}
                save_checkpoint(out, model, args, best=best, visual_dim=visual_dim, visual_encoder=args.visual_encoder)
        finally:
            step_log.close()
    summary = {"best": best, "args": vars(args), "visual_dim": visual_dim, "visual_encoder": args.visual_encoder, "semantic_dim": QWEN_VISUAL_DIM, "ensemble_size": int(args.ensemble_size), "num_train_items": num_train_items, "num_test_items": num_test_items, "outputs": {"best_value_ensemble": str(out / "best_value_ensemble.pt"), "train_log": str(out / "train_log.csv")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.summary.update({"best_test_mse": best["test_mse"], "best_epoch": best["epoch"], "output_dir": str(out)})
        wandb_run.finish()


if __name__ == "__main__":
    main()
