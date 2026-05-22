"""Small resumable WM predictor fine-tune/eval on EB-Nav fork samples.

Each fork sample provides a same-state history, a candidate action, and the
saved first counterfactual observation image.  This script trains the existing
WM predictor to map (history, candidate action) -> first fork observation
latent.  It is intended as a diagnostic smoke run for fork data quality, not a
replacement for the main WM training pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.train_eb_nav_value_head_predicted import (  # noqa: E402
    NUM_PATCHES,
    QWEN_VISUAL_DIM,
    build_visual_encoder,
    build_wm_from_checkpoint,
    encode_many,
    freeze_qwen,
    resolve_repo_path,
)
from dev.training_observability import add_observability_args, emit_metrics, init_wandb  # noqa: E402
from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402


@dataclass
class ForkWMBatch:
    z_hist: torch.Tensor
    teacher_action: torch.Tensor
    target_z: torch.Tensor
    sample_weight: torch.Tensor


def one_hot(action_id: int, *, device: torch.device) -> torch.Tensor:
    out = torch.zeros(8, dtype=torch.float32, device=device)
    if 0 <= int(action_id) < 8:
        out[int(action_id)] = 1.0
    return out


def target_image_for(row: dict[str, Any]) -> str:
    image_t = Path(str(row["image_t"]))
    rollout_id = int(row.get("rollout_id", 0))
    step = int(row.get("step", 0))
    action_id = int(row.get("candidate_action_id", row.get("action_id", 0)))
    return str(image_t.parent / f"rollout_{rollout_id:04d}_step_{step:03d}_fork_a{action_id}_first.png")


class ForkWMDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path, *, max_samples: int = 0) -> None:
        rows: list[dict[str, Any]] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("skipped"):
                continue
            if row.get("skip_for_training"):
                continue
            if "history_images" not in row or "candidate_action_id" not in row or "image_t" not in row:
                continue
            target = resolve_repo_path(target_image_for(row))
            if not target.exists():
                continue
            row = dict(row)
            row["target_image"] = str(target)
            rows.append(row)
        self.rows = rows[: int(max_samples)] if int(max_samples) > 0 else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def make_collate(*, visual_encoder: Any, device: torch.device, visual_dim: int) -> Any:
    def collate(items: list[dict[str, Any]]) -> ForkWMBatch:
        bsz = len(items)
        histories = [list(r["history_images"]) for r in items]
        hist_len = len(histories[0])
        flat_hist = [p for seq in histories for p in seq]
        z_hist = encode_many(visual_encoder, flat_hist, None, device).reshape(bsz, hist_len, NUM_PATCHES, visual_dim)
        target_z = encode_many(visual_encoder, [str(r["target_image"]) for r in items], None, device).reshape(bsz, NUM_PATCHES, visual_dim)
        teacher_action = torch.zeros(bsz, hist_len, 8, dtype=torch.float32, device=device)
        for i, row in enumerate(items):
            hist_actions = list(row.get("history_actions") or [-1] * hist_len)[-hist_len:]
            hist_actions = ([-1] * (hist_len - len(hist_actions))) + hist_actions
            for t, aid in enumerate(hist_actions):
                if 0 <= int(aid) < 8:
                    teacher_action[i, t, int(aid)] = 1.0
            teacher_action[i, -1, :] = one_hot(int(row["candidate_action_id"]), device=device)
        sample_weight = torch.tensor(
            [max(0.0, float(r.get("effective_lr_scale", 1.0))) for r in items],
            dtype=torch.float32,
            device=device,
        )
        return ForkWMBatch(z_hist=z_hist, teacher_action=teacher_action, target_z=target_z, sample_weight=sample_weight)
    return collate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fork-jsonl", required=True)
    p.add_argument("--wm-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min-effective-lr-scale", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--save-every-steps", type=int, default=20)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-dtype", default="auto")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--visual-encoder", choices=["qwen", "dino"], default="qwen")
    p.add_argument("--dino-model-name", default="dinov2_vits14")
    p.add_argument("--dino-image-size", type=int, default=224)
    add_observability_args(p, default_project="flower", default_run_name="fork-wm-predictor-smoke")
    return p.parse_args()


def latest_checkpoint(out: Path) -> Path | None:
    ckpts = sorted((out / "checkpoints").glob("checkpoint_step_*.pt"))
    return ckpts[-1] if ckpts else None


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed)); torch.manual_seed(int(args.seed))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); (out / "checkpoints").mkdir(exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    wandb_run = init_wandb(args, task_name="fork_wm_predictor", config=vars(args) | {"output_dir": str(out)}, output_dir=out)
    device = torch.device("cpu" if str(args.cuda_device) in {"", "-1", "cpu"} else f"cuda:{args.cuda_device}")

    adapter = QwenVLMAdapter(model_name=args.model_name, latent_dim=NUM_PATCHES * QWEN_VISUAL_DIM, enabled=True, fallback_enabled=False, device_map=None if str(args.device_map).lower() in {"", "none"} else args.device_map, model_dtype=args.model_dtype)
    freeze_qwen(adapter)
    visual_encoder, visual_dim, visual_latent_dim = build_visual_encoder(args, adapter)
    wm = build_wm_from_checkpoint(resolve_repo_path(args.wm_checkpoint), device=device, visual_dim=visual_dim, latent_dim=visual_latent_dim)
    wm.train()
    for param in wm.parameters():
        param.requires_grad = True

    ds = ForkWMDataset(resolve_repo_path(args.fork_jsonl), max_samples=int(args.max_samples))
    if float(args.min_effective_lr_scale) > 0.0:
        ds.rows = [r for r in ds.rows if float(r.get("effective_lr_scale", 1.0)) >= float(args.min_effective_lr_scale)]
    if len(ds) < 2:
        raise RuntimeError(f"not enough fork WM samples: {len(ds)}")
    test_n = max(1, int(len(ds) * float(args.test_ratio))); train_n = max(1, len(ds) - test_n)
    train_ds, test_ds = random_split(ds, [train_n, len(ds) - train_n], generator=torch.Generator().manual_seed(int(args.seed)))
    collate = make_collate(visual_encoder=visual_encoder, device=device, visual_dim=visual_dim)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate)
    opt = torch.optim.AdamW(wm.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    mse = nn.MSELoss()
    start_epoch = 1; global_step = 0; best = {"test_mse": float("inf"), "epoch": -1}
    if bool(args.resume):
        ckpt = latest_checkpoint(out)
        if ckpt is not None:
            state = torch.load(ckpt, map_location=device)
            wm.load_state_dict(state["wm_state"], strict=False); opt.load_state_dict(state["optimizer_state"])
            start_epoch = int(state.get("epoch", 0)) + 1; global_step = int(state.get("global_step", 0)); best = dict(state.get("best", best))
            print(json.dumps({"resume": True, "checkpoint": str(ckpt), "start_epoch": start_epoch, "global_step": global_step}), flush=True)

    step_log = (out / "train_step_log.csv").open("a", newline="", encoding="utf-8")
    step_fields = ["global_step", "epoch", "batch_idx", "train_loss", "batch_size"]
    step_writer = csv.DictWriter(step_log, fieldnames=step_fields)
    if step_log.tell() == 0:
        step_writer.writeheader(); step_log.flush()
    log_path = out / "train_log.csv"
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "test_mse", "num_train", "num_test", "global_step"])
        if write_header:
            writer.writeheader(); f.flush()
        try:
            for epoch in range(start_epoch, int(args.epochs) + 1):
                wm.train(); total = 0.0; n_total = 0
                for batch_idx, batch in enumerate(train_loader, start=1):
                    if hasattr(wm, "predict_next_ensemble"):
                        pred_members = wm.predict_next_ensemble(batch.z_hist, batch.teacher_action)
                        loss_each = (pred_members - batch.target_z.unsqueeze(0).expand_as(pred_members)).pow(2).flatten(2).mean(dim=2)
                        loss = (loss_each * batch.sample_weight.unsqueeze(0)).sum() / batch.sample_weight.sum().clamp_min(1e-6) / pred_members.size(0)
                    else:
                        loss_each = (wm.predict_next(batch.z_hist, batch.teacher_action) - batch.target_z).pow(2).flatten(1).mean(dim=1)
                        loss = (loss_each * batch.sample_weight).sum() / batch.sample_weight.sum().clamp_min(1e-6)
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    bsz = int(batch.target_z.size(0)); total += float(loss.item()) * bsz; n_total += bsz; global_step += 1
                    row = {"global_step": global_step, "epoch": epoch, "batch_idx": batch_idx, "train_loss": float(loss.item()), "batch_size": bsz}
                    step_writer.writerow(row); step_log.flush(); emit_metrics(row, wandb_run=wandb_run, step=global_step, prefix="train_step/")
                    if int(args.save_every_steps) > 0 and global_step % int(args.save_every_steps) == 0:
                        torch.save({"wm_state": wm.state_dict(), "optimizer_state": opt.state_dict(), "epoch": epoch, "global_step": global_step, "best": best, "args": vars(args)}, out / "checkpoints" / f"checkpoint_step_{global_step:08d}.pt")
                wm.eval(); test_loss = 0.0; test_n = 0
                with torch.no_grad():
                    for batch in test_loader:
                        pred = wm.predict_next(batch.z_hist, batch.teacher_action)
                        loss = mse(pred, batch.target_z); bsz = int(batch.target_z.size(0))
                        test_loss += float(loss.item()) * bsz; test_n += bsz
                epoch_row = {"epoch": epoch, "train_loss": total / max(1, n_total), "test_mse": test_loss / max(1, test_n), "num_train": n_total, "num_test": test_n, "global_step": global_step}
                writer.writerow(epoch_row); f.flush(); emit_metrics(epoch_row, wandb_run=wandb_run, step=global_step, prefix="epoch/")
                if float(epoch_row["test_mse"]) < float(best["test_mse"]):
                    best = {"test_mse": float(epoch_row["test_mse"]), "epoch": epoch}
                    torch.save({"wm_state": wm.state_dict(), "args": vars(args), "best": best, "visual_dim": visual_dim, "visual_encoder": args.visual_encoder}, out / "best_wm.pt")
                torch.save({"wm_state": wm.state_dict(), "optimizer_state": opt.state_dict(), "epoch": epoch, "global_step": global_step, "best": best, "args": vars(args)}, out / "checkpoints" / f"checkpoint_step_{global_step:08d}.pt")
        finally:
            step_log.close()
    (out / "summary.json").write_text(json.dumps({"best": best, "num_samples": len(ds), "args": vars(args)}, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.summary.update({"best_test_mse": best["test_mse"], "best_epoch": best["epoch"], "output_dir": str(out)})
        wandb_run.finish()


if __name__ == "__main__":
    main()
