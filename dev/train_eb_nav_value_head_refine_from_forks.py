"""Refine an RPE value ensemble on uncertainty-fork counterfactual outcomes."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.train_eb_nav_value_head_ensemble_rpe import RPEValueEnsemble, pairwise_rank_loss  # noqa: E402
from dev.train_eb_nav_value_head_predicted import (  # noqa: E402
    NUM_PATCHES,
    QWEN_VISUAL_DIM,
    build_visual_encoder,
    build_wm_from_checkpoint,
    encode_many,
    freeze_qwen,
    pearson_r,
    resolve_repo_path,
)
from dev.training_observability import add_observability_args, emit_metrics, init_wandb  # noqa: E402
from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402


@dataclass
class ForkBatch:
    semantic: torch.Tensor
    z_current: torch.Tensor
    z_next: torch.Tensor
    action: torch.Tensor
    value: torch.Tensor
    sample_weight: torch.Tensor
    group_keys: list[str]


def group_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("rollout_id", "")),
            str(row.get("episode_id", "")),
            str(row.get("eval_set", "")),
            str(row.get("task_key", "")),
            str(row.get("step", "")),
        ]
    )


class ForkDataset(Dataset[dict[str, Any]]):
    def __init__(self, paths: list[str | Path], *, max_samples: int = 0) -> None:
        rows = []
        for path in expand_fork_jsonl(paths):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("skipped"):
                    continue
                if r.get("skip_for_training"):
                    continue
                if "history_images" not in r or "candidate_action_id" not in r:
                    continue
                rows.append(r)
        self.rows = rows[: int(max_samples)] if int(max_samples) > 0 else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def expand_fork_jsonl(inputs: list[str | Path]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        raw = str(item)
        p = resolve_repo_path(raw)
        if p.is_dir():
            out.extend(sorted(p.glob("**/fork_samples.jsonl")))
            continue
        matches = sorted(glob.glob(str(p)))
        if matches:
            out.extend(Path(m) for m in matches)
        else:
            out.append(p)
    seen: set[str] = set()
    uniq: list[Path] = []
    for path in out:
        key = str(path)
        if key not in seen:
            seen.add(key)
            uniq.append(path)
    return uniq


def one_hot_action(action_id: int, *, device: torch.device) -> torch.Tensor:
    out = torch.zeros(8, dtype=torch.float32, device=device)
    if 0 <= int(action_id) < 8:
        out[int(action_id)] = 1.0
    return out


def make_collate(*, visual_encoder: Any, semantic_encoder: QwenLLMLatentEncoder, wm: torch.nn.Module, device: torch.device, visual_dim: int, target_field: str, target_scale: float) -> Any:
    def collate(items: list[dict[str, Any]]) -> ForkBatch:
        bsz = len(items)
        histories = [list(r["history_images"]) for r in items]
        hist_len = len(histories[0])
        flat = [p for seq in histories for p in seq]
        z_hist = encode_many(visual_encoder, flat, None, device).reshape(bsz, hist_len, NUM_PATCHES, visual_dim)
        z_current = z_hist[:, -1]
        prompts = [str(r.get("instruction") or r.get("prompt") or "") for r in items]
        semantic = encode_many(semantic_encoder, [seq[-1] for seq in histories], prompts, device, expected_flat_dim=QWEN_VISUAL_DIM).reshape(bsz, QWEN_VISUAL_DIM)
        teacher_action = torch.zeros(bsz, hist_len, 8, dtype=torch.float32, device=device)
        for i, r in enumerate(items):
            hist_actions = list(r.get("history_actions") or [-1] * hist_len)[-hist_len:]
            hist_actions = ([-1] * (hist_len - len(hist_actions))) + hist_actions
            for t, aid in enumerate(hist_actions):
                if 0 <= int(aid) < 8:
                    teacher_action[i, t, int(aid)] = 1.0
            teacher_action[i, -1, :] = one_hot_action(int(r["candidate_action_id"]), device=device)
        z_next = wm.predict_next(z_hist, teacher_action).detach()
        action = torch.stack([one_hot_action(int(r["candidate_action_id"]), device=device) for r in items], dim=0)
        value = torch.tensor([float(r.get(target_field, 0.0)) * float(target_scale) for r in items], dtype=torch.float32, device=device)
        sample_weight = torch.tensor([max(0.0, float(r.get("effective_lr_scale", 1.0))) for r in items], dtype=torch.float32, device=device)
        group_keys = [
            group_key(r)
            for r in items
        ]
        return ForkBatch(semantic=semantic, z_current=z_current, z_next=z_next, action=action, value=value, sample_weight=sample_weight, group_keys=group_keys)
    return collate


def grouped_pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, group_keys: list[str], *, margin: float, min_delta: float, max_pairs: int, rng: random.Random) -> torch.Tensor:
    losses = []
    for key in sorted(set(group_keys)):
        idx = [i for i, k in enumerate(group_keys) if k == key]
        if len(idx) < 2:
            continue
        ids = torch.tensor(idx, dtype=torch.long, device=pred.device)
        losses.append(pairwise_rank_loss(pred[ids], target[ids], margin=margin, min_delta=min_delta, max_pairs=max_pairs, rng=rng))
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def grouped_batch_indices(rows: list[dict[str, Any]], *, test_ratio: float, batch_size: int, seed: int) -> tuple[list[list[int]], list[list[int]]]:
    rng = random.Random(int(seed))
    by_group: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_group.setdefault(group_key(row), []).append(idx)
    groups = list(by_group.items())
    rng.shuffle(groups)
    if len(groups) < 2:
        return _pack_groups(groups, batch_size=batch_size), _pack_groups([], batch_size=batch_size)
    test_group_count = max(1, min(len(groups) - 1, int(round(len(groups) * float(test_ratio)))))
    test_groups = groups[:test_group_count]
    train_groups = groups[test_group_count:]
    return _pack_groups(train_groups, batch_size=batch_size), _pack_groups(test_groups, batch_size=batch_size)


def _pack_groups(groups: list[tuple[str, list[int]]], *, batch_size: int) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    limit = max(1, int(batch_size))
    for _, indices in groups:
        if current and len(current) + len(indices) > limit:
            batches.append(current)
            current = []
        current.extend(indices)
    if current:
        batches.append(current)
    return batches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fork-jsonl", nargs="+", required=True, help="fork_samples.jsonl files, globs, or directories.")
    p.add_argument("--init-checkpoint", required=True)
    p.add_argument("--wm-checkpoint", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--target-field", default="continuation_reward")
    p.add_argument("--target-scale", type=float, default=1.0)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-effective-lr-scale", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--bootstrap-prob", type=float, default=0.8)
    p.add_argument("--rank-loss-weight", type=float, default=0.5)
    p.add_argument("--rank-margin", type=float, default=0.02)
    p.add_argument("--rank-min-delta", type=float, default=0.01)
    p.add_argument("--max-rank-pairs", type=int, default=256)
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
    add_observability_args(p, default_project="flower", default_run_name="fork-refine")
    return p.parse_args()


def latest_checkpoint(out: Path) -> Path | None:
    ckpts = sorted((out / "checkpoints").glob("checkpoint_step_*.pt"))
    return ckpts[-1] if ckpts else None


def main() -> None:
    args = parse_args(); rng = random.Random(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cpu" if str(args.cuda_device) in {"", "-1", "cpu"} else f"cuda:{args.cuda_device}")
    out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "dev" / f"{datetime.now():%Y%m%d_%H%M%S}_fork_refine"
    out.mkdir(parents=True, exist_ok=True); (out / "checkpoints").mkdir(exist_ok=True); (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    wandb_run = init_wandb(args, task_name="fork_refine", config=vars(args) | {"output_dir": str(out)}, output_dir=out)
    ck = torch.load(resolve_repo_path(args.init_checkpoint), map_location="cpu")
    visual_dim = int(ck.get("visual_dim", 3584 if args.visual_encoder == "qwen" else 384))
    model = RPEValueEnsemble(ensemble_size=int(ck.get("ensemble_size", 8)), semantic_dim=int(ck.get("semantic_dim", QWEN_VISUAL_DIM)), visual_dim=visual_dim, action_dim=8, hidden=int(ck.get("args", {}).get("hidden", 512)), prior_scale=float(ck.get("prior_scale", ck.get("args", {}).get("prior_scale", 0.1)))).to(device)
    model.load_state_dict(ck["ensemble_state"], strict=True)
    adapter = QwenVLMAdapter(model_name=args.model_name, latent_dim=NUM_PATCHES * QWEN_VISUAL_DIM, enabled=True, fallback_enabled=False, device_map=None if str(args.device_map).lower() in {"", "none"} else args.device_map, model_dtype=args.model_dtype)
    freeze_qwen(adapter)
    visual_encoder, _, visual_latent_dim = build_visual_encoder(args, adapter)
    semantic_encoder = QwenLLMLatentEncoder(QWEN_VISUAL_DIM, name="qwen_llm_semantic", model_name=args.model_name, qwen_adapter=adapter, use_vision_only=False, visual_pooling="last", cache_latents=True)
    wm = build_wm_from_checkpoint(resolve_repo_path(args.wm_checkpoint), device=device, visual_dim=visual_dim, latent_dim=visual_latent_dim)
    ds = ForkDataset(args.fork_jsonl, max_samples=int(args.max_samples))
    if float(args.min_effective_lr_scale) > 0.0:
        ds.rows = [r for r in ds.rows if float(r.get("effective_lr_scale", 1.0)) >= float(args.min_effective_lr_scale)]
    if len(ds) < 2:
        raise RuntimeError(f"not enough fork samples: {len(ds)}")
    collate = make_collate(visual_encoder=visual_encoder, semantic_encoder=semantic_encoder, wm=wm, device=device, visual_dim=visual_dim, target_field=args.target_field, target_scale=float(args.target_scale))
    train_batches, test_batches = grouped_batch_indices(
        ds.rows,
        test_ratio=float(args.test_ratio),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    if not train_batches:
        raise RuntimeError("not enough fork groups for train split")
    train_loader = DataLoader(ds, batch_sampler=train_batches, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(ds, batch_sampler=test_batches, num_workers=0, collate_fn=collate)
    opt = torch.optim.AdamW(model.heads.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    reg_loss_fn = nn.SmoothL1Loss(reduction="none")
    best = {"test_mse": float("inf"), "epoch": -1}
    resume_epoch = 1; resume_batch_idx = 0; global_step = 0
    if bool(args.resume):
        resume_ckpt = latest_checkpoint(out)
        if resume_ckpt is not None:
            state = torch.load(resume_ckpt, map_location=device)
            model.load_state_dict(state["ensemble_state"], strict=True)
            opt.load_state_dict(state["optimizer_state"])
            resume_epoch = int(state.get("epoch", 1))
            resume_batch_idx = int(state.get("batch_idx", 0))
            global_step = int(state.get("global_step", 0))
            best = dict(state.get("best", best))
            print(json.dumps({"resume": True, "checkpoint": str(resume_ckpt), "epoch": resume_epoch, "batch_idx": resume_batch_idx, "global_step": global_step}), flush=True)
    fields = ["epoch", "train_loss", "train_reg_loss", "train_rank_loss", "test_loss", "test_mse", "test_pearson", "test_ensemble_std", "num_train", "num_test"]
    step_log = (out / "train_step_log.csv").open("a", newline="", encoding="utf-8")
    step_writer = csv.DictWriter(step_log, fieldnames=["global_step", "epoch", "batch_idx", "train_loss", "train_reg_loss", "train_rank_loss", "batch_size", "seen"])
    if step_log.tell() == 0:
        step_writer.writeheader(); step_log.flush()
    log_path = out / "train_log.csv"
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader(); f.flush()
        try:
          for epoch in range(resume_epoch, int(args.epochs) + 1):
            model.train(); sums = {"loss": 0.0, "reg": 0.0, "rank": 0.0, "n": 0}
            for batch_idx, batch in enumerate(train_loader, start=1):
                if epoch == resume_epoch and batch_idx <= resume_batch_idx:
                    continue
                members = model.forward_members(batch.semantic, batch.z_current, batch.z_next, batch.action)
                reg_each = reg_loss_fn(members, batch.value.unsqueeze(0).expand_as(members))
                mask = (torch.rand_like(reg_each) < float(args.bootstrap_prob)).float()
                weighted_mask = mask * batch.sample_weight.unsqueeze(0)
                reg = (reg_each * weighted_mask).sum() / weighted_mask.sum().clamp_min(1e-6)
                ranks = [
                    grouped_pairwise_rank_loss(
                        members[k],
                        batch.value,
                        batch.group_keys,
                        margin=float(args.rank_margin),
                        min_delta=float(args.rank_min_delta),
                        max_pairs=int(args.max_rank_pairs),
                        rng=rng,
                    )
                    for k in range(members.size(0))
                ]
                rank = torch.stack(ranks).mean(); loss = reg + float(args.rank_loss_weight) * rank
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                n = int(batch.value.numel()); sums["loss"] += float(loss.item()) * n; sums["reg"] += float(reg.item()) * n; sums["rank"] += float(rank.item()) * n; sums["n"] += n; global_step += 1
                step_row = {"global_step": global_step, "epoch": epoch, "batch_idx": batch_idx, "train_loss": float(loss.item()), "train_reg_loss": float(reg.item()), "train_rank_loss": float(rank.item()), "batch_size": n, "seen": int(sums["n"])}
                step_writer.writerow(step_row); step_log.flush()
                if int(args.log_every_steps) <= 1 or global_step % int(args.log_every_steps) == 0:
                    emit_metrics(step_row, wandb_run=wandb_run, step=global_step, prefix="train_step/")
                if int(args.save_every_steps) > 0 and global_step % int(args.save_every_steps) == 0:
                    torch.save({"ensemble_state": model.state_dict(), "optimizer_state": opt.state_dict(), "epoch": epoch, "batch_idx": batch_idx, "global_step": global_step, "best": best, "args": vars(args)}, out / "checkpoints" / f"checkpoint_step_{global_step:08d}.pt")
            model.eval(); preds=[]; targets=[]; tl=tm=ts=tn=0.0
            with torch.no_grad():
                for batch in test_loader:
                    members = model.forward_members(batch.semantic, batch.z_current, batch.z_next, batch.action); mean = members.mean(0); std = members.std(0, unbiased=False)
                    loss_vec = reg_loss_fn(mean, batch.value); mse = nn.functional.mse_loss(mean, batch.value, reduction="none"); n = int(batch.value.numel())
                    tl += float(loss_vec.mean().item()) * n; tm += float(mse.mean().item()) * n; ts += float(std.mean().item()) * n; tn += n
                    preds.extend(float(x) for x in mean.detach().cpu()); targets.extend(float(x) for x in batch.value.detach().cpu())
            row = {"epoch": epoch, "train_loss": sums["loss"] / max(1, sums["n"]), "train_reg_loss": sums["reg"] / max(1, sums["n"]), "train_rank_loss": sums["rank"] / max(1, sums["n"]), "test_loss": tl / max(1, tn), "test_mse": tm / max(1, tn), "test_pearson": pearson_r(preds, targets), "test_ensemble_std": ts / max(1, tn), "num_train": int(sums["n"]), "num_test": int(tn)}
            writer.writerow(row); f.flush(); emit_metrics(row | {"global_step": global_step}, wandb_run=wandb_run, step=global_step, prefix="epoch/")
            if float(row["test_mse"]) < float(best["test_mse"]):
                best = {"test_mse": float(row["test_mse"]), "epoch": epoch}
                torch.save({"ensemble_state": model.state_dict(), "args": vars(args), "best": best, "visual_dim": visual_dim, "visual_encoder": args.visual_encoder, "semantic_dim": QWEN_VISUAL_DIM, "ensemble_size": model.ensemble_size, "prior_scale": model.prior_scale, "checkpoint_format": "rpe_value_ensemble_v1", "source_init": str(args.init_checkpoint)}, out / "best_value_ensemble.pt")
            torch.save({"ensemble_state": model.state_dict(), "optimizer_state": opt.state_dict(), "epoch": epoch + 1, "batch_idx": 0, "global_step": global_step, "best": best, "args": vars(args)}, out / "checkpoints" / f"checkpoint_step_{global_step:08d}.pt")
        finally:
            step_log.close()
    (out / "summary.json").write_text(json.dumps({"best": best, "args": vars(args), "num_samples": len(ds), "outputs": {"best_value_ensemble": str(out / "best_value_ensemble.pt")}}, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.summary.update({"best_test_mse": best["test_mse"], "best_epoch": best["epoch"], "output_dir": str(out)})
        wandb_run.finish()


if __name__ == "__main__":
    main()
