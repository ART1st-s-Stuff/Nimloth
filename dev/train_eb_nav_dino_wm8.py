"""Train DINO-visual LeWM world model on EB-Nav 8-action manifests from scratch."""
from __future__ import annotations

import argparse, csv, json, random, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.train_eb_nav_value_head_predicted import NUM_PATCHES, DINO_VISUAL_DIM, build_visual_encoder, resolve_repo_path  # noqa: E402
from src.train.train_wm_joint import CustomJointSequenceDataset, _joint_collate_fn  # noqa: E402
from src.wm.predictor.lewm import LeWMWorldModel  # noqa: E402


@dataclass
class WMBatch:
    z_history: torch.Tensor
    action_history: torch.Tensor
    z_future: torch.Tensor
    gt_action_future: torch.Tensor


def make_subset(manifest: str, images_base_dir: str, *, max_samples: int) -> Subset:
    ds = CustomJointSequenceDataset(
        manifest, images_base_dir, history_len=4, temporal_stride=3, action_dim=8, max_samples=0, require_prompt=False
    )
    idx = list(range(len(ds)))[: int(max_samples) or len(ds)]
    return Subset(ds, idx)


@torch.no_grad()
def encode_paths_batched(encoder: Any, paths: Sequence[str], device: torch.device, *, batch_size: int, visual_dim: int) -> torch.Tensor:
    outs: list[torch.Tensor] = []
    if hasattr(encoder, "encode_image_paths"):
        for start in range(0, len(paths), batch_size):
            chunk = list(paths[start : start + batch_size])
            encoded = encoder.encode_image_paths(chunk)
            outs.extend(x.z.detach().float() for x in encoded)
    else:
        for path in paths:
            outs.append(encoder.encode_image_path(path).z.detach().float())
    values = []
    for z in outs:
        if z.dim() == 1:
            z = z.reshape(NUM_PATCHES, visual_dim)
        values.append(z.to(device))
    return torch.stack(values, dim=0)


def encode_wm_batch(raw: dict[str, Any], visual_encoder: Any, device: torch.device, *, visual_dim: int, encode_batch_size: int) -> WMBatch:
    history_images: list[list[str]] = raw["history_images"]
    future_images: list[list[str]] = raw["future_images"]
    bsz, history_len, future_len = len(history_images), len(history_images[0]), len(future_images[0])
    flat_hist = [p for seq in history_images for p in seq]
    flat_future = [p for seq in future_images for p in seq]
    z_history = encode_paths_batched(visual_encoder, flat_hist, device, batch_size=encode_batch_size, visual_dim=visual_dim).reshape(bsz, history_len, NUM_PATCHES, visual_dim)
    z_future = encode_paths_batched(visual_encoder, flat_future, device, batch_size=encode_batch_size, visual_dim=visual_dim).reshape(bsz, future_len, NUM_PATCHES, visual_dim)
    return WMBatch(
        z_history=z_history,
        action_history=raw["history_actions"].float().to(device),
        z_future=z_future,
        gt_action_future=raw["future_actions"].float().to(device),
    )


def build_wm(device: torch.device, *, visual_dim: int, ensemble_size: int = 1) -> LeWMWorldModel:
    wm = LeWMWorldModel(
        latent_dim=NUM_PATCHES * int(visual_dim),
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
        ensemble_size=int(ensemble_size),
        predict_delta=True,
        delta_scale=1.0,
        zero_init_delta_head=True,
    ).to(device)
    return wm


def wm_rollout_loss(wm: LeWMWorldModel, batch: WMBatch, *, free_run_start: int, detach_rollout: bool) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_z = batch.z_history
    teacher_action = batch.action_history.clone()
    losses=[]; mse_values=[]; copy_values=[]; cos_values=[]
    horizon = int(batch.z_future.size(1))
    for step_idx in range(horizon):
        teacher_action[:, -1, :] = batch.gt_action_future[:, step_idx, :]
        pred_z = wm.predict_next(teacher_z, teacher_action)
        target_z = batch.z_future[:, step_idx].detach()
        last_z = teacher_z[:, -1]
        loss = F.mse_loss(pred_z, target_z)
        losses.append(loss)
        mse_values.append(loss.detach())
        copy_values.append(F.mse_loss(last_z, target_z).detach())
        cos_values.append(F.cosine_similarity((pred_z-last_z).flatten(1), (target_z-last_z).flatten(1), dim=1).mean().detach())
        use_free = horizon > 1 and (step_idx + 1) >= int(free_run_start)
        next_teacher = pred_z if use_free else target_z
        if use_free and detach_rollout:
            next_teacher = next_teacher.detach()
        teacher_z = torch.cat([teacher_z[:, 1:], next_teacher.unsqueeze(1)], dim=1)
        if step_idx < horizon - 1:
            teacher_action = torch.cat([teacher_action[:, 1:], batch.gt_action_future[:, step_idx].unsqueeze(1)], dim=1)
    wm_mse=torch.stack(mse_values).mean(); copy_mse=torch.stack(copy_values).mean()
    return torch.stack(losses).mean(), {
        "wm_mse": float(wm_mse.item()),
        "copy_mse": float(copy_mse.item()),
        "wm_margin": float((wm_mse-copy_mse).item()),
        "delta_cos": float(torch.stack(cos_values).mean().item()),
    }


@torch.no_grad()
def evaluate(loader: DataLoader, *, visual_encoder: Any, wm: LeWMWorldModel, device: torch.device, visual_dim: int, encode_batch_size: int, free_run_start: int, detach_rollout: bool) -> dict[str, float]:
    wm.eval(); sums={"loss":0.0,"wm_mse":0.0,"copy_mse":0.0,"wm_margin":0.0,"delta_cos":0.0}; n=0
    for raw in loader:
        batch=encode_wm_batch(raw, visual_encoder, device, visual_dim=visual_dim, encode_batch_size=encode_batch_size)
        loss, m=wm_rollout_loss(wm,batch,free_run_start=free_run_start,detach_rollout=detach_rollout)
        b=int(batch.z_history.size(0)); n+=b; sums["loss"] += float(loss.item())*b
        for k,v in m.items(): sums[k]+=float(v)*b
    wm.train(); return {k:v/max(1,n) for k,v in sums.items()} | {"n":float(n)}


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", required=True); p.add_argument("--test-manifest", required=True); p.add_argument("--rollout-test-manifest", default="")
    p.add_argument("--images-base-dir", default="."); p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=16384); p.add_argument("--test-max-samples", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=5); p.add_argument("--batch-size", type=int, default=4); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--free-run-start", type=int, default=1); p.add_argument("--detach-rollout", action="store_true"); p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--cuda-device", default="0")
    p.add_argument("--dino-model-name", default="dinov2_vits14"); p.add_argument("--dino-image-size", type=int, default=224); p.add_argument("--encode-batch-size", type=int, default=32)
    p.add_argument("--ensemble-size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args=parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cpu" if str(args.cuda_device) in {"","-1","cpu"} else f"cuda:{args.cuda_device}")
    out=resolve_repo_path(args.output_dir); out.mkdir(parents=True, exist_ok=True); (out/"args.json").write_text(json.dumps(vars(args),indent=2))
    build_args=argparse.Namespace(visual_encoder="dino", dino_model_name=args.dino_model_name, dino_image_size=args.dino_image_size)
    visual_encoder, visual_dim, _ = build_visual_encoder(build_args, None)
    if visual_dim != DINO_VISUAL_DIM: print(f"visual_dim={visual_dim}", flush=True)
    wm=build_wm(device, visual_dim=visual_dim, ensemble_size=args.ensemble_size); opt=torch.optim.AdamW(wm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds=make_subset(args.train_manifest,args.images_base_dir,max_samples=args.max_samples)
    test_ds=make_subset(args.test_manifest,args.images_base_dir,max_samples=args.test_max_samples)
    rollout_ds=make_subset(args.rollout_test_manifest,args.images_base_dir,max_samples=args.test_max_samples) if args.rollout_test_manifest else None
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=_joint_collate_fn)
    test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=_joint_collate_fn)
    rollout_loader=DataLoader(rollout_ds,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=_joint_collate_fn) if rollout_ds else None
    rows=[]; best={"test_wm_mse": float("inf"), "epoch": -1}
    with (out/"train_log.csv").open("w", newline="") as f:
        fields=["epoch","train_loss","train_wm_mse","train_copy_mse","train_wm_margin","train_delta_cos","test_loss","test_wm_mse","test_copy_mse","test_wm_margin","test_delta_cos","rollout_loss","rollout_wm_mse","rollout_copy_mse","rollout_wm_margin","rollout_delta_cos"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for epoch in range(1,args.epochs+1):
            wm.train(); sums={"loss":0.0,"wm_mse":0.0,"copy_mse":0.0,"wm_margin":0.0,"delta_cos":0.0}; n=0
            for step, raw in enumerate(train_loader, start=1):
                batch=encode_wm_batch(raw, visual_encoder, device, visual_dim=visual_dim, encode_batch_size=args.encode_batch_size)
                loss,m=wm_rollout_loss(wm,batch,free_run_start=args.free_run_start,detach_rollout=args.detach_rollout)
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(wm.parameters(), args.grad_clip); opt.step()
                b=int(batch.z_history.size(0)); n+=b; sums["loss"]+=float(loss.item())*b
                for k,v in m.items(): sums[k]+=float(v)*b
                if step % 100 == 0:
                    print(json.dumps({"epoch":epoch,"step":step,"train_loss":sums["loss"]/max(1,n),"seen":n}), flush=True)
            train={k:v/max(1,n) for k,v in sums.items()}
            test=evaluate(test_loader,visual_encoder=visual_encoder,wm=wm,device=device,visual_dim=visual_dim,encode_batch_size=args.encode_batch_size,free_run_start=args.free_run_start,detach_rollout=args.detach_rollout)
            rollout=evaluate(rollout_loader,visual_encoder=visual_encoder,wm=wm,device=device,visual_dim=visual_dim,encode_batch_size=args.encode_batch_size,free_run_start=args.free_run_start,detach_rollout=args.detach_rollout) if rollout_loader else {}
            row={"epoch":epoch,"train_loss":train["loss"],"train_wm_mse":train["wm_mse"],"train_copy_mse":train["copy_mse"],"train_wm_margin":train["wm_margin"],"train_delta_cos":train["delta_cos"],
                 "test_loss":test["loss"],"test_wm_mse":test["wm_mse"],"test_copy_mse":test["copy_mse"],"test_wm_margin":test["wm_margin"],"test_delta_cos":test["delta_cos"],
                 "rollout_loss":rollout.get("loss"),"rollout_wm_mse":rollout.get("wm_mse"),"rollout_copy_mse":rollout.get("copy_mse"),"rollout_wm_margin":rollout.get("wm_margin"),"rollout_delta_cos":rollout.get("delta_cos")}
            rows.append(row); w.writerow(row); f.flush(); print(json.dumps(row), flush=True)
            if float(row["test_wm_mse"]) < float(best["test_wm_mse"]):
                best={"epoch":epoch,"test_wm_mse":float(row["test_wm_mse"]),"test_wm_margin":float(row["test_wm_margin"]),"rollout_wm_mse":row.get("rollout_wm_mse")}
                torch.save({"wm_state":wm.state_dict(),"args":vars(args),"best":best,"visual_dim":visual_dim,"visual_encoder":"dino","ensemble_size":int(args.ensemble_size),"semantic_dim":3584*2+8}, out/"best_dino_wm8.pt")
    torch.save({"wm_state":wm.state_dict(),"args":vars(args),"best":best,"visual_dim":visual_dim,"visual_encoder":"dino","ensemble_size":int(args.ensemble_size),"semantic_dim":3584*2+8}, out/"final_dino_wm8.pt")
    (out/"summary.json").write_text(json.dumps({"best":best,"last":rows[-1] if rows else None,"visual_dim":visual_dim,"visual_encoder":"dino"}, indent=2))

if __name__ == "__main__": main()
