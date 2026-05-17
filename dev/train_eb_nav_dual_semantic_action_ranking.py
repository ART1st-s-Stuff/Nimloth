"""Train dual-semantic EB-Nav action ranker with no-CoT and planner/CoT latents."""
from __future__ import annotations

import argparse, csv, json, random, sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.eval_eb_nav_value_action_ranking import one_hot, discounted_return  # noqa: E402
from dev.train_eb_nav_value_head_predicted import (  # noqa: E402
    NUM_PATCHES,
    QWEN_VISUAL_DIM,
    SemanticWMValueHead,
    build_visual_encoder,
    build_wm_from_checkpoint,
    encode_many,
    resolve_repo_path,
)
from dev.train_eb_nav_joint_wm_value import freeze_qwen, make_subset  # noqa: E402
from src.train.train_wm_joint import _joint_collate_fn  # noqa: E402
from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402


class IndexedDataset(Dataset):
    def __init__(self, ds: Dataset): self.ds=ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i: int): return int(i), self.ds[i]


def collate_indexed(batch: list[tuple[int, Any]]) -> dict[str, Any]:
    idx=[int(x[0]) for x in batch]
    raw=_joint_collate_fn([x[1] for x in batch])
    raw["_cache_indices"] = torch.tensor(idx, dtype=torch.long)
    return raw


def load_head_init(head: SemanticWMValueHead, path: str) -> None:
    if not path: return
    ckpt=torch.load(resolve_repo_path(path), map_location="cpu")
    state=ckpt.get("head_state", ckpt)
    current=head.state_dict()
    filtered={k:v for k,v in state.items() if k in current and tuple(v.shape)==tuple(current[k].shape)}
    skipped=[k for k,v in state.items() if k in current and tuple(v.shape)!=tuple(current[k].shape)]
    missing, unexpected=head.load_state_dict(filtered, strict=False)
    print(f"loaded_head_init={path} loaded={len(filtered)} skipped_shape={skipped[:8]} missing={missing[:8]} unexpected={unexpected[:8]}", flush=True)


def load_cache(path: str) -> dict[str, Any]:
    data=torch.load(resolve_repo_path(path), map_location="cpu")
    if "latents" not in data:
        raise RuntimeError(f"planner cache missing latents: {path}")
    return data


def make_semantic(no_cot: torch.Tensor, cot: torch.Tensor, prior: torch.Tensor, mode: str) -> torch.Tensor:
    z0=torch.zeros_like(no_cot); zc=torch.zeros_like(cot); zp=torch.zeros_like(prior)
    if mode == "fast":
        return torch.cat([no_cot, zc, zp], dim=-1)
    if mode == "planner":
        return torch.cat([z0, cot, prior], dim=-1)
    if mode == "hybrid":
        return torch.cat([no_cot, cot, prior], dim=-1)
    raise ValueError(mode)


def choose_train_mode(probs: tuple[float,float,float]) -> str:
    r=random.random(); a,b,c=probs
    if r < a: return "fast"
    if r < a+b: return "planner"
    return "hybrid"


def encode_scores(
    raw: dict[str, Any], *, visual_encoder: Any, no_cot_encoder: QwenLLMLatentEncoder,
    planner_cache: dict[str, Any], wm: torch.nn.Module, head: SemanticWMValueHead, device: torch.device,
    gamma: float, mode: str, visual_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history_images: list[list[str]] = raw["history_images"]
    bsz=len(history_images)
    prompts=[str(x or "") for x in raw.get("prompts", raw.get("instructions", [""]*bsz))]
    hist_actions=raw["history_actions"].float().to(device)
    labels=raw["future_action_ids"][:,0].long().to(device).clamp(0,7)
    returns=discounted_return(raw["future_rewards"].float().to(device), gamma)
    idx=raw["_cache_indices"].long()

    flat=[p for seq in history_images for p in seq]
    z_hist=encode_many(visual_encoder, flat, None, device).reshape(bsz, len(history_images[0]), NUM_PATCHES, visual_dim)
    z_current=z_hist[:,-1]
    no_cot=encode_many(no_cot_encoder, [seq[-1] for seq in history_images], prompts, device, expected_flat_dim=3584).reshape(bsz,3584)
    cot=planner_cache["latents"].index_select(0, idx).to(device=device, dtype=no_cot.dtype)
    prior=planner_cache["action_prior"].index_select(0, idx).to(device=device, dtype=no_cot.dtype)
    semantic=make_semantic(no_cot, cot, prior, mode)

    cols=[]
    for action_id in range(8):
        ids=torch.full((bsz,), action_id, dtype=torch.long, device=device)
        avec=one_hot(ids, 8)
        teacher=hist_actions.clone(); teacher[:,-1,:]=avec
        with torch.no_grad():
            z_next=wm.predict_next(z_hist, teacher)
        cols.append(head(semantic, z_current, z_next, avec))
    return torch.stack(cols, dim=1), labels, returns


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, returns: torch.Tensor) -> dict[str, Any]:
    scores=logits.detach().float().cpu(); y=labels.detach().long().cpu(); ret=returns.detach().float().cpu()
    n=int(y.numel()); order=torch.argsort(scores, dim=1, descending=True)
    pred=order[:,0]
    ranks=[]; mrr=[]; pair=0; gaps=[]; logged=[]
    per={str(i): {"n":0,"top1":0} for i in range(8)}
    for i in range(n):
        lab=int(y[i]); per[str(lab)]["n"] += 1; per[str(lab)]["top1"] += int(pred[i].item()==lab)
        pos=(order[i] == lab).nonzero(as_tuple=False)
        rank=int(pos[0].item())+1 if pos.numel() else 9
        ranks.append(rank); mrr.append(1.0/rank)
        s_lab=float(scores[i,lab]); others=[j for j in range(8) if j != lab]
        other_scores=scores[i, others]
        pair += int((s_lab > other_scores).sum().item())
        gaps.append(s_lab - float(other_scores.max().item()))
        logged.append(s_lab)
    top1=float((pred==y).float().mean().item()) if n else 0.0
    macro=sum((v["top1"]/v["n"] if v["n"] else 0.0) for v in per.values())/8.0
    logged_t=torch.tensor(logged); ret_t=ret[:len(logged)]
    pearson=0.0
    if len(logged) > 1 and float(logged_t.std()) > 1e-8 and float(ret_t.std()) > 1e-8:
        pearson=float(torch.corrcoef(torch.stack([logged_t, ret_t]))[0,1].item())
    return {"n":n,"top1_acc":top1,"macro_top1_acc":macro,"mean_rank":sum(ranks)/max(1,n),"mrr":sum(mrr)/max(1,n),"pairwise_logged_gt_other":pair/max(1,n*7),"score_gap_logged_vs_max_other":sum(gaps)/max(1,n),"logged_score_return_pearson":pearson,"per_action":{k:{"n":v["n"],"top1_acc":v["top1"]/v["n"] if v["n"] else 0.0} for k,v in per.items()}}


def evaluate(loader: DataLoader, **kwargs: Any) -> dict[str, Any]:
    all_logits=[]; all_labels=[]; all_returns=[]
    kwargs["head"].eval()
    with torch.no_grad():
        for raw in loader:
            logits, labels, returns=encode_scores(raw, **kwargs)
            all_logits.append(logits.cpu()); all_labels.append(labels.cpu()); all_returns.append(returns.cpu())
    return metrics_from_logits(torch.cat(all_logits), torch.cat(all_labels), torch.cat(all_returns))


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", required=True); p.add_argument("--test-manifest", required=True); p.add_argument("--rollout-test-manifest", default="")
    p.add_argument("--train-planner-cache", required=True); p.add_argument("--test-planner-cache", required=True); p.add_argument("--rollout-planner-cache", default="")
    p.add_argument("--images-base-dir", default="."); p.add_argument("--wm-checkpoint", required=True); p.add_argument("--head-init", default=""); p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=8192); p.add_argument("--test-max-samples", type=int, default=4096); p.add_argument("--epochs", type=int, default=5); p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--gamma", type=float, default=0.95); p.add_argument("--value-aux-weight", type=float, default=0.05)
    p.add_argument("--class-balanced", action="store_true"); p.add_argument("--mode-probs", default="0.25,0.35,0.40", help="fast,planner,hybrid train sampling probs")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--cuda-device", default="0"); p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct"); p.add_argument("--model-dtype", default="auto"); p.add_argument("--device-map", default="auto")
    p.add_argument("--visual-encoder", choices=["qwen", "dino"], default="qwen")
    p.add_argument("--dino-model-name", default="dinov2_vits14")
    p.add_argument("--dino-image-size", type=int, default=224)
    return p.parse_args()


def main() -> None:
    args=parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cpu" if str(args.cuda_device) in {"","-1","cpu"} else f"cuda:{args.cuda_device}")
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); (out/"args.json").write_text(json.dumps(vars(args), indent=2))
    adapter=QwenVLMAdapter(model_name=args.model_name, latent_dim=NUM_PATCHES * QWEN_VISUAL_DIM, enabled=True, fallback_enabled=False, device_map=None if str(args.device_map).lower() in {"","none"} else args.device_map, model_dtype=args.model_dtype)
    freeze_qwen(adapter)
    visual_encoder, visual_dim, visual_latent_dim = build_visual_encoder(args, adapter)
    no_cot_encoder=QwenLLMLatentEncoder(QWEN_VISUAL_DIM, name="qwen_no_cot", model_name=args.model_name, qwen_adapter=adapter, use_vision_only=False, visual_pooling="last", cache_latents=True)
    wm=build_wm_from_checkpoint(resolve_repo_path(args.wm_checkpoint), device, visual_dim=visual_dim, latent_dim=visual_latent_dim); wm.eval()
    head=SemanticWMValueHead(semantic_dim=QWEN_VISUAL_DIM*2+8, visual_dim=visual_dim, action_dim=8, hidden=512).to(device)
    load_head_init(head, args.head_init)
    train_cache=load_cache(args.train_planner_cache); test_cache=load_cache(args.test_planner_cache); rollout_cache=load_cache(args.rollout_planner_cache) if args.rollout_planner_cache else None
    train_ds=IndexedDataset(make_subset(args.train_manifest, args.images_base_dir, max_samples=args.max_samples)); test_ds=IndexedDataset(make_subset(args.test_manifest, args.images_base_dir, max_samples=args.test_max_samples))
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=collate_indexed); test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate_indexed)
    rollout_loader=None
    if args.rollout_test_manifest:
        rollout_loader=DataLoader(IndexedDataset(make_subset(args.rollout_test_manifest,args.images_base_dir,max_samples=args.test_max_samples)),batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate_indexed)
    labels=torch.tensor(train_cache.get("labels", []), dtype=torch.long)
    ce_weight=None
    if args.class_balanced and labels.numel():
        counts=torch.bincount(labels.clamp(0,7), minlength=8).float(); ce_weight=(counts.sum()/(counts.clamp_min(1).sqrt())); ce_weight=ce_weight/ce_weight.mean(); ce_weight=ce_weight.to(device)
        print(f"class_weights={ce_weight.detach().cpu().tolist()}", flush=True)
    opt=torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    probs_raw=tuple(float(x) for x in args.mode_probs.split(",")); s=sum(probs_raw); probs=(probs_raw[0]/s, probs_raw[1]/s, probs_raw[2]/s)
    rows=[]; best=None
    for epoch in range(1,args.epochs+1):
        head.train(); n=0; loss_sum=ce_sum=aux_sum=correct=0.0
        for raw in train_loader:
            mode=choose_train_mode(probs)
            logits, y, ret=encode_scores(raw, visual_encoder=visual_encoder, no_cot_encoder=no_cot_encoder, planner_cache=train_cache, wm=wm, head=head, device=device, gamma=args.gamma, mode=mode, visual_dim=visual_dim)
            ce=F.cross_entropy(logits, y, weight=ce_weight)
            logged=logits.gather(1, y[:,None]).squeeze(1); aux=F.smooth_l1_loss(logged, ret)
            loss=ce + args.value_aux_weight*aux
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            b=int(y.numel()); n+=b; loss_sum+=float(loss.item())*b; ce_sum+=float(ce.item())*b; aux_sum+=float(aux.item())*b; correct+=float((logits.argmax(1)==y).sum().item())
        row={"epoch":epoch,"train_loss":loss_sum/max(1,n),"train_ce":ce_sum/max(1,n),"train_value_aux":aux_sum/max(1,n),"train_top1":correct/max(1,n),"train_n":n}
        for mode in ["fast","planner","hybrid"]:
            m=evaluate(test_loader, visual_encoder=visual_encoder, no_cot_encoder=no_cot_encoder, planner_cache=test_cache, wm=wm, head=head, device=device, gamma=args.gamma, mode=mode, visual_dim=visual_dim)
            row.update({f"test_{mode}_{k}":v for k,v in m.items() if k != "per_action"})
        if rollout_loader is not None and rollout_cache is not None:
            m=evaluate(rollout_loader, visual_encoder=visual_encoder, no_cot_encoder=no_cot_encoder, planner_cache=rollout_cache, wm=wm, head=head, device=device, gamma=args.gamma, mode="hybrid", visual_dim=visual_dim)
            row.update({f"rollout_hybrid_{k}":v for k,v in m.items() if k != "per_action"})
        rows.append(row); print(json.dumps(row), flush=True)
        score=float(row.get("test_hybrid_mrr",0.0))
        if best is None or score > float(best["test_hybrid_mrr"]):
            best={"epoch":epoch,"test_hybrid_mrr":score,"test_hybrid_top1":row.get("test_hybrid_top1_acc"),"test_planner_top1":row.get("test_planner_top1_acc"),"test_fast_top1":row.get("test_fast_top1_acc")}
            torch.save({"wm_state":wm.state_dict(),"head_state":head.state_dict(),"args":vars(args),"best":best,"semantic_dim":QWEN_VISUAL_DIM*2+8,"visual_dim":visual_dim,"visual_encoder":args.visual_encoder}, out/"best_dual_semantic_action_head.pt")
    torch.save({"wm_state":wm.state_dict(),"head_state":head.state_dict(),"args":vars(args),"best":best,"semantic_dim":QWEN_VISUAL_DIM*2+8,"visual_dim":visual_dim,"visual_encoder":args.visual_encoder}, out/"final_dual_semantic_action_head.pt")
    (out/"summary.json").write_text(json.dumps({"best":best,"last":rows[-1] if rows else None,"visual_dim":visual_dim,"visual_encoder":args.visual_encoder}, indent=2))
    with (out/"train_log.csv").open("w",newline="") as f:
        fields=sorted({k for r in rows for k in r.keys()}); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

if __name__ == "__main__": main()
