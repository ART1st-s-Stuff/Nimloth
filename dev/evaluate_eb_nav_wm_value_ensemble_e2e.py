"""Online EB-Nav E2E eval for WM predictor ensemble + value-head ensemble."""
from __future__ import annotations

import argparse, glob, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.collect_eb_nav_random_rollouts import _extract_rgb, _import_eb_navigation_env, _matches_split, _safe_float, _safe_int, _save_rgb  # noqa: E402
from dev.eval_eb_nav_value_action_ranking import one_hot  # noqa: E402
from dev.train_eb_nav_dual_semantic_action_ranking import make_semantic  # noqa: E402
from dev.train_eb_nav_joint_wm_value import freeze_qwen  # noqa: E402
from dev.train_eb_nav_value_head_predicted import NUM_PATCHES, QWEN_VISUAL_DIM, SemanticWMValueHead, build_visual_encoder, build_wm_from_checkpoint, encode_many, resolve_repo_path  # noqa: E402
from src.data.eb_nav_dataset import ACTION_NAMES  # noqa: E402
from src.vlm.qwen_adapter import QwenVLM  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402


def _entropy_from_probs(probs: torch.Tensor) -> float:
    p = probs.float().flatten()
    if p.numel() == 0:
        return 0.0
    p = p / p.sum().clamp_min(1e-12)
    return float((-(p * p.clamp_min(1e-12).log()).sum()).item())


def _build_meta(record: dict[str, Any], idx: int) -> dict[str, Any]:
    eval_set = str(record.get("eval_set", "unknown"))
    episode_id = _safe_int(record.get("episode_id", idx + 1), default=idx + 1)
    return {"record_idx": idx, "episode_id": int(episode_id), "eval_set": eval_set, "task_key": f"{eval_set}:{episode_id:03d}", "instruction": str(record.get("instruction", "")), "prompt": str(record.get("input", record.get("instruction", ""))), "model_name": str(record.get("model_name", ""))}


def select_records(records: list[dict[str, Any]], split: str, num_episodes: int, seed: int) -> list[dict[str, Any]]:
    metas = [_build_meta(r, i) for i, r in enumerate(records) if _matches_split(r, split)]
    rng = random.Random(seed); rng.shuffle(metas)
    return metas[: int(num_episodes)]


def expand_checkpoints(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in patterns:
        for part in str(item).split(','):
            part = part.strip()
            if not part:
                continue
            matches = sorted(glob.glob(str(resolve_repo_path(part))))
            if matches:
                out.extend(Path(m) for m in matches)
            else:
                out.append(resolve_repo_path(part))
    # deterministic unique
    seen=set(); uniq=[]
    for p in out:
        s=str(p)
        if s not in seen:
            seen.add(s); uniq.append(p)
    return uniq


@torch.no_grad()
def choose_action(*, image_history: list[str], action_history: list[int], prompt: str, visual_encoder: Any, no_cot_encoder: QwenLLMLatentEncoder, planner_adapter: QwenVLM, wm: torch.nn.Module, heads: list[SemanticWMValueHead], device: torch.device, mode: str, max_new_tokens: int, visual_dim: int, risk_lambda: float, semantic_dim: int) -> tuple[int, dict[str, Any]]:
    hist_len = len(image_history)
    z_hist = encode_many(visual_encoder, image_history, None, device).reshape(1, hist_len, NUM_PATCHES, visual_dim)
    no_cot = encode_many(no_cot_encoder, [image_history[-1]], [prompt], device, expected_flat_dim=QWEN_VISUAL_DIM).reshape(1, QWEN_VISUAL_DIM)
    z_current = z_hist[:, -1]
    cot = torch.zeros(1, 3584, device=device, dtype=no_cot.dtype)
    prior = torch.zeros(1, 8, device=device, dtype=no_cot.dtype)
    planner_text = ""; planner_failed = False; planner_error = ""
    if mode in {"planner", "hybrid"}:
        try:
            got = planner_adapter.get_planner_latent_and_action_prior_batch(image_paths=[image_history[-1]], prompts=[prompt], responses=None, max_new_tokens=max_new_tokens)
            cot = cast(torch.Tensor, got["latent"]).to(device=device, dtype=no_cot.dtype).reshape(1, -1)
            prior = cast(torch.Tensor, got["action_prior"]).to(device=device, dtype=no_cot.dtype).reshape(1, -1)
            planner_text = str(got.get("text", [""])[0])
        except Exception as exc:
            planner_failed = True; planner_error = f"{type(exc).__name__}: {exc}"
    semantic = no_cot if semantic_dim == QWEN_VISUAL_DIM else make_semantic(no_cot, cot, prior, mode)
    hist_actions = torch.zeros(1, hist_len, 8, dtype=torch.float32, device=device)
    for i, a in enumerate(action_history[-hist_len:]):
        if 0 <= int(a) < 8:
            hist_actions[0, i, int(a)] = 1.0
    mean_scores=[]; std_scores=[]; all_scores=[]; pred_unc=[]
    for action_id in range(8):
        ids = torch.tensor([action_id], dtype=torch.long, device=device)
        action_vec = one_hot(ids, 8)
        teacher_action = hist_actions.clone(); teacher_action[:, -1, :] = action_vec
        ens_preds = None
        if hasattr(wm, 'predict_next_ensemble'):
            ens_preds = wm.predict_next_ensemble(z_hist, teacher_action)
            z_next = ens_preds.mean(dim=0)
            pred_unc.append(float(ens_preds.float().var(dim=0, unbiased=False).flatten(1).mean().item()))
        else:
            z_next = wm.predict_next(z_hist, teacher_action)
            pred_unc.append(0.0)
        vals=[]
        for head in heads:
            vals.append(float(head(semantic, z_current, z_next, action_vec).item()))
        t=torch.tensor(vals, dtype=torch.float32)
        all_scores.append(vals); mean_scores.append(float(t.mean().item())); std_scores.append(float(t.std(unbiased=False).item()))
    has_planner_prior = bool(mode in {"planner", "hybrid"} and not planner_failed and prior.numel() > 0 and float(prior.float().abs().sum().item()) > 0.0)
    if has_planner_prior:
        prior_probs = torch.softmax(prior.float().reshape(-1), dim=0)
        planner_prior = [float(x) for x in prior_probs.detach().cpu().tolist()]
        planner_order = sorted(range(len(planner_prior)), key=lambda i: planner_prior[i], reverse=True)
        planner_top1 = int(planner_order[0]) if planner_order else -1
        planner_top2 = int(planner_order[1]) if len(planner_order) > 1 else -1
        planner_margin = float(planner_prior[planner_top1] - planner_prior[planner_top2]) if planner_top1 >= 0 and planner_top2 >= 0 else 0.0
        planner_entropy = _entropy_from_probs(prior_probs)
    else:
        planner_prior = []
        planner_top1 = -1
        planner_margin = 0.0
        planner_entropy = 0.0
    policy_scores=[m - risk_lambda*s for m,s in zip(mean_scores,std_scores)]
    best = int(max(range(8), key=lambda i: policy_scores[i]))
    order=sorted(range(8), key=lambda i: policy_scores[i], reverse=True)
    margin=float(policy_scores[order[0]]-policy_scores[order[1]]) if len(order)>1 else 0.0
    return best, {"score_mean":mean_scores,"score_std":std_scores,"score_all":all_scores,"policy_scores":policy_scores,"top1_margin":margin,"selected_score_mean":mean_scores[best],"selected_score_std":std_scores[best],"selected_pred_uncertainty":pred_unc[best],"pred_uncertainty_by_action":pred_unc,"planner_failed":planner_failed,"planner_error":planner_error,"planner_text":planner_text,"planner_action_prior":planner_prior,"planner_entropy":planner_entropy,"planner_top1_action":planner_top1,"planner_top1_prob":float(planner_prior[planner_top1]) if planner_top1 >= 0 else 0.0,"planner_top1_margin":planner_margin,"planner_value_conflict":bool(planner_top1 >= 0 and planner_top1 != best),"planner_value_gap":float(policy_scores[best] - policy_scores[planner_top1]) if planner_top1 >= 0 else 0.0}


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="datasets/EB-Nav/eb-nav_dataset_single_step.json")
    p.add_argument("--embodiedbench-root", default="/project/peilab/atst/EmbodiedBench")
    p.add_argument("--wm-checkpoint", required=True)
    p.add_argument("--value-checkpoints", nargs='+', required=True)
    p.add_argument("--planner-lora", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--split", choices=["all","train","test"], default="all")
    p.add_argument("--seed", type=int, default=20260517)
    p.add_argument("--history-len", type=int, default=4)
    p.add_argument("--resolution", type=int, default=500)
    p.add_argument("--fov", type=int, default=100)
    p.add_argument("--exp-name", default="eval_ens8_value_e2e")
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-dtype", default="auto")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--visual-encoder", choices=["qwen","dino"], required=True)
    p.add_argument("--dino-model-name", default="dinov2_vits14")
    p.add_argument("--dino-image-size", type=int, default=224)
    p.add_argument("--mode", choices=["fast","planner","hybrid"], default="hybrid")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--risk-lambda", type=float, default=0.0)
    p.add_argument("--save-screenshots", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args=parse_args()
    out=Path(args.output_dir); shots=out/"step_screenshots"; out.mkdir(parents=True, exist_ok=True); shots.mkdir(parents=True, exist_ok=True)
    value_paths=expand_checkpoints(args.value_checkpoints)
    (out/"args.json").write_text(json.dumps(vars(args) | {"expanded_value_checkpoints":[str(p) for p in value_paths]}, indent=2))
    if not value_paths:
        raise RuntimeError("no value checkpoints found")
    device=torch.device("cpu" if str(args.cuda_device) in {"","-1","cpu"} else f"cuda:{args.cuda_device}")
    first_v=torch.load(value_paths[0], map_location="cpu")
    visual_dim=int(first_v.get("visual_dim") or (384 if args.visual_encoder=="dino" else QWEN_VISUAL_DIM))
    visual_latent_dim=NUM_PATCHES*visual_dim
    wm=build_wm_from_checkpoint(resolve_repo_path(args.wm_checkpoint), device, visual_dim=visual_dim, latent_dim=visual_latent_dim); wm.eval()
    semantic_dim=int(first_v.get("semantic_dim", QWEN_VISUAL_DIM))
    heads=[]
    for pth in value_paths:
        ck=torch.load(pth, map_location="cpu")
        h=SemanticWMValueHead(semantic_dim=semantic_dim, visual_dim=visual_dim, action_dim=8, hidden=512).to(device)
        h.load_state_dict(ck["head_state"], strict=True); h.eval(); heads.append(h)
    visual_adapter=QwenVLM(model_name=args.model_name, latent_dim=NUM_PATCHES*QWEN_VISUAL_DIM, enabled=True, fallback_enabled=False, device_map=None if str(args.device_map).lower() in {"","none"} else args.device_map, model_dtype=args.model_dtype)
    freeze_qwen(visual_adapter)
    planner_adapter=QwenVLM(model_name=args.model_name, latent_dim=QWEN_VISUAL_DIM, enabled=True, fallback_enabled=False, device_map=None if str(args.device_map).lower() in {"","none"} else args.device_map, model_dtype=args.model_dtype, max_new_tokens=int(args.max_new_tokens))
    planner_adapter.load_lora_adapter(str(resolve_repo_path(args.planner_lora)), trainable=False); planner_adapter.planner_inference_mode=True; planner_adapter.max_new_tokens=int(args.max_new_tokens); freeze_qwen(planner_adapter)
    visual_build_args=argparse.Namespace(**(vars(args) | {"visual_encoder":args.visual_encoder}))
    visual_encoder, _, _ = build_visual_encoder(visual_build_args, visual_adapter)
    no_cot_encoder=QwenLLMLatentEncoder(QWEN_VISUAL_DIM, name="qwen_no_cot", model_name=args.model_name, qwen_adapter=visual_adapter, use_vision_only=False, visual_pooling="last", cache_latents=True)
    records=json.load(open(args.dataset, encoding="utf-8")); selected=select_records(records,args.split,args.num_episodes,args.seed)
    by_eval: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in selected: by_eval[str(m["eval_set"])].append(m)
    EBNavigationEnv=_import_eb_navigation_env(args.embodiedbench_root)
    transitions=[]; episodes=[]; action_counts=Counter(); planner_failures=0; rollout_id=0
    trans_path=out/"transitions.jsonl"; eps_path=out/"episodes.jsonl"; trans_path.write_text(""); eps_path.write_text("")
    for eval_set, items in sorted(by_eval.items()):
        env=EBNavigationEnv(eval_set=eval_set, exp_name=f"{args.exp_name}_{eval_set}", selected_indexes=[max(0,int(x["episode_id"])-1) for x in items], resolution=int(args.resolution), fov=int(args.fov))
        env._max_episode_steps=int(args.max_steps)
        try:
            for item in items:
                rollout_id+=1; obs=env.reset(); done=False; step_idx=0; info={}; ep_reward=0.0
                prompt=str(item.get("prompt") or item.get("instruction") or ""); instruction=str(item.get("instruction") or "")
                cur_path=_save_rgb(shots/f"rollout_{rollout_id:04d}_step_{step_idx:03d}_obs.png", _extract_rgb(obs))
                image_hist=[cur_path]*int(args.history_len); action_hist=[-1]*int(args.history_len)
                while not done and step_idx<int(args.max_steps):
                    action_id, dbg=choose_action(image_history=image_hist, action_history=action_hist, prompt=prompt, visual_encoder=visual_encoder, no_cot_encoder=no_cot_encoder, planner_adapter=planner_adapter, wm=wm, heads=heads, device=device, mode=args.mode, max_new_tokens=int(args.max_new_tokens), visual_dim=visual_dim, risk_lambda=float(args.risk_lambda), semantic_dim=semantic_dim)
                    planner_failures += int(bool(dbg.get("planner_failed")))
                    next_obs,reward,done,info=env.step(action_id,{"policy":"wm_value_ens8","task_key":str(item.get("task_key",""))},1)
                    ep_reward+=float(reward); action_counts[action_id]+=1
                    next_path=_save_rgb(shots/f"rollout_{rollout_id:04d}_step_{step_idx:03d}_next.png", _extract_rgb(next_obs))
                    row={"rollout_id":rollout_id,"episode_id":int(item.get("episode_id",-1)),"task_key":str(item.get("task_key","")),"eval_set":eval_set,"instruction":instruction,"step":step_idx,"image_t":image_hist[-1],"image_next":next_path,"sampled_action_id":action_id,"sampled_action_name":ACTION_NAMES.get(action_id,f"action_{action_id}"),"reward":_safe_float(reward,0.0),"done":bool(done),"task_success":_safe_float(info.get("task_success",0.0),0.0),"last_action_success":_safe_int(info.get("last_action_success",0),0),"distance":_safe_float(info.get("distance",-1.0),-1.0),"collision":_safe_int(info.get("collision",0),0)} | dbg
                    transitions.append(row)
                    with trans_path.open("a",encoding="utf-8") as f: f.write(json.dumps(row, ensure_ascii=False)+"\n"); f.flush()
                    image_hist=(image_hist+[next_path])[-int(args.history_len):]; action_hist=(action_hist+[action_id])[-int(args.history_len):]
                    obs=next_obs; step_idx+=1
                ep={"rollout_id":rollout_id,"episode_id":int(item.get("episode_id",-1)),"task_key":str(item.get("task_key","")),"eval_set":eval_set,"instruction":instruction,"steps":step_idx,"episode_reward":ep_reward,"task_success":_safe_float(info.get("task_success",0.0),0.0),"collision":_safe_int(info.get("collision",0),0),"final_distance":_safe_float(info.get("distance",-1.0),-1.0)}
                episodes.append(ep)
                with eps_path.open("a",encoding="utf-8") as f: f.write(json.dumps(ep, ensure_ascii=False)+"\n"); f.flush()
                print(f"[{rollout_id}/{len(selected)}] task={item.get('task_key','')} success={ep['task_success']} steps={step_idx}", flush=True)
        finally:
            try: env.env.stop()
            except Exception: pass
    succ_steps={True:[],False:[]}; succ_vu={True:[],False:[]}; succ_pu={True:[],False:[]}
    ep_success={int(e['rollout_id']): bool(float(e.get('task_success',0.0))) for e in episodes}
    for t in transitions:
        s=ep_success.get(int(t['rollout_id']), False); succ_vu[s].append(float(t.get('selected_score_std',0.0))); succ_pu[s].append(float(t.get('selected_pred_uncertainty',0.0)))
    def mean(xs): return sum(xs)/max(1,len(xs))
    summary={"wm_checkpoint":str(resolve_repo_path(args.wm_checkpoint)),"value_checkpoints":[str(p) for p in value_paths],"value_ensemble_size":len(value_paths),"mode":args.mode,"risk_lambda":float(args.risk_lambda),"visual_encoder":args.visual_encoder,"visual_dim":visual_dim,"num_selected_episodes":len(selected),"num_collected_episodes":len(episodes),"num_transitions":len(transitions),"task_success_rate":sum(float(e.get('task_success',0.0)) for e in episodes)/max(1,len(episodes)),"collision_rate":sum(1 for e in episodes if int(e.get('collision',0)))/max(1,len(episodes)),"planner_failure_rate":planner_failures/max(1,len(transitions)),"planner_failures":planner_failures,"action_distribution":{str(k):int(v) for k,v in sorted(action_counts.items())},"mean_selected_value_uncertainty":mean([float(t.get('selected_score_std',0.0)) for t in transitions]),"mean_selected_pred_uncertainty":mean([float(t.get('selected_pred_uncertainty',0.0)) for t in transitions]),"value_unc_success":mean(succ_vu[True]),"value_unc_fail":mean(succ_vu[False]),"pred_unc_success":mean(succ_pu[True]),"pred_unc_fail":mean(succ_pu[False]),"outputs":{"transitions_jsonl":str(trans_path),"episodes_jsonl":str(eps_path),"summary_json":str(out/'summary.json')}}
    (out/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)); print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
