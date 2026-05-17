"""Online EB-Nav end-to-end eval for frozen-Qwen/frozen-WM action-ranking policy."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.collect_eb_nav_random_rollouts import (  # noqa: E402
    _build_record_meta,
    _extract_rgb,
    _import_eb_navigation_env,
    _matches_split,
    _safe_float,
    _safe_int,
    _save_rgb,
)
from dev.eval_eb_nav_value_action_ranking import one_hot  # noqa: E402
from dev.train_eb_nav_action_ranking_head import load_head_init  # noqa: E402
from dev.train_eb_nav_joint_wm_value import freeze_qwen  # noqa: E402
from dev.train_eb_nav_value_head_predicted import SemanticWMValueHead, build_wm_from_checkpoint, encode_many, resolve_repo_path  # noqa: E402
from src.data.eb_nav_dataset import ACTION_NAMES  # noqa: E402
from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402


def select_records(records: list[dict[str, Any]], split: str, num_episodes: int, seed: int) -> list[dict[str, Any]]:
    metas = [_build_record_meta(r, i) for i, r in enumerate(records) if _matches_split(r, split)]
    rng = random.Random(seed); rng.shuffle(metas)
    return metas[: int(num_episodes)]


@torch.no_grad()
def choose_action(
    *,
    image_history: list[str],
    action_history: list[int],
    instruction: str,
    visual_encoder: QwenLLMLatentEncoder,
    semantic_encoder: QwenLLMLatentEncoder,
    wm: torch.nn.Module,
    head: SemanticWMValueHead,
    device: torch.device,
) -> tuple[int, list[float]]:
    hist_len = len(image_history)
    z_hist = encode_many(visual_encoder, image_history, None, device).reshape(1, hist_len, 16, 3584)
    semantic = encode_many(semantic_encoder, [image_history[-1]], [instruction], device, expected_flat_dim=3584).reshape(1, 3584)
    z_current = z_hist[:, -1]
    hist_actions = torch.zeros(1, hist_len, 8, dtype=torch.float32, device=device)
    for i, a in enumerate(action_history[-hist_len:]):
        if 0 <= int(a) < 8:
            hist_actions[0, i, int(a)] = 1.0
    scores=[]
    for action_id in range(8):
        ids = torch.tensor([action_id], dtype=torch.long, device=device)
        action_vec = one_hot(ids, 8)
        teacher_action = hist_actions.clone()
        teacher_action[:, -1, :] = action_vec
        z_next = wm.predict_next(z_hist, teacher_action)
        score = head(semantic, z_current, z_next, action_vec)
        scores.append(float(score.item()))
    best = int(max(range(8), key=lambda i: scores[i]))
    return best, scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="datasets/EB-Nav/eb-nav_dataset_single_step.json")
    p.add_argument("--embodiedbench-root", default="/project/peilab/atst/EmbodiedBench")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--split", choices=["all", "train", "test"], default="test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history-len", type=int, default=4)
    p.add_argument("--resolution", type=int, default=500)
    p.add_argument("--fov", type=int, default=100)
    p.add_argument("--exp-name", default="eval_action_rank_e2e")
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-dtype", default="auto")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--save-screenshots", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out=Path(args.output_dir); shots=out/"step_screenshots"; out.mkdir(parents=True, exist_ok=True); shots.mkdir(parents=True, exist_ok=True)
    (out/"args.json").write_text(json.dumps(vars(args), indent=2))
    device = torch.device("cpu" if str(args.cuda_device) in {"", "-1", "cpu"} else f"cuda:{args.cuda_device}")

    ckpt_path=resolve_repo_path(args.checkpoint)
    wm=build_wm_from_checkpoint(ckpt_path, device); wm.eval()
    ckpt=torch.load(ckpt_path, map_location="cpu")
    if "wm_state" in ckpt:
        wm.load_state_dict(ckpt["wm_state"], strict=False)
    head=SemanticWMValueHead(semantic_dim=3584, visual_dim=3584, action_dim=8, hidden=512).to(device)
    if "head_state" in ckpt:
        head.load_state_dict(ckpt["head_state"], strict=True)
    else:
        load_head_init(head, str(ckpt_path))
    head.eval()

    adapter=QwenVLMAdapter(model_name=args.model_name, latent_dim=57344, enabled=True, fallback_enabled=False, device_map=None if str(args.device_map).lower() in {"", "none"} else args.device_map, model_dtype=args.model_dtype)
    freeze_qwen(adapter)
    visual_encoder=QwenLLMLatentEncoder(57344, name="qwen_visual_tokens", model_name=args.model_name, qwen_adapter=adapter, use_vision_only=True, visual_pooling="tokens", visual_num_tokens=16, cache_latents=True)
    semantic_encoder=QwenLLMLatentEncoder(3584, name="qwen_llm_semantic", model_name=args.model_name, qwen_adapter=adapter, use_vision_only=False, visual_pooling="last", cache_latents=True)

    records=json.load(open(args.dataset, encoding="utf-8"))
    selected=select_records(records, args.split, int(args.num_episodes), int(args.seed))
    by_eval: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in selected:
        by_eval[str(m["eval_set"])].append(m)
    EBNavigationEnv=_import_eb_navigation_env(args.embodiedbench_root)
    transitions=[]; episodes=[]; action_counts=Counter(); rollout_id=0
    trans_path=out/"transitions.jsonl"; eps_path=out/"episodes.jsonl"
    trans_path.write_text(""); eps_path.write_text("")

    for eval_set, items in sorted(by_eval.items()):
        env=EBNavigationEnv(eval_set=eval_set, exp_name=f"{args.exp_name}_{eval_set}", selected_indexes=[max(0,int(x["episode_id"])-1) for x in items], resolution=int(args.resolution), fov=int(args.fov))
        env._max_episode_steps=int(args.max_steps)
        try:
            for item in items:
                rollout_id += 1
                obs=env.reset(); done=False; step_idx=0; info={}; ep_reward=0.0
                instruction=str(item.get("instruction") or getattr(env, "episode_language_instruction", ""))
                rgb=_extract_rgb(obs)
                cur_path=_save_rgb(shots/f"rollout_{rollout_id:04d}_step_{step_idx:03d}_obs.png", rgb)
                image_hist=[cur_path]*int(args.history_len)
                action_hist=[-1]*int(args.history_len)
                while not done and step_idx < int(args.max_steps):
                    action_id, scores = choose_action(image_history=image_hist, action_history=action_hist, instruction=instruction, visual_encoder=visual_encoder, semantic_encoder=semantic_encoder, wm=wm, head=head, device=device)
                    next_obs, reward, done, info = env.step(action_id, {"policy":"action_ranking", "task_key":str(item.get("task_key",""))}, 1)
                    ep_reward += float(reward)
                    action_counts[action_id] += 1
                    next_rgb=_extract_rgb(next_obs)
                    next_path=_save_rgb(shots/f"rollout_{rollout_id:04d}_step_{step_idx:03d}_next.png", next_rgb)
                    row={"rollout_id":rollout_id,"episode_id":int(item.get("episode_id",-1)),"task_key":str(item.get("task_key","")),"eval_set":eval_set,"instruction":instruction,"step":step_idx,"image_t":image_hist[-1],"image_next":next_path,"sampled_action_id":action_id,"sampled_action_name":ACTION_NAMES.get(action_id,f"action_{action_id}"),"scores":scores,"reward":_safe_float(reward,0.0),"done":bool(done),"task_success":_safe_float(info.get("task_success",0.0),0.0),"last_action_success":_safe_int(info.get("last_action_success",0),0),"distance":_safe_float(info.get("distance",-1.0),-1.0),"collision":_safe_int(info.get("collision",0),0)}
                    transitions.append(row)
                    with trans_path.open("a", encoding="utf-8") as f: f.write(json.dumps(row, ensure_ascii=False)+"\n"); f.flush()
                    image_hist=(image_hist+[next_path])[-int(args.history_len):]
                    action_hist=(action_hist+[action_id])[-int(args.history_len):]
                    obs=next_obs; step_idx += 1
                ep={"rollout_id":rollout_id,"episode_id":int(item.get("episode_id",-1)),"task_key":str(item.get("task_key","")),"eval_set":eval_set,"instruction":instruction,"steps":step_idx,"episode_reward":ep_reward,"task_success":_safe_float(info.get("task_success",0.0),0.0),"collision":_safe_int(info.get("collision",0),0),"final_distance":_safe_float(info.get("distance",-1.0),-1.0)}
                episodes.append(ep)
                with eps_path.open("a", encoding="utf-8") as f: f.write(json.dumps(ep, ensure_ascii=False)+"\n"); f.flush()
                print(f"[{rollout_id}/{len(selected)}] task={item.get('task_key','')} success={ep['task_success']} steps={step_idx}", flush=True)
        finally:
            try: env.env.stop()
            except Exception: pass
    summary={"checkpoint":str(ckpt_path),"num_selected_episodes":len(selected),"num_collected_episodes":len(episodes),"num_transitions":len(transitions),"task_success_rate":sum(float(e.get("task_success",0.0)) for e in episodes)/max(1,len(episodes)),"collision_rate":sum(1 for e in episodes if int(e.get("collision",0)))/max(1,len(episodes)),"action_distribution":{str(k):int(v) for k,v in sorted(action_counts.items())},"outputs":{"transitions_jsonl":str(trans_path),"episodes_jsonl":str(eps_path),"summary_json":str(out/"summary.json")}}
    (out/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
