"""Single-run EB-Nav eval with conservative learned override on top of Qwen.

Default controller:
  direct Qwen planner action.

Override rule:
  replace Qwen's action only when the learned WM/value policy prefers another
  action by a clear margin and the selected action is low-uncertainty.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

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
from dev.collect_eb_nav_uncertainty_fork_rollouts import expand_checkpoints, load_value_members  # noqa: E402
from dev.evaluate_eb_nav_wm_value_ensemble_e2e import choose_action  # noqa: E402
from dev.train_eb_nav_joint_wm_value import freeze_qwen  # noqa: E402
from dev.train_eb_nav_value_head_predicted import (  # noqa: E402
    NUM_PATCHES,
    QWEN_VISUAL_DIM,
    build_visual_encoder,
    build_wm_from_checkpoint,
    resolve_repo_path,
)
from src.data.eb_nav_dataset import ACTION_NAMES  # noqa: E402
from src.vlm.qwen_adapter import QwenVLM  # noqa: E402
from src.vlm.qwen_planner import generate_planner_response, validate_planner_special_output  # noqa: E402
from src.wm.encoder.qwen import QwenLLMLatentEncoder  # noqa: E402


PROMPT_TEMPLATE = """## You are a robot operating in a home. You can do various tasks and output a sequence of actions to accomplish a given task with images of your status.

## The available action id (0 ~ 7) and action names are:
action id 0: Move forward by 0.25,
action id 1: Move backward by 0.25,
action id 2: Move rightward by 0.25,
action id 3: Move leftward by 0.25,
action id 4: Rotate to the right by 90 degrees.,
action id 5: Rotate to the left by 90 degrees.,
action id 6: Tilt the camera upward by 30 degrees.,
action id 7: Tilt the camera downward by 30 degrees..

*** Strategy ***

1. Locate the Target Object Type: Clearly describe the spatial location of the target object from the observation image.
2. Navigate by using Move forward and Move right/left as main strategy.
3. Focus on primary goal and do not over-correct unless invalid actions block progress.
4. Use Rotation Sparingly, only when you lose track of the target object.
5. Do not complete task too early until you cannot move any closer to the object.

----------

## Now the human instruction is: {instruction}
To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided.
Aim for about 1-2 actions in this step. Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.
At last, output the action id(s) (0 ~ 7) from the available actions to execute.

The input given to you is an first person view observation. Plan accordingly based on the visual observation.

You are supposed to output in JSON."""

SPECIAL_TOKEN_PROMPT_TEMPLATE = """You are a robot operating in a home from a first-person image.

Available action ids:
0 Move forward, 1 Move backward, 2 Move right, 3 Move left, 4 Rotate right, 5 Rotate left, 6 Look up, 7 Look down.

Goal instruction:
{instruction}

Complete action history before the current image:
{action_history}

Choose exactly one next action using the image, goal instruction, and action history. Do not use distance-to-goal values.

Respond exactly in this format:
<think>brief visual and task reasoning</think><|latent_token|><|action_start|><|action_N|><|action_end|>
where N is one action id from 0 to 7."""


def select_records(
    records: list[dict[str, Any]],
    *,
    split: str,
    seed: int,
    num_episodes: int,
    eval_set: str,
    episode_id: int,
) -> list[dict[str, Any]]:
    metas = [_build_record_meta(r, i) for i, r in enumerate(records) if _matches_split(r, split)]
    if eval_set:
        metas = [m for m in metas if str(m.get("eval_set")) == eval_set]
    if episode_id > 0:
        metas = [m for m in metas if int(m.get("episode_id", -1)) == int(episode_id)]
    if not metas:
        raise RuntimeError(f"no matching EB-Nav records: split={split} eval_set={eval_set!r} episode_id={episode_id}")
    rng = random.Random(seed)
    rng.shuffle(metas)
    return metas[: int(num_episodes)]


def format_action_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "No previous actions in this episode."
    lines = []
    for item in history:
        action_id = int(item.get("action_id", -1))
        name = ACTION_NAMES.get(action_id, f"action_{action_id}")
        ok = "success" if int(item.get("last_action_success", 0)) else "failed"
        lines.append(f"{int(item.get('step', -1)):02d}. action {action_id} ({name}) -> {ok}")
    return "\n".join(lines)


def qwen_action_from_image(
    *,
    adapter: QwenVLM,
    image_path: str,
    instruction: str,
    action_history: list[dict[str, Any]],
    prompt_format: str,
    max_new_tokens: int,
) -> tuple[int, dict[str, Any]]:
    if str(prompt_format) == "special_token":
        prompt = SPECIAL_TOKEN_PROMPT_TEMPLATE.format(
            instruction=instruction,
            action_history=format_action_history(action_history),
        )
    else:
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    response = generate_planner_response(
        model=adapter._model,
        processor=adapter._processor,
        image_path=image_path,
        prompt=prompt,
        max_new_tokens=int(max_new_tokens),
    )
    valid, reason, action_id = validate_planner_special_output(response)
    if not valid or action_id is None:
        return 0, {"qwen_valid": False, "qwen_error": reason, "qwen_response": response[:4000]}
    return int(action_id), {"qwen_valid": True, "qwen_error": "", "qwen_response": response[:4000]}


def should_override(
    *,
    qwen_action: int,
    learned_action: int,
    dbg: dict[str, Any],
    margin: float,
    max_value_std: float,
    max_pred_uncertainty: float,
) -> tuple[bool, dict[str, Any]]:
    scores = [float(x) for x in dbg.get("policy_scores", [])]
    value_stds = [float(x) for x in dbg.get("score_std", [])]
    pred_unc = [float(x) for x in dbg.get("pred_uncertainty_by_action", [])]
    if len(scores) < 8 or len(value_stds) < 8 or len(pred_unc) < 8:
        return False, {"override_reason": "missing_learned_scores"}
    learned_margin = float(scores[learned_action] - scores[qwen_action])
    selected_value_std = float(value_stds[learned_action])
    selected_pred_unc = float(pred_unc[learned_action])
    ok = (
        int(learned_action) != int(qwen_action)
        and learned_margin >= float(margin)
        and selected_value_std <= float(max_value_std)
        and selected_pred_unc <= float(max_pred_uncertainty)
    )
    reason = "override" if ok else "fallback_qwen"
    if int(learned_action) == int(qwen_action):
        reason = "same_action"
    elif learned_margin < float(margin):
        reason = "margin_too_small"
    elif selected_value_std > float(max_value_std):
        reason = "value_uncertainty_too_high"
    elif selected_pred_unc > float(max_pred_uncertainty):
        reason = "wm_uncertainty_too_high"
    return ok, {
        "override_reason": reason,
        "override_margin_over_qwen": learned_margin,
        "override_selected_value_std": selected_value_std,
        "override_selected_pred_uncertainty": selected_pred_unc,
        "qwen_policy_score": float(scores[qwen_action]),
        "learned_policy_score": float(scores[learned_action]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="datasets/EB-Nav/eb-nav_dataset_single_step.json")
    p.add_argument("--embodiedbench-root", default="/project/peilab/atst/EmbodiedBench")
    p.add_argument("--wm-checkpoint", required=True)
    p.add_argument("--value-checkpoints", nargs="+", required=True)
    p.add_argument("--planner-lora", default="models/qwen_planner_lora_preaction_leakfree_8192_step00008000")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--split", choices=["all", "train", "test"], default="all")
    p.add_argument("--seed", type=int, default=20260522)
    p.add_argument("--eval-set", default="")
    p.add_argument("--episode-id", type=int, default=-1)
    p.add_argument("--history-len", type=int, default=4)
    p.add_argument("--resolution", type=int, default=500)
    p.add_argument("--fov", type=int, default=100)
    p.add_argument("--exp-name", default="qwen_override_single")
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-dtype", default="auto")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--visual-encoder", choices=["qwen", "dino"], default="qwen")
    p.add_argument("--dino-model-name", default="dinov2_vits14")
    p.add_argument("--dino-image-size", type=int, default=224)
    p.add_argument("--mode", choices=["fast", "planner", "hybrid"], default="hybrid")
    p.add_argument("--qwen-prompt-format", choices=["vagen_json", "special_token"], default="special_token")
    p.add_argument("--risk-lambda", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--override-margin", type=float, default=0.05)
    p.add_argument("--override-max-value-std", type=float, default=0.03)
    p.add_argument("--override-max-pred-uncertainty", type=float, default=0.08)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    shots = out / "step_screenshots"
    out.mkdir(parents=True, exist_ok=True)
    shots.mkdir(parents=True, exist_ok=True)
    value_paths = expand_checkpoints(args.value_checkpoints)
    (out / "args.json").write_text(
        json.dumps(vars(args) | {"expanded_value_checkpoints": [str(p) for p in value_paths]}, indent=2),
        encoding="utf-8",
    )
    if not value_paths:
        raise RuntimeError("no value checkpoints found")

    device = torch.device("cpu" if str(args.cuda_device) in {"", "-1", "cpu"} else f"cuda:{args.cuda_device}")
    first_value = torch.load(value_paths[0], map_location="cpu")
    visual_dim = int(first_value.get("visual_dim") or (384 if args.visual_encoder == "dino" else QWEN_VISUAL_DIM))
    semantic_dim = int(first_value.get("semantic_dim", QWEN_VISUAL_DIM))
    visual_latent_dim = NUM_PATCHES * visual_dim

    wm = build_wm_from_checkpoint(resolve_repo_path(args.wm_checkpoint), device, visual_dim=visual_dim, latent_dim=visual_latent_dim)
    wm.eval()
    heads = load_value_members(value_paths, device=device, visual_dim=visual_dim)

    qwen = QwenVLM(
        model_name=args.model_name,
        latent_dim=NUM_PATCHES * QWEN_VISUAL_DIM,
        enabled=True,
        fallback_enabled=False,
        device_map=None if str(args.device_map).lower() in {"", "none"} else args.device_map,
        model_dtype=args.model_dtype,
        max_new_tokens=int(args.max_new_tokens),
    )
    qwen._ensure_model()
    if qwen._model is None or qwen._processor is None:
        raise RuntimeError(f"Failed to load Qwen model: {qwen.init_error}")
    qwen.load_lora_adapter(str(resolve_repo_path(args.planner_lora)), trainable=False)
    qwen.planner_inference_mode = True
    qwen.max_new_tokens = int(args.max_new_tokens)
    freeze_qwen(qwen)

    visual_build_args = argparse.Namespace(**vars(args))
    visual_encoder, _, _ = build_visual_encoder(visual_build_args, qwen)
    no_cot_encoder = QwenLLMLatentEncoder(
        QWEN_VISUAL_DIM,
        name="qwen_no_cot",
        model_name=args.model_name,
        qwen_adapter=qwen,
        use_vision_only=False,
        visual_pooling="last",
        cache_latents=True,
    )

    records = json.load(open(args.dataset, encoding="utf-8"))
    selected = select_records(
        records,
        split=args.split,
        seed=int(args.seed),
        num_episodes=int(args.num_episodes),
        eval_set=str(args.eval_set),
        episode_id=int(args.episode_id),
    )
    by_eval: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        by_eval[str(item["eval_set"])].append(item)

    EBNavigationEnv = _import_eb_navigation_env(args.embodiedbench_root)
    transitions: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    action_counts: Counter[int] = Counter()
    override_counts: Counter[str] = Counter()
    planner_failures = 0
    rollout_id = 0
    trans_path = out / "transitions.jsonl"
    eps_path = out / "episodes.jsonl"
    trans_path.write_text("", encoding="utf-8")
    eps_path.write_text("", encoding="utf-8")

    for eval_set, items in sorted(by_eval.items()):
        env = EBNavigationEnv(
            eval_set=eval_set,
            exp_name=f"{args.exp_name}_{eval_set}",
            selected_indexes=[max(0, int(x["episode_id"]) - 1) for x in items],
            resolution=int(args.resolution),
            fov=int(args.fov),
        )
        env._max_episode_steps = int(args.max_steps)
        try:
            for item in items:
                rollout_id += 1
                obs = env.reset()
                done = False
                step_idx = 0
                info: dict[str, Any] = {}
                ep_reward = 0.0
                instruction = str(item.get("instruction") or getattr(env, "episode_language_instruction", ""))
                prompt = str(item.get("prompt") or item.get("input") or instruction)
                cur_path = _save_rgb(shots / f"rollout_{rollout_id:04d}_step_{step_idx:03d}_obs.png", _extract_rgb(obs))
                image_hist = [cur_path] * int(args.history_len)
                action_hist = [-1] * int(args.history_len)
                qwen_history: list[dict[str, Any]] = []

                while not done and step_idx < int(args.max_steps):
                    qwen_action, qwen_dbg = qwen_action_from_image(
                        adapter=qwen,
                        image_path=str(image_hist[-1]),
                        instruction=instruction,
                        action_history=qwen_history,
                        prompt_format=str(args.qwen_prompt_format),
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    planner_failures += int(not qwen_dbg["qwen_valid"])
                    learned_action, learned_dbg = choose_action(
                        image_history=[str(p) for p in image_hist],
                        action_history=action_hist,
                        prompt=prompt,
                        visual_encoder=visual_encoder,
                        no_cot_encoder=no_cot_encoder,
                        planner_adapter=qwen,
                        wm=wm,
                        heads=heads,
                        device=device,
                        mode=args.mode,
                        max_new_tokens=int(args.max_new_tokens),
                        visual_dim=visual_dim,
                        risk_lambda=float(args.risk_lambda),
                        semantic_dim=semantic_dim,
                    )
                    do_override, override_dbg = should_override(
                        qwen_action=qwen_action,
                        learned_action=learned_action,
                        dbg=learned_dbg,
                        margin=float(args.override_margin),
                        max_value_std=float(args.override_max_value_std),
                        max_pred_uncertainty=float(args.override_max_pred_uncertainty),
                    )
                    final_action = int(learned_action if do_override else qwen_action)
                    override_counts[override_dbg["override_reason"]] += 1

                    next_obs, reward, done, info = env.step(
                        final_action,
                        {
                            "policy": "qwen_override",
                            "task_key": str(item.get("task_key", "")),
                            "qwen_action": int(qwen_action),
                            "learned_action": int(learned_action),
                            "override": bool(do_override),
                        },
                        1,
                    )
                    ep_reward += float(reward)
                    action_counts[final_action] += 1
                    next_path = _save_rgb(shots / f"rollout_{rollout_id:04d}_step_{step_idx:03d}_next.png", _extract_rgb(next_obs))
                    last_action_success = _safe_int(info.get("last_action_success", 0), 0)
                    row = {
                        "rollout_id": rollout_id,
                        "episode_id": int(item.get("episode_id", -1)),
                        "task_key": str(item.get("task_key", "")),
                        "eval_set": eval_set,
                        "instruction": instruction,
                        "step": step_idx,
                        "image_t": str(image_hist[-1]),
                        "image_next": str(next_path),
                        "qwen_action_id": int(qwen_action),
                        "qwen_action_name": ACTION_NAMES.get(int(qwen_action), f"action_{qwen_action}"),
                        "learned_action_id": int(learned_action),
                        "learned_action_name": ACTION_NAMES.get(int(learned_action), f"action_{learned_action}"),
                        "sampled_action_id": final_action,
                        "sampled_action_name": ACTION_NAMES.get(final_action, f"action_{final_action}"),
                        "override": bool(do_override),
                        "reward": _safe_float(reward, 0.0),
                        "done": bool(done),
                        "task_success": _safe_float(info.get("task_success", 0.0), 0.0),
                        "last_action_success": last_action_success,
                        "distance": _safe_float(info.get("distance", -1.0), -1.0),
                        "collision": _safe_int(info.get("collision", 0), 0),
                    } | qwen_dbg | override_dbg | learned_dbg
                    transitions.append(row)
                    with trans_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                    print(
                        json.dumps(
                            {
                                "step": step_idx,
                                "qwen": qwen_action,
                                "learned": learned_action,
                                "final": final_action,
                                "override": bool(do_override),
                                "reason": override_dbg["override_reason"],
                                "success": row["task_success"],
                                "done": bool(done),
                            }
                        ),
                        flush=True,
                    )
                    image_hist = (image_hist + [next_path])[-int(args.history_len):]
                    action_hist = (action_hist + [final_action])[-int(args.history_len):]
                    qwen_history.append(
                        {
                            "step": int(step_idx),
                            "action_id": int(final_action),
                            "last_action_success": int(last_action_success),
                        }
                    )
                    obs = next_obs
                    step_idx += 1

                ep = {
                    "rollout_id": rollout_id,
                    "episode_id": int(item.get("episode_id", -1)),
                    "task_key": str(item.get("task_key", "")),
                    "eval_set": eval_set,
                    "instruction": instruction,
                    "steps": step_idx,
                    "episode_reward": ep_reward,
                    "task_success": _safe_float(info.get("task_success", 0.0), 0.0),
                    "collision": _safe_int(info.get("collision", 0), 0),
                    "final_distance": _safe_float(info.get("distance", -1.0), -1.0),
                }
                episodes.append(ep)
                with eps_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(ep, ensure_ascii=False) + "\n")
                    f.flush()
                print(f"[{rollout_id}/{len(selected)}] success={ep['task_success']} steps={step_idx}", flush=True)
        finally:
            try:
                env.env.stop()
            except Exception:
                pass

    summary = {
        "wm_checkpoint": str(resolve_repo_path(args.wm_checkpoint)),
        "value_checkpoints": [str(p) for p in value_paths],
        "planner_lora": args.planner_lora,
        "num_selected_episodes": len(selected),
        "num_collected_episodes": len(episodes),
        "num_transitions": len(transitions),
        "task_success_rate": sum(float(e.get("task_success", 0.0)) for e in episodes) / max(1, len(episodes)),
        "collision_rate": sum(1 for e in episodes if int(e.get("collision", 0))) / max(1, len(episodes)),
        "planner_failure_rate": planner_failures / max(1, len(transitions)),
        "planner_failures": planner_failures,
        "override_rate": sum(1 for t in transitions if bool(t.get("override", False))) / max(1, len(transitions)),
        "same_as_qwen_rate": sum(1 for t in transitions if int(t.get("sampled_action_id", -1)) == int(t.get("qwen_action_id", -2))) / max(1, len(transitions)),
        "action_distribution": {str(k): int(v) for k, v in sorted(action_counts.items())},
        "override_distribution": {str(k): int(v) for k, v in sorted(override_counts.items())},
        "outputs": {
            "transitions_jsonl": str(trans_path),
            "episodes_jsonl": str(eps_path),
            "summary_json": str(out / "summary.json"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
