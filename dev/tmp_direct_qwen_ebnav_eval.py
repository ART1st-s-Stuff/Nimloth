from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path("/project/peilab/atst/flower")
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
from src.data.eb_nav_dataset import ACTION_NAMES  # noqa: E402
from src.vlm.qwen_adapter import QwenVLMAdapter  # noqa: E402
from src.vlm.qwen_planner import (  # noqa: E402
    generate_planner_response,
    validate_planner_special_output,
)


BASELINE_PROMPT_TEMPLATE = """## You are a robot operating in a home. You can do various tasks and output a sequence of actions to accomplish a given task with images of your status.

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


RECOVERY_PROMPT_TEMPLATE = """You are controlling a robot in a home from a first-person image.

Available actions:
0: Move forward by 0.25
1: Move backward by 0.25
2: Move right by 0.25
3: Move left by 0.25
4: Rotate right by 90 degrees
5: Rotate left by 90 degrees
6: Look up by 30 degrees
7: Look down by 30 degrees

Goal instruction: {instruction}

Recent action history:
{action_history}

Decision rules:
1. Output exactly one action.
2. If the target is not visible or not in the forward view, rotate once to search or center it.
3. If the last Move forward failed or hit an obstacle, do not choose Move forward again from the same view; rotate or strafe.
4. Use Move forward only when the image shows a clear traversable path toward the target.
5. Use side steps to adjust lateral offset; use look up/down only when the target is vertically out of view.
6. Do not rely on distance values. Use only the image, task instruction, and recent action outcomes.

Examples:
- Target appears on the right side of the image -> choose action 4.
- Target appears on the left side of the image -> choose action 5.
- Last action was Move forward and it failed -> choose action 4, 5, 2, or 3, not action 0.
- Target is centered with open floor ahead -> choose action 0.
- A wall or object blocks the center path -> choose action 4 or 5.

Respond with the trained planner format only:
<think>brief visual reasoning and recovery reasoning</think><|latent_token|><|action_start|><|action_N|><|action_end|>
Replace N with one id from 0 to 7."""


BALANCED_FEWSHOT_PROMPT_TEMPLATE = """You are controlling a robot in a home from a first-person image.

Available action ids:
0 Move forward, 1 Move backward, 2 Move right, 3 Move left, 4 Rotate right, 5 Rotate left, 6 Look up, 7 Look down.

Goal instruction: {instruction}

Recent action history:
{action_history}

Choose exactly one action. First decide whether the target is visible and centered. Then decide whether forward motion is safe. Avoid repeating the same failed action.

Few-shot format examples:
Observation: the target is not visible after facing a wall.
Output: <think>The target is not visible and the center is blocked, so I should rotate to inspect another direction.</think><|latent_token|><|action_start|><|action_4|><|action_end|>

Observation: the target is visible on the left edge.
Output: <think>The target is left of the camera center, so I should rotate left before moving.</think><|latent_token|><|action_start|><|action_5|><|action_end|>

Observation: the previous forward move failed in the same view.
Output: <think>Forward was blocked, so repeating it is unlikely to help. I should sidestep to search for a free path.</think><|latent_token|><|action_start|><|action_2|><|action_end|>

Observation: the target is centered and the floor ahead is open.
Output: <think>The target is centered with a clear path, so a single forward step is appropriate.</think><|latent_token|><|action_start|><|action_0|><|action_end|>

Now respond for the current image using the same format:
<think>brief reasoning</think><|latent_token|><|action_start|><|action_N|><|action_end|>"""


SUCCESS_HISTORY_PROMPT_TEMPLATE = """You are a robot navigating a home from a first-person image.

Available actions:
0: Move forward by 0.25
1: Move backward by 0.25
2: Move rightward by 0.25
3: Move leftward by 0.25
4: Rotate to the right by 90 degrees
5: Rotate to the left by 90 degrees
6: Tilt the camera upward by 30 degrees
7: Tilt the camera downward by 30 degrees

Goal instruction:
{instruction}

Complete action history for this episode before the current image:
{action_history}

Choose the single next expert navigation action for the current image.
Use the image, goal instruction, and complete action history. If a previous
action failed, avoid repeating the same blocked move unless the view has changed.
Do not use distance-to-goal values.

Respond exactly in this format:
<think>brief expert navigation reasoning</think><|latent_token|><|action_start|><|action_N|><|action_end|>
where N is one action id from 0 to 7."""


PROMPT_TEMPLATES = {
    "baseline": BASELINE_PROMPT_TEMPLATE,
    "recovery": RECOVERY_PROMPT_TEMPLATE,
    "balanced_fewshot": BALANCED_FEWSHOT_PROMPT_TEMPLATE,
    "success_history": SUCCESS_HISTORY_PROMPT_TEMPLATE,
}


def format_action_history(history: list[dict[str, Any]], limit: int | None = None) -> str:
    if not history:
        return "No previous actions in this episode."
    lines = []
    selected = history if limit is None else history[-int(limit) :]
    for item in selected:
        action_id = int(item.get("action_id", -1))
        name = ACTION_NAMES.get(action_id, f"action_{action_id}")
        ok = "success" if int(item.get("last_action_success", 0)) else "failed"
        collision = int(item.get("collision", 0))
        feedback = " with collision" if collision else ""
        lines.append(f"{int(item.get('step', -1)):02d}. action {action_id} ({name}) -> {ok}{feedback}")
    return "\n".join(lines)


def build_prompt(variant: str, *, instruction: str, action_history: list[dict[str, Any]]) -> str:
    template = PROMPT_TEMPLATES[str(variant)]
    history_limit = None if str(variant) == "success_history" else 4
    return template.format(
        instruction=instruction,
        action_history=format_action_history(action_history, limit=history_limit),
    )


def select_records(records: list[dict[str, Any]], split: str, num_episodes: int, seed: int) -> list[dict[str, Any]]:
    metas = [_build_record_meta(r, i) for i, r in enumerate(records) if _matches_split(r, split)]
    rng = random.Random(seed)
    rng.shuffle(metas)
    return metas[: int(num_episodes)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct SFT-Qwen closed-loop EB-Nav eval without WM/value head.")
    p.add_argument("--dataset", default="datasets/EB-Nav/eb-nav_dataset_single_step.json")
    p.add_argument("--embodiedbench-root", default="/project/peilab/atst/EmbodiedBench")
    p.add_argument("--planner-lora", default="models/qwen_planner_lora")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--split", choices=["all", "train", "test"], default="test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resolution", type=int, default=500)
    p.add_argument("--fov", type=int, default=100)
    p.add_argument("--exp-name", default="eval_direct_qwen_sft")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-dtype", default="auto")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--prompt-variant", choices=sorted(PROMPT_TEMPLATES), default="baseline")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    shots = out / "step_screenshots"
    tmp_obs = out / "tmp_obs"
    out.mkdir(parents=True, exist_ok=True)
    shots.mkdir(parents=True, exist_ok=True)
    tmp_obs.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    records = json.load(open(args.dataset, encoding="utf-8"))
    selected = select_records(records, args.split, int(args.num_episodes), int(args.seed))
    by_eval: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        by_eval[str(item["eval_set"])].append(item)

    adapter = QwenVLMAdapter(
        model_name=args.model_name,
        latent_dim=4096,
        enabled=True,
        fallback_enabled=False,
        device_map=None if str(args.device_map).lower() in {"", "none"} else args.device_map,
        model_dtype=args.model_dtype,
    )
    adapter._ensure_model()
    if adapter._model is None or adapter._processor is None:
        raise RuntimeError(f"Failed to load Qwen model: {adapter.init_error}")
    adapter.load_lora_adapter(args.planner_lora, trainable=False)
    adapter._model.eval()

    EBNavigationEnv = _import_eb_navigation_env(args.embodiedbench_root)
    transitions = []
    episodes = []
    action_counts = Counter()
    planner_failures = 0
    blocked_forward_count = 0
    blocked_forward_repeat_count = 0
    repeated_action_steps = 0
    longest_repeat_streak = 0
    rollout_id = 0
    trans_path = out / "transitions.jsonl"
    eps_path = out / "episodes.jsonl"
    trans_path.write_text("")
    eps_path.write_text("")

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
                action_history: list[dict[str, Any]] = []
                prev_action_id: int | None = None
                repeat_streak = 0
                prev_forward_failed = False
                while not done and step_idx < int(args.max_steps):
                    rgb = _extract_rgb(obs)
                    cur_path = _save_rgb(shots / f"rollout_{rollout_id:04d}_step_{step_idx:03d}_obs.png", rgb)
                    prompt = build_prompt(
                        args.prompt_variant,
                        instruction=instruction,
                        action_history=action_history,
                    )
                    response = generate_planner_response(
                        model=adapter._model,
                        processor=adapter._processor,
                        image_path=str(cur_path),
                        prompt=prompt,
                        max_new_tokens=int(args.max_new_tokens),
                    )
                    valid, reason, action_id = validate_planner_special_output(response)
                    if not valid or action_id is None:
                        planner_failures += 1
                        action_id = 0
                    next_obs, reward, done, info = env.step(
                        int(action_id),
                        {"policy": "direct_qwen_planner", "task_key": str(item.get("task_key", ""))},
                        1,
                    )
                    ep_reward += float(reward)
                    action_counts[int(action_id)] += 1
                    if prev_action_id is not None and int(action_id) == int(prev_action_id):
                        repeat_streak += 1
                        repeated_action_steps += 1
                    else:
                        repeat_streak = 1
                    longest_repeat_streak = max(longest_repeat_streak, repeat_streak)
                    if int(action_id) == 0 and _safe_int(info.get("last_action_success", 0), 0) == 0:
                        blocked_forward_count += 1
                    if prev_forward_failed and int(action_id) == 0:
                        blocked_forward_repeat_count += 1
                    prev_forward_failed = (
                        int(action_id) == 0
                        and _safe_int(info.get("last_action_success", 0), 0) == 0
                    )
                    prev_action_id = int(action_id)
                    next_rgb = _extract_rgb(next_obs)
                    next_path = _save_rgb(shots / f"rollout_{rollout_id:04d}_step_{step_idx:03d}_next.png", next_rgb)
                    row = {
                        "rollout_id": rollout_id,
                        "episode_id": int(item.get("episode_id", -1)),
                        "task_key": str(item.get("task_key", "")),
                        "eval_set": eval_set,
                        "instruction": instruction,
                        "step": step_idx,
                        "image_t": str(cur_path),
                        "image_next": str(next_path),
                        "planner_response": response[:4000],
                        "planner_valid": bool(valid),
                        "planner_error": "" if valid else reason,
                        "prompt_variant": str(args.prompt_variant),
                        "sampled_action_id": int(action_id),
                        "sampled_action_name": ACTION_NAMES.get(int(action_id), f"action_{action_id}"),
                        "reward": _safe_float(reward, 0.0),
                        "done": bool(done),
                        "task_success": _safe_float(info.get("task_success", 0.0), 0.0),
                        "last_action_success": _safe_int(info.get("last_action_success", 0), 0),
                        "distance": _safe_float(info.get("distance", -1.0), -1.0),
                        "collision": _safe_int(info.get("collision", 0), 0),
                        "repeat_streak": int(repeat_streak),
                        "prev_forward_failed": bool(prev_forward_failed),
                    }
                    action_history.append(
                        {
                            "step": step_idx,
                            "action_id": int(action_id),
                            "last_action_success": row["last_action_success"],
                            "collision": row["collision"],
                        }
                    )
                    transitions.append(row)
                    with trans_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
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
                print(f"[{rollout_id}/{len(selected)}] task={item.get('task_key', '')} success={ep['task_success']} steps={step_idx}", flush=True)
        finally:
            try:
                env.env.stop()
            except Exception:
                pass

    summary = {
        "planner_lora": args.planner_lora,
        "prompt_variant": args.prompt_variant,
        "num_selected_episodes": len(selected),
        "num_collected_episodes": len(episodes),
        "num_transitions": len(transitions),
        "task_success_rate": sum(float(e.get("task_success", 0.0)) for e in episodes) / max(1, len(episodes)),
        "collision_rate": sum(1 for e in episodes if int(e.get("collision", 0))) / max(1, len(episodes)),
        "planner_failure_rate": planner_failures / max(1, len(transitions)),
        "planner_failures": planner_failures,
        "action_distribution": {str(k): int(v) for k, v in sorted(action_counts.items())},
        "blocked_forward_count": blocked_forward_count,
        "blocked_forward_repeat_count": blocked_forward_repeat_count,
        "repeated_action_steps": repeated_action_steps,
        "longest_repeat_streak": longest_repeat_streak,
        "outputs": {
            "transitions_jsonl": str(trans_path),
            "episodes_jsonl": str(eps_path),
            "summary_json": str(out / "summary.json"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
