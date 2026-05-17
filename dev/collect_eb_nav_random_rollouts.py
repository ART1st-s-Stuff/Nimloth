"""Collect EB-Nav simulator rollouts with random/semi-random policies.

This script collects off-policy/random transitions from EmbodiedBench
EBNavigationEnv for later world-model/Q-head training experiments.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.data.eb_nav_dataset import ACTION_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/EB-Nav/eb-nav_dataset_single_step.json")
    parser.add_argument("--embodiedbench-root", default="/home/jincai_guo/atst/EmbodiedBench")
    parser.add_argument("--output-dir", default="outputs/dev/eb_nav_random_rollouts")
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--policy", choices=["random", "action_balanced", "epsilon_logged", "repeat_mixed"], default="random")
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=500)
    parser.add_argument("--fov", type=int, default=100)
    parser.add_argument("--exp-name", default="collect_eb_nav_random")
    parser.add_argument("--balance-scope", choices=["global", "rollout"], default="global")
    parser.add_argument("--repeat-prob", type=float, default=0.25)
    parser.add_argument("--repeat-logged-prob", type=float, default=0.5)
    parser.add_argument("--save-screenshots", dest="save_screenshots", action="store_true")
    parser.add_argument("--no-save-screenshots", dest="save_screenshots", action="store_false")
    parser.set_defaults(save_screenshots=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _import_eb_navigation_env(embodiedbench_root: str):
    root = str(Path(embodiedbench_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv

    return EBNavigationEnv


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_logged_actions(record: dict[str, Any]) -> list[int]:
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, list):
        return []
    actions: list[int] = []
    for step in trajectory:
        plans = step.get("executable_plan") if isinstance(step, dict) else None
        if not isinstance(plans, list) or not plans:
            continue
        plan = plans[0] if isinstance(plans[0], dict) else {}
        action = plan.get("action", [0, ""])
        if isinstance(action, list) and action:
            actions.append(_safe_int(action[0], default=0))
    return actions


def _record_eval_set(record: dict[str, Any]) -> str:
    for key in ("eval_set", "split", "subset"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _matches_split(record: dict[str, Any], split: str) -> bool:
    if split == "all":
        return True
    eval_set = _record_eval_set(record).lower()
    if split == "train":
        return "train" in eval_set
    if split == "test":
        return "test" in eval_set or "heldout" in eval_set or "val" in eval_set
    return True


def _build_record_meta(record: dict[str, Any], record_idx: int) -> dict[str, Any]:
    eval_set = _record_eval_set(record)
    episode_id = _safe_int(record.get("episode_id", record_idx + 1), default=record_idx + 1)
    return {
        "record_idx": int(record_idx),
        "episode_id": int(episode_id),
        "eval_set": eval_set,
        "task_key": f"{eval_set}:{episode_id:03d}",
        "instruction": str(record.get("instruction", "")),
        "model_name": str(record.get("model_name", "")),
        "logged_actions": _extract_logged_actions(record),
    }


def _select_records(*, records: list[dict[str, Any]], split: str, num_episodes: int, seed: int) -> list[dict[str, Any]]:
    metas = [_build_record_meta(record, idx) for idx, record in enumerate(records) if _matches_split(record, split)]
    if not metas:
        return []
    rng = random.Random(seed)
    rng.shuffle(metas)

    if split == "all":
        return metas[:num_episodes]
    return metas[:num_episodes]


def _extract_rgb(obs: Any) -> np.ndarray | None:
    if isinstance(obs, dict):
        for key in ("head_rgb", "rgb", "image", "obs"):
            if key in obs:
                try:
                    return np.asarray(obs[key])
                except Exception:
                    continue
    try:
        return np.asarray(obs)
    except Exception:
        return None


def _save_rgb(path: Path, rgb: np.ndarray | None) -> str:
    if rgb is None:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(rgb).save(path)
        return str(path)
    except Exception:
        return ""


class ActionSampler:
    def __init__(self, *, args: argparse.Namespace, rng: random.Random):
        self.args = args
        self.rng = rng
        self.global_balanced_queue: list[int] = []
        self.global_counts: Counter[int] = Counter()

    def _rand_action(self) -> int:
        return int(self.rng.randrange(int(self.args.action_dim)))

    def _next_balanced(self, queue: list[int]) -> int:
        if not queue:
            queue.extend(list(range(int(self.args.action_dim))))
            self.rng.shuffle(queue)
        return int(queue.pop())

    def sample(
        self,
        *,
        step_idx: int,
        logged_actions: list[int],
        prev_action: int | None,
        rollout_balanced_queue: list[int],
    ) -> dict[str, Any]:
        policy = str(self.args.policy)
        logged_action = int(logged_actions[step_idx]) if step_idx < len(logged_actions) else None
        action_id: int

        if policy == "random":
            action_id = self._rand_action()
        elif policy == "action_balanced":
            queue = self.global_balanced_queue if self.args.balance_scope == "global" else rollout_balanced_queue
            action_id = self._next_balanced(queue)
        elif policy == "epsilon_logged":
            if logged_action is not None and self.rng.random() > float(self.args.epsilon):
                action_id = int(logged_action)
            else:
                action_id = self._rand_action()
        elif policy == "repeat_mixed":
            if prev_action is not None and self.rng.random() < float(self.args.repeat_prob):
                action_id = int(prev_action)
            else:
                use_logged = logged_action is not None and self.rng.random() < float(self.args.repeat_logged_prob)
                if use_logged:
                    action_id = int(logged_action) if logged_action is not None else self._rand_action()
                else:
                    action_id = self._rand_action()
        else:
            action_id = self._rand_action()

        action_id = int(max(0, min(int(self.args.action_dim) - 1, action_id)))
        self.global_counts[action_id] += 1
        return {
            "sampled_action_id": action_id,
            "sampled_action_name": ACTION_NAMES.get(action_id, f"action_{action_id}"),
            "logged_action_id": logged_action,
            "policy": policy,
            "epsilon": float(self.args.epsilon),
        }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))

    output_dir = Path(args.output_dir)
    screenshot_dir = output_dir / "step_screenshots"
    transitions_path = output_dir / "transitions.jsonl"
    episodes_path = output_dir / "episodes.jsonl"
    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_screenshots:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        transitions_path.write_text("", encoding="utf-8")
        episodes_path.write_text("", encoding="utf-8")

    records = json.load(open(args.dataset, encoding="utf-8"))
    if not isinstance(records, list):
        raise RuntimeError(f"dataset is not a list: {args.dataset}")

    selected = _select_records(
        records=records,
        split=str(args.split),
        num_episodes=int(args.num_episodes),
        seed=int(args.seed),
    )
    if not selected:
        raise RuntimeError(f"no records selected for split={args.split}")

    selected_by_eval: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        selected_by_eval[str(item["eval_set"])].append(item)

    if args.dry_run:
        summary = {
            "dry_run": True,
            "dataset": str(args.dataset),
            "selected_count": len(selected),
            "selected_preview": selected,
            "args": vars(args),
            "outputs": {
                "transitions_jsonl": str(transitions_path),
                "episodes_jsonl": str(episodes_path),
                "summary_json": str(summary_path),
                "step_screenshots_dir": str(screenshot_dir),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"dry-run selected {len(selected)} records")
        print(f"wrote {summary_path}")
        return

    EBNavigationEnv = _import_eb_navigation_env(args.embodiedbench_root)
    sampler = ActionSampler(args=args, rng=rng)
    transitions: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []

    rollout_id = 0
    for eval_set, items in sorted(selected_by_eval.items()):
        selected_indexes = [max(0, int(item["episode_id"]) - 1) for item in items]
        env = EBNavigationEnv(
            eval_set=eval_set,
            exp_name=f"{args.exp_name}_{eval_set}",
            selected_indexes=selected_indexes,
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
                prev_action: int | None = None
                rollout_balanced_queue: list[int] = []
                episode_reward = 0.0
                info: dict[str, Any] = {}

                instruction = str(item.get("instruction") or getattr(env, "episode_language_instruction", ""))
                while not done and step_idx < int(args.max_steps):
                    rgb_t = _extract_rgb(obs)
                    image_t_path = ""
                    if args.save_screenshots:
                        image_t_path = _save_rgb(
                            screenshot_dir / f"rollout_{rollout_id:04d}_step_{step_idx:03d}_t.png",
                            rgb_t,
                        )

                    sampled = sampler.sample(
                        step_idx=step_idx,
                        logged_actions=list(item.get("logged_actions", [])),
                        prev_action=prev_action,
                        rollout_balanced_queue=rollout_balanced_queue,
                    )
                    action_id = int(sampled["sampled_action_id"])
                    next_obs, reward, done, info = env.step(
                        action_id,
                        {"policy": str(args.policy), "task_key": str(item.get("task_key", ""))},
                        1,
                    )
                    rgb_next = _extract_rgb(next_obs)
                    image_next_path = ""
                    if args.save_screenshots:
                        image_next_path = _save_rgb(
                            screenshot_dir / f"rollout_{rollout_id:04d}_step_{step_idx:03d}_next.png",
                            rgb_next,
                        )

                    transition = {
                        "rollout_id": int(rollout_id),
                        "record_idx": int(item.get("record_idx", -1)),
                        "episode_id": int(item.get("episode_id", -1)),
                        "task_key": str(item.get("task_key", "")),
                        "eval_set": str(item.get("eval_set", "")),
                        "instruction": instruction,
                        "step": int(step_idx),
                        "image_t": image_t_path,
                        "image_next": image_next_path,
                        "logged_action_id": sampled.get("logged_action_id"),
                        "sampled_action_id": int(action_id),
                        "sampled_action_name": str(sampled.get("sampled_action_name", "")),
                        "reward": _safe_float(reward, 0.0),
                        "done": bool(done),
                        "task_success": _safe_float(info.get("task_success", 0.0), 0.0),
                        "last_action_success": _safe_int(info.get("last_action_success", 0), 0),
                        "distance": _safe_float(info.get("distance", -1.0), -1.0),
                        "env_step": _safe_int(info.get("env_step", step_idx + 1), step_idx + 1),
                        "policy": str(sampled.get("policy", args.policy)),
                        "epsilon": _safe_float(sampled.get("epsilon", args.epsilon), args.epsilon),
                        "seed": int(args.seed),
                    }
                    transitions.append(transition)
                    _append_jsonl(transitions_path, transition)
                    episode_reward += float(transition["reward"])
                    prev_action = action_id
                    obs = next_obs
                    step_idx += 1

                episode_row = {
                        "rollout_id": int(rollout_id),
                        "record_idx": int(item.get("record_idx", -1)),
                        "episode_id": int(item.get("episode_id", -1)),
                        "task_key": str(item.get("task_key", "")),
                        "eval_set": str(item.get("eval_set", "")),
                        "instruction": instruction,
                        "policy": str(args.policy),
                        "steps": int(step_idx),
                        "episode_reward": float(episode_reward),
                        "task_success": _safe_float(info.get("task_success", 0.0), 0.0),
                        "collision": _safe_int(info.get("collision", 0), 0),
                        "final_distance": _safe_float(info.get("distance", -1.0), -1.0),
                        "seed": int(args.seed),
                    }
                episodes.append(episode_row)
                _append_jsonl(episodes_path, episode_row)
                print(
                    f"[{rollout_id}/{len(selected)}] eval_set={item.get('eval_set','')} "
                    f"task={item.get('task_key','')} success={_safe_float(info.get('task_success',0.0),0.0):.0f} "
                    f"steps={step_idx}"
                )
        finally:
            try:
                env.env.stop()
            except Exception:
                pass

    _write_jsonl(transitions_path, transitions)
    _write_jsonl(episodes_path, episodes)

    action_distribution = Counter(int(row["sampled_action_id"]) for row in transitions)
    action_success_values = [int(row.get("last_action_success", 0)) for row in transitions]
    collisions = [int(ep.get("collision", 0)) for ep in episodes]
    summary = {
        "dataset": str(args.dataset),
        "embodiedbench_root": str(Path(args.embodiedbench_root).resolve()),
        "num_selected_episodes": len(selected),
        "num_collected_episodes": len(episodes),
        "num_transitions": len(transitions),
        "policy": str(args.policy),
        "split": str(args.split),
        "seed": int(args.seed),
        "epsilon": float(args.epsilon),
        "action_dim": int(args.action_dim),
        "action_distribution": {str(k): int(v) for k, v in sorted(action_distribution.items())},
        "action_success_rate": float(sum(action_success_values) / max(1, len(action_success_values))),
        "task_success_rate": float(sum(float(ep.get("task_success", 0.0)) for ep in episodes) / max(1, len(episodes))),
        "collision_rate": float(sum(1 for x in collisions if x) / max(1, len(collisions))),
        "outputs": {
            "transitions_jsonl": str(transitions_path),
            "episodes_jsonl": str(episodes_path),
            "summary_json": str(summary_path),
            "step_screenshots_dir": str(screenshot_dir),
        },
        "args": vars(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"task_success_rate": summary["task_success_rate"], "num_transitions": len(transitions)}, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
