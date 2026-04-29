"""Prepare EB-Nav assets for Phase 2 planner SFT and reward training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from src.data.eb_nav_dataset import (
    ACTION_NAMES,
    compute_eb_nav_reward,
    get_eb_nav_action_id,
    resolve_eb_nav_image_path,
)
from src.visualize.wandb_tracker import init_tracker
from src.vlm.qwen_planner import (
    PLANNER_ACTION_END_TOKEN,
    PLANNER_ACTION_START_TOKEN,
    PLANNER_ACTION_TOKENS,
    PLANNER_LATENT_TOKEN,
    build_planner_special_response,
)

console = Console()

def _iter_records(
    *,
    dataset_path: Path,
    images_base_dir: Path,
    limit_episodes: int,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with open(dataset_path) as f:
        data = json.load(f)
    if limit_episodes > 0:
        data = data[:limit_episodes]

    sft_records: list[dict[str, Any]] = []
    reward_records: list[dict[str, Any]] = []
    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    task_id = progress.add_task("Preparing EB-Nav Phase 2 records", total=len(data))
    live_context = Live(Group(_build_prepare_table(0, len(data), 0, 0), progress), console=console, refresh_per_second=4) if show_progress else None
    if live_context is not None:
        live_context.__enter__()
    try:
        for ep_idx, episode in enumerate(data):
            trajectory = episode.get("trajectory", [])
            if "input" not in episode or not str(episode["input"]).strip():
                raise ValueError(f"episode_{ep_idx} missing required raw prompt field: input")
            prompt = str(episode["input"])
            instruction = str(episode.get("instruction", ""))
            episode_success = episode.get("success", 0)
            for step_idx, step in enumerate(trajectory):
                plans = step.get("executable_plan") or []
                for plan_idx, plan in enumerate(plans):
                    action_id = get_eb_nav_action_id(plan)
                    img_path = str(plan.get("img_path", step.get("input_image_path", "")))
                    image_path = resolve_eb_nav_image_path(img_path, images_base_dir)
                    if not image_path or not Path(image_path).is_file():
                        raise FileNotFoundError(f"image file not found for episode_{ep_idx}_step_{step_idx}: {image_path}")
                    if "reasoning_and_reflection" not in step:
                        raise ValueError(f"episode_{ep_idx}_step_{step_idx} missing reasoning_and_reflection")
                    cot = str(step["reasoning_and_reflection"])
                    response = build_planner_special_response(
                        cot=cot,
                        action_id=action_id,
                    )
                    record_id = f"episode_{ep_idx}_step_{step_idx}_plan_{plan_idx}"
                    sft_records.append(
                        {
                            "id": record_id,
                            "image": image_path,
                            "prompt": prompt,
                            "response": response,
                            "instruction": instruction,
                            "cot": cot,
                            "action_id": action_id,
                            "action_name": ACTION_NAMES.get(action_id, ""),
                            "episode_idx": ep_idx,
                            "trajectory_step_idx": step_idx,
                            "plan_idx": plan_idx,
                        }
                    )

                    reward = compute_eb_nav_reward(
                        action_success=bool(plan.get("action_success", False)),
                        env_feedback=str(plan.get("env_feedback", "")),
                        is_terminal=step_idx == len(trajectory) - 1 and plan_idx == len(plans) - 1,
                        episode_success=episode_success,
                    )
                    reward_records.append(
                        {
                            "id": record_id,
                            "episode_idx": ep_idx,
                            "episode_id": episode.get("episode_id", str(ep_idx)),
                            "trajectory_step_idx": step_idx,
                            "plan_idx": plan_idx,
                            "action_id": action_id,
                            "action_success": bool(plan.get("action_success", False)),
                            "env_feedback": str(plan.get("env_feedback", "")),
                            "terminal": bool(step_idx == len(trajectory) - 1 and plan_idx == len(plans) - 1),
                            "episode_success": bool(episode_success),
                            "reward": reward,
                        }
                    )
            progress.update(task_id, advance=1)
            if live_context is not None:
                live_context.update(
                    Group(_build_prepare_table(ep_idx + 1, len(data), len(sft_records), len(reward_records)), progress)
                )
    finally:
        if live_context is not None:
            live_context.__exit__(None, None, None)
    return sft_records, reward_records


def _build_prepare_table(
    episodes_done: int,
    episodes_total: int,
    sft_records: int,
    reward_records: int,
) -> Table:
    table = Table(title="EB-Nav Phase 2 数据准备", expand=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("episodes", f"{episodes_done}/{episodes_total}")
    table.add_row("sft_records", str(sft_records))
    table.add_row("reward_records", str(reward_records))
    return table


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _validate_sft_records(records: list[dict[str, Any]]) -> None:
    for record in records:
        action_id = int(record["action_id"])
        if action_id < 0 or action_id > 7:
            raise ValueError(f"{record['id']} invalid action_id={action_id}")
        if not str(record.get("prompt", "")).strip():
            raise ValueError(f"{record['id']} prompt is empty")
        expected_tail = (
            f"{PLANNER_LATENT_TOKEN}{PLANNER_ACTION_START_TOKEN}"
            f"{PLANNER_ACTION_TOKENS[action_id]}{PLANNER_ACTION_END_TOKEN}"
        )
        response = str(record["response"])
        if not response.startswith("<think>") or not response.endswith(expected_tail):
            raise ValueError(f"{record['id']} invalid special-token planner response")
        if "<LATENT_STATE>" in response or "action_prior" in response or "planner_trigger" in response:
            raise ValueError(f"{record['id']} contains deprecated JSON planner fields")


def _build_stats(reward_records: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(item["reward"]) for item in reward_records]
    positives = sum(1 for x in rewards if x > 0)
    negatives = sum(1 for x in rewards if x < 0)
    terminals = sum(1 for item in reward_records if item.get("terminal"))
    return {
        "num_records": len(reward_records),
        "reward_mean": mean(rewards) if rewards else 0.0,
        "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "positive_ratio": positives / max(1, len(rewards)),
        "negative_ratio": negatives / max(1, len(rewards)),
        "terminal_ratio": terminals / max(1, len(rewards)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/EB-Nav/eb-nav_dataset_single_step.json")
    parser.add_argument("--images-base-dir", default="datasets/EB-Nav")
    parser.add_argument("--sft-output", default="datasets/EB-Nav/phase2_qwen_planner_sft.jsonl")
    parser.add_argument("--reward-output", default="datasets/EB-Nav/phase2_reward_cache.jsonl")
    parser.add_argument("--stats-output", default="datasets/EB-Nav/phase2_reward_stats.json")
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true", help="Disable Rich TUI/progress output.")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable W&B logging for this script.")
    args = parser.parse_args()

    tracker = None
    if not args.disable_wandb:
        tracker = init_tracker(
            task_name="prepare_eb_nav_phase2",
            config={
                "dataset": args.dataset,
                "images_base_dir": args.images_base_dir,
                "sft_output": args.sft_output,
                "reward_output": args.reward_output,
                "stats_output": args.stats_output,
                "limit_episodes": int(args.limit_episodes),
            },
        )

    sft_records, reward_records = _iter_records(
        dataset_path=Path(args.dataset),
        images_base_dir=Path(args.images_base_dir),
        limit_episodes=int(args.limit_episodes),
        show_progress=not bool(args.no_progress),
    )
    _validate_sft_records(sft_records)
    _write_jsonl(Path(args.sft_output), sft_records)
    _write_jsonl(Path(args.reward_output), reward_records)
    stats = _build_stats(reward_records)
    Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats_output, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    if tracker is not None:
        tracker.log_metrics(stats, step=0)
        tracker.log_artifact_path("phase2-qwen-planner-sft", args.sft_output, artifact_type="dataset")
        tracker.log_artifact_path("phase2-reward-cache", args.reward_output, artifact_type="dataset")
        tracker.log_artifact_path("phase2-reward-stats", args.stats_output, artifact_type="metrics")
        tracker.finish()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
