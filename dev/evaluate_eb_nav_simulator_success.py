"""Evaluate EB-Nav simulator success using the flower WM reward-head policy.

This script bridges the real EmbodiedBench EB-Navigation simulator with the
existing WM reward-head action policy used in flower. Split membership follows
flower's historical sequence split, but all 300 simulator tasks appear in both
train and heldout record pools because the trajectory dataset contains multiple
model-generated trajectories per task. Therefore split labels here mean
record-level membership, not unseen simulator tasks.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from PIL import Image

from scripts.evaluate_eb_nav_value_head_actions import _action_tensor
from src.data.eb_nav_dataset import ACTION_MAP, ACTION_NAMES, EBNavSequenceDataset
from src.train.train_wm_joint import _normalize_patch_latent, _resolve_joint_resume_checkpoint
from src.train.visualize_eb_nav_rollout import _build_wm, _load_cfg
from src.utils.env import load_project_env
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.qwen_planner import build_planner_special_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/wm/joint_qwen/checkpoint_step_00002000.pt")
    parser.add_argument("--dataset", default="datasets/EB-Nav/eb-nav_dataset_single_step.json")
    parser.add_argument("--images-base-dir", default="datasets/EB-Nav")
    parser.add_argument("--embodiedbench-root", default="/home/jincai_guo/atst/EmbodiedBench")
    parser.add_argument("--planner-lora", default="models/qwen_planner_lora")
    parser.add_argument("--output-dir", default="outputs/dev/eb_nav_simulator_success")
    parser.add_argument("--train-max-samples", type=int, default=22528)
    parser.add_argument("--temporal-stride", type=int, default=4)
    parser.add_argument("--num-train-rollouts", type=int, default=8)
    parser.add_argument("--num-test-rollouts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qwen-dtype", default="bfloat16")
    parser.add_argument("--qwen-device-map", default="auto", choices=["auto", "none"])
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--history-len", type=int, default=0)
    parser.add_argument("--anchor-action-id", type=int, default=0)
    parser.add_argument("--env-max-steps", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=500)
    parser.add_argument("--fov", type=int, default=100)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--keep-image-cache", action="store_true")
    return parser.parse_args()


def _import_eb_navigation_env(embodiedbench_root: str):
    root = str(Path(embodiedbench_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv

    return EBNavigationEnv


def _classify_record_pool(
    *,
    dataset_path: str,
    images_base_dir: str,
    action_dim: int,
    history_len: int,
    temporal_stride: int,
    train_max_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    sequence_dataset = EBNavSequenceDataset(
        json_path=dataset_path,
        images_base_dir=images_base_dir,
        action_dim=action_dim,
        history_len=history_len,
        temporal_stride=temporal_stride,
    )
    records = json.load(open(dataset_path))
    record_counts: dict[int, dict[str, int]] = defaultdict(lambda: {"train": 0, "heldout": 0})
    for seq_idx, seq in enumerate(sequence_dataset.sequences):
        episode_idx = int(seq["episode_idx"])
        split = "train" if seq_idx < train_max_samples else "heldout"
        record_counts[episode_idx][split] += 1

    record_pool: list[dict[str, Any]] = []
    overlap_records = 0
    for episode_idx, counts in sorted(record_counts.items()):
        if counts["train"] > 0 and counts["heldout"] > 0:
            overlap_records += 1
            continue
        split = "train_seen" if counts["train"] > 0 else "heldout_tail"
        record = records[episode_idx]
        record_pool.append(
            {
                "split": split,
                "episode_idx": episode_idx,
                "episode_id": int(record["episode_id"]),
                "eval_set": str(record["eval_set"]),
                "model_name": str(record["model_name"]),
                "instruction": str(record["instruction"]),
                "task_key": f"{record['eval_set']}:{int(record['episode_id']):02d}",
                "sequence_count": counts["train"] + counts["heldout"],
            }
        )
    info = {
        "total_sequences": len(sequence_dataset.sequences),
        "train_sequence_count": min(train_max_samples, len(sequence_dataset.sequences)),
        "heldout_tail_sequence_count": max(0, len(sequence_dataset.sequences) - train_max_samples),
        "record_pool_size": len(record_pool),
        "overlap_record_count_skipped": overlap_records,
    }
    return record_pool, info, records


def _sample_records(
    *,
    record_pool: list[dict[str, Any]],
    num_train: int,
    num_test: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for split, target in (("train_seen", num_train), ("heldout_tail", num_test)):
        candidates = [item for item in record_pool if item["split"] == split]
        rng.shuffle(candidates)
        seen_task_keys: set[str] = set()
        split_selected = []
        for item in candidates:
            task_key = str(item["task_key"])
            if task_key in seen_task_keys:
                continue
            seen_task_keys.add(task_key)
            split_selected.append(item)
            if len(split_selected) >= target:
                break
        selected.extend(split_selected)
    return selected


def _encode_observation_latent(
    *,
    qwen_adapter: QwenVLMAdapter,
    image_array: np.ndarray,
    prompt: str,
    anchor_response: str,
    device: torch.device,
    num_patches: int,
    token_dim: int,
    tmp_dir: Path,
    obs_key: str,
) -> torch.Tensor:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image_path = tmp_dir / f"{obs_key}.png"
    Image.fromarray(image_array).save(image_path)
    extracted = qwen_adapter.get_planner_latent_and_action_prior_batch(
        image_paths=[str(image_path)],
        prompts=[prompt],
        responses=[anchor_response],
        llm_backbone_trainable=False,
    )
    latent = extracted["latent"][0].to(device)
    return _normalize_patch_latent(latent, num_patches, token_dim)


def _build_history_tensors(
    *,
    latents: list[torch.Tensor],
    executed_actions: list[int],
    history_len: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_index = len(latents) - 1
    selected_indices = list(range(max(0, current_index - history_len + 1), current_index + 1))
    pad_count = history_len - len(selected_indices)

    z_items: list[torch.Tensor] = []
    a_items: list[torch.Tensor] = []
    zero_action = torch.zeros(action_dim, device=device, dtype=dtype)

    for _ in range(pad_count):
        z_items.append(latents[0])
        a_items.append(zero_action)

    for obs_idx in selected_indices:
        z_items.append(latents[obs_idx])
        if obs_idx < len(executed_actions):
            a_items.append(_action_tensor(executed_actions[obs_idx], action_dim=action_dim, device=device, dtype=dtype))
        else:
            a_items.append(zero_action)

    return torch.stack(z_items, dim=0), torch.stack(a_items, dim=0)


def _score_candidate_actions(
    *,
    wm_model: torch.nn.Module,
    z_history: torch.Tensor,
    action_history_base: torch.Tensor,
    action_dim: int,
    eval_batch_size: int,
) -> dict[int, float]:
    action_ids = sorted(ACTION_MAP.keys())
    batch_z = []
    batch_a = []
    for action_id in action_ids:
        action_history = action_history_base.clone()
        action_history[-1] = _action_tensor(action_id, action_dim=action_dim, device=z_history.device, dtype=z_history.dtype)
        batch_z.append(z_history)
        batch_a.append(action_history)

    scores: dict[int, float] = {}
    with torch.no_grad():
        for start in range(0, len(action_ids), eval_batch_size):
            end = min(len(action_ids), start + eval_batch_size)
            pred_z = wm_model.predict_next(torch.stack(batch_z[start:end], dim=0), torch.stack(batch_a[start:end], dim=0))
            pred_scores = wm_model.predict_reward(pred_z).detach().float().cpu().tolist()
            for action_id, score in zip(action_ids[start:end], pred_scores):
                scores[int(action_id)] = float(score)
    return scores


def _aggregate_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if str(row["split"]) == split]
        n = len(split_rows)
        result[split] = {
            "num_rollouts": n,
            "success_rate": sum(float(row["task_success"]) for row in split_rows) / n if n else 0.0,
            "mean_steps": sum(int(row["num_steps"]) for row in split_rows) / n if n else 0.0,
            "mean_first_step_match": sum(int(row["first_step_match"]) for row in split_rows) / n if n else 0.0,
            "mean_action_accuracy": sum(float(row["action_accuracy"]) for row in split_rows) / n if n else 0.0,
        }
    return result


def _init_wandb(args: argparse.Namespace, selected: list[dict[str, Any]], split_info: dict[str, Any]) -> wandb.sdk.wandb_run.Run | None:
    if args.disable_wandb:
        return None
    project = args.wandb_project or os.getenv("WANDB_PROJECT", "flower")
    entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
    run_name = args.wandb_run_name or f"eb-nav-sim-{args.num_train_rollouts}x{args.num_test_rollouts}"
    config = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "embodiedbench_root": args.embodiedbench_root,
        "train_max_samples": int(args.train_max_samples),
        "temporal_stride": int(args.temporal_stride),
        "num_train_rollouts": int(args.num_train_rollouts),
        "num_test_rollouts": int(args.num_test_rollouts),
        "seed": int(args.seed),
        "env_max_steps": int(args.env_max_steps),
        "selected_rollout_count": len(selected),
        **split_info,
    }
    return wandb.init(project=project, entity=entity, name=run_name, config=config)


def _log_wandb_tables(
    *,
    run: wandb.sdk.wandb_run.Run | None,
    rollout_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
) -> None:
    if run is None:
        return
    rollout_columns = [
        "rollout_index",
        "split",
        "eval_set",
        "task_key",
        "episode_record_idx",
        "episode_id",
        "model_name",
        "instruction",
        "task_success",
        "num_steps",
        "first_step_match",
        "action_accuracy",
    ]
    rollout_table = wandb.Table(columns=rollout_columns)
    for row in rollout_rows:
        rollout_table.add_data(*[row.get(col, "") for col in rollout_columns])

    step_columns = [
        "rollout_index",
        "split",
        "eval_set",
        "task_key",
        "episode_record_idx",
        "model_name",
        "step_index",
        "screenshot",
        "pred_action_id",
        "pred_action_name",
        "logged_action_id",
        "logged_action_name",
        "reward",
        "task_success",
        "last_action_success",
        "distance",
    ]
    step_table = wandb.Table(columns=step_columns)
    for row in step_rows:
        image_obj = wandb.Image(row["screenshot_path"], caption=f"{row['task_key']} step {row['step_index']}")
        step_table.add_data(
            row["rollout_index"],
            row["split"],
            row["eval_set"],
            row["task_key"],
            row["episode_record_idx"],
            row["model_name"],
            row["step_index"],
            image_obj,
            row["pred_action_id"],
            row["pred_action_name"],
            row["logged_action_id"],
            row["logged_action_name"],
            row["reward"],
            row["task_success"],
            row["last_action_success"],
            row["distance"],
        )
    run.log({"rollout_summary": rollout_table, "step_details": step_table})


def main() -> None:
    load_project_env()
    args = parse_args()
    cfg = _load_cfg()
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    checkpoint_path = _resolve_joint_resume_checkpoint(args.checkpoint, Path("models/wm/joint_qwen"))
    if checkpoint_path is None:
        raise FileNotFoundError(args.checkpoint)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    wm_state = checkpoint["wm_state"]
    if isinstance(checkpoint_config, dict):
        reward_cfg = dict(checkpoint_config.get("reward", {}))
        reward_cfg["enabled"] = True
        checkpoint_config = dict(checkpoint_config)
        checkpoint_config["reward"] = reward_cfg
    if not any(key.startswith("reward_head.") for key in wm_state):
        raise RuntimeError("checkpoint wm_state has no reward_head.* weights")

    action_dim = int(checkpoint_config.get("action_dim", getattr(cfg.wm, "action_dim", 3)))
    history_len = int(args.history_len or cfg.wm.history_len)
    record_pool, split_info, records = _classify_record_pool(
        dataset_path=args.dataset,
        images_base_dir=args.images_base_dir,
        action_dim=action_dim,
        history_len=history_len,
        temporal_stride=int(args.temporal_stride),
        train_max_samples=int(args.train_max_samples),
    )
    selected = _sample_records(
        record_pool=record_pool,
        num_train=int(args.num_train_rollouts),
        num_test=int(args.num_test_rollouts),
        seed=int(args.seed),
    )
    if not selected:
        raise RuntimeError("no records selected")

    qwen_adapter = QwenVLMAdapter(
        model_name=str(cfg.wm.encoder.model_name),
        latent_dim=int(cfg.wm.latent_dim),
        enabled=True,
        fallback_enabled=False,
        model_dtype=args.qwen_dtype,
        device_map=None if args.qwen_device_map == "none" else "auto",
    )
    qwen_adapter.planner_low_memory = True
    qwen_adapter.planner_inference_mode = True
    qwen_adapter._ensure_model()
    if qwen_adapter._model is None:
        raise RuntimeError(f"Failed to load Qwen model: {qwen_adapter.init_error}")
    qwen_adapter.load_lora_adapter(args.planner_lora, trainable=False)
    qwen_adapter._model.load_state_dict(checkpoint["vision_encoder_state"], strict=False)
    if args.qwen_device_map == "none":
        qwen_adapter._model.to(device)
    qwen_adapter._model.eval()
    for param in qwen_adapter._model.parameters():
        param.requires_grad = False

    wm_model = _build_wm(cfg, checkpoint_config, device)
    wm_model.load_state_dict(wm_state, strict=False)
    if wm_model.reward_head is None:
        raise RuntimeError("WM reward head missing after build")
    wm_model.eval()
    del checkpoint
    gc.collect()

    EBNavigationEnv = _import_eb_navigation_env(args.embodiedbench_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_image_dir = output_dir / "tmp_obs"
    screenshot_dir = output_dir / "step_screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    anchor_response = build_planner_special_response(cot="", action_id=int(args.anchor_action_id))
    wandb_run = _init_wandb(args, selected, split_info)

    rollout_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        grouped[(str(item["split"]), str(item["eval_set"]))].append(item)

    rollout_index = 0
    for (split, eval_set), items in sorted(grouped.items()):
        selected_indexes = [int(item["episode_id"]) - 1 for item in items]
        env = EBNavigationEnv(
            eval_set=eval_set,
            exp_name=f"sim_eval_{split}_{eval_set}",
            selected_indexes=selected_indexes,
            resolution=int(args.resolution),
            fov=int(args.fov),
        )
        env._max_episode_steps = int(args.env_max_steps)
        try:
            for item in items:
                rollout_index += 1
                record = records[int(item["episode_idx"])]
                logged_actions = [
                    int(step["executable_plan"][0]["action"][0])
                    for step in record["trajectory"]
                    if step.get("executable_plan")
                ]
                obs = env.reset()
                latents: list[torch.Tensor] = []
                executed_actions: list[int] = []
                prompt = str(env.episode_language_instruction)
                done = False
                first_step_match = 0
                correct_steps = 0
                total_steps = 0
                info: dict[str, Any] = {}
                task_key = str(item["task_key"])
                while not done and total_steps < int(args.env_max_steps):
                    screenshot_path = screenshot_dir / f"rollout_{rollout_index:03d}_step_{total_steps:02d}.png"
                    Image.fromarray(np.asarray(obs["head_rgb"])).save(screenshot_path)
                    latent = _encode_observation_latent(
                        qwen_adapter=qwen_adapter,
                        image_array=np.asarray(obs["head_rgb"]),
                        prompt=prompt,
                        anchor_response=anchor_response,
                        device=device,
                        num_patches=int(cfg.wm.num_patches),
                        token_dim=int(cfg.wm.token_dim),
                        tmp_dir=tmp_image_dir,
                        obs_key=f"r{rollout_index}_s{total_steps}",
                    )
                    latents.append(latent)
                    z_history, action_history = _build_history_tensors(
                        latents=latents,
                        executed_actions=executed_actions,
                        history_len=history_len,
                        action_dim=action_dim,
                        device=device,
                        dtype=latent.dtype,
                    )
                    scores = _score_candidate_actions(
                        wm_model=wm_model,
                        z_history=z_history,
                        action_history_base=action_history,
                        action_dim=action_dim,
                        eval_batch_size=int(args.eval_batch_size),
                    )
                    pred_action = max(sorted(ACTION_MAP.keys()), key=lambda action_id: scores.get(action_id, float("-inf")))
                    logged_action = int(logged_actions[total_steps]) if total_steps < len(logged_actions) else -1
                    if total_steps == 0:
                        first_step_match = int(pred_action == logged_action)
                    correct_steps += int(pred_action == logged_action)
                    obs, reward, done, info = env.step(
                        int(pred_action),
                        {"policy": "wm_reward_head", "task_key": task_key},
                        1,
                    )
                    step_row = {
                        "rollout_index": rollout_index,
                        "split": split,
                        "eval_set": eval_set,
                        "task_key": task_key,
                        "episode_record_idx": int(item["episode_idx"]),
                        "model_name": str(item["model_name"]),
                        "step_index": total_steps,
                        "pred_action_id": int(pred_action),
                        "pred_action_name": ACTION_NAMES.get(int(pred_action), ""),
                        "logged_action_id": int(logged_action),
                        "logged_action_name": ACTION_NAMES.get(int(logged_action), "") if logged_action >= 0 else "",
                        "reward": float(reward),
                        "task_success": float(info["task_success"]),
                        "last_action_success": int(info["last_action_success"]),
                        "distance": float(info["distance"]),
                        "screenshot_path": str(screenshot_path),
                    }
                    step_row.update({f"score_a{aid}": scores.get(aid, "") for aid in sorted(ACTION_MAP.keys())})
                    step_rows.append(step_row)
                    executed_actions.append(int(pred_action))
                    total_steps += 1

                rollout_rows.append(
                    {
                        "rollout_index": rollout_index,
                        "split": split,
                        "eval_set": eval_set,
                        "task_key": task_key,
                        "episode_record_idx": int(item["episode_idx"]),
                        "episode_id": int(item["episode_id"]),
                        "model_name": str(item["model_name"]),
                        "instruction": str(item["instruction"]),
                        "task_success": float(info.get("task_success", 0.0)),
                        "num_steps": int(info.get("env_step", total_steps)),
                        "first_step_match": int(first_step_match),
                        "action_accuracy": float(correct_steps / max(1, total_steps)),
                    }
                )
                print(
                    f"[{rollout_index}/{len(selected)}] split={split} eval_set={eval_set} task={task_key} "
                    f"success={float(info.get('task_success', 0.0)):.0f} steps={int(info.get('env_step', total_steps))}"
                )
                if not args.keep_image_cache:
                    shutil.rmtree(tmp_image_dir, ignore_errors=True)
        finally:
            try:
                env.env.stop()
            except Exception:
                pass

    rollout_csv = output_dir / "rollout_summary.csv"
    step_csv = output_dir / "step_predictions.csv"
    with open(rollout_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rollout_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rollout_rows)
    with open(step_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
        writer.writeheader()
        writer.writerows(step_rows)

    summary = {
        "checkpoint": str(checkpoint_path),
        "embodiedbench_root": str(Path(args.embodiedbench_root).resolve()),
        "dataset": args.dataset,
        "split_info": split_info,
        "selected_rollouts": selected,
        "note": (
            "train_seen/heldout_tail are record-level labels from flower's historical sequence split. "
            "All 300 simulator tasks appear in both pools because the trajectory dataset contains multiple model trajectories per task."
        ),
        "metrics_by_split": _aggregate_rollouts(rollout_rows),
        "pred_action_distribution": {
            str(action_id): count for action_id, count in Counter(int(row["pred_action_id"]) for row in step_rows).items()
        },
        "outputs": {
            "rollout_summary_csv": str(rollout_csv),
            "step_predictions_csv": str(step_csv),
            "step_screenshots_dir": str(screenshot_dir),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if wandb_run is not None:
        wandb_run.summary.update(summary["metrics_by_split"])
        wandb_run.summary["pred_action_distribution"] = summary["pred_action_distribution"]
        wandb_run.summary["output_dir"] = str(output_dir)
        _log_wandb_tables(run=wandb_run, rollout_rows=rollout_rows, step_rows=step_rows)
        wandb_run.finish()
    print(json.dumps(summary["metrics_by_split"], indent=2, ensure_ascii=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
