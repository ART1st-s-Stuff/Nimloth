"""Evaluate EB-Nav rollout success with the available offline trajectory replay backend.

This repo currently has EB-Nav logged trajectories and ai2thor installed, but no
EmbodiedBench EB-Nav interactive simulator package/config.  The default backend
therefore performs strict offline replay: at each logged observation, the agent
chooses an action from the WM reward/value head; once the choice diverges from
the logged action, counterfactual environment state is unavailable, so the
rollout is counted as replay-failed.  A replay success means the agent exactly
recovers the logged action suffix and the logged episode itself is successful.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from scripts.evaluate_eb_nav_value_head_actions import (
    _action_tensor,
    _encode_episode_latents,
    _materialize_image,
    _step_action_id,
    _step_image_path,
    _step_response,
)
from src.data.eb_nav_dataset import ACTION_MAP, ACTION_NAMES, EBNavSequenceDataset
from src.train.train_wm_joint import _resolve_joint_resume_checkpoint
from src.train.visualize_eb_nav_rollout import _build_wm, _load_cfg
from src.utils.env import load_project_env
from src.vlm.qwen_adapter import QwenVLMAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/wm/joint_qwen/checkpoint_step_00002000.pt")
    parser.add_argument("--dataset", default="datasets/EB-Nav/eb-nav_dataset_single_step.json")
    parser.add_argument("--images-base-dir", default="datasets/EB-Nav")
    parser.add_argument("--images-zip", default="datasets/EB-Nav/images.zip")
    parser.add_argument("--image-cache-dir", default="datasets/EB-Nav/.image_cache/rollout_success_eval")
    parser.add_argument("--planner-lora", default="models/qwen_planner_lora")
    parser.add_argument("--output-dir", default="outputs/dev/eb_nav_rollout_success")
    parser.add_argument("--backend", default="offline_replay", choices=["offline_replay"])
    parser.add_argument("--train-max-samples", type=int, default=22528)
    parser.add_argument("--temporal-stride", type=int, default=4)
    parser.add_argument("--num-train-rollouts", type=int, default=16)
    parser.add_argument("--num-test-rollouts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--qwen-dtype", default="bfloat16")
    parser.add_argument("--qwen-device-map", default="auto", choices=["auto", "none"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-micro-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--history-len", type=int, default=0, help="0 uses cfg.wm.history_len")
    parser.add_argument("--max-steps-per-rollout", type=int, default=0, help="0 evaluates the full available suffix")
    parser.add_argument("--keep-image-cache", action="store_true")
    return parser.parse_args()


def _sample_sequence_rollouts(
    *,
    dataset_path: str,
    images_base_dir: str,
    action_dim: int,
    history_len: int,
    temporal_stride: int,
    train_max_samples: int,
    num_train: int,
    num_test: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sequence_dataset = EBNavSequenceDataset(
        json_path=dataset_path,
        images_base_dir=images_base_dir,
        action_dim=action_dim,
        history_len=history_len,
        temporal_stride=temporal_stride,
    )
    total_sequences = len(sequence_dataset.sequences)
    if train_max_samples <= 0 or train_max_samples >= total_sequences:
        raise ValueError(
            f"--train-max-samples must split the sequence list; got {train_max_samples=} {total_sequences=}"
        )

    rng = random.Random(seed)
    train_seq_indices = list(range(train_max_samples))
    test_seq_indices = list(range(train_max_samples, total_sequences))
    rng.shuffle(train_seq_indices)
    rng.shuffle(test_seq_indices)

    def collect(split: str, indices: list[int], target_count: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen_episodes: set[int] = set()
        for seq_idx in indices:
            seq = sequence_dataset.sequences[seq_idx]
            episode_idx = int(seq["episode_idx"])
            if episode_idx in seen_episodes:
                continue
            seen_episodes.add(episode_idx)
            selected.append(
                {
                    "split": split,
                    "sequence_index": seq_idx,
                    "episode_idx": episode_idx,
                    "episode_id": seq.get("episode_id", str(episode_idx)),
                }
            )
            if len(selected) >= target_count:
                break
        return selected

    selected = collect("train_seen", train_seq_indices, num_train) + collect("heldout_tail", test_seq_indices, num_test)
    split_info = {
        "total_sequences": total_sequences,
        "train_sequence_count": train_max_samples,
        "heldout_tail_sequence_count": total_sequences - train_max_samples,
        "history_len": history_len,
        "temporal_stride": temporal_stride,
        "seed": seed,
    }
    return selected, split_info


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


def _evaluate_episode_offline_replay(
    *,
    ep_idx: int,
    split: str,
    seed_sequence_index: int,
    episode: dict[str, Any],
    qwen_adapter: QwenVLMAdapter,
    wm_model: torch.nn.Module,
    cfg: Any,
    device: torch.device,
    action_dim: int,
    history_len: int,
    max_steps_per_rollout: int,
    images_base_dir: Path,
    images_zip: zipfile.ZipFile | None,
    cache_dir: Path,
    encode_micro_batch_size: int,
    eval_batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traj = episode.get("trajectory", [])
    if len(traj) < history_len:
        return {
            "split": split,
            "episode_idx": ep_idx,
            "episode_id": episode.get("episode_id", ""),
            "eval_set": episode.get("eval_set", ""),
            "seed_sequence_index": seed_sequence_index,
            "logged_success": float(episode.get("success", 0)),
            "evaluated_steps": 0,
            "matched_steps": 0,
            "first_mismatch_step": "too_short",
            "replay_success": 0,
            "action_accuracy": 0.0,
            "prefix_match_ratio": 0.0,
        }, []

    image_paths = [_materialize_image(_step_image_path(step), images_base_dir, images_zip, cache_dir) for step in traj]
    responses = [_step_response(step) for step in traj]
    z = _encode_episode_latents(
        adapter=qwen_adapter,
        image_paths=image_paths,
        prompt=str(episode.get("input") or episode.get("instruction") or ""),
        responses=responses,
        device=device,
        num_patches=int(cfg.wm.num_patches),
        token_dim=int(cfg.wm.token_dim),
        micro_batch_size=encode_micro_batch_size,
    )
    logged_actions = [_step_action_id(step) for step in traj]
    action_vecs = torch.stack(
        [_action_tensor(action_id, action_dim=action_dim, device=device, dtype=z.dtype) for action_id in logged_actions],
        dim=0,
    )

    start_step = history_len - 1
    end_step = len(traj)
    if max_steps_per_rollout > 0:
        end_step = min(end_step, start_step + max_steps_per_rollout)

    step_rows: list[dict[str, Any]] = []
    matched_steps = 0
    first_mismatch_step: int | None = None
    diverged = False
    action_ids = sorted(ACTION_MAP.keys())
    for step_idx in range(start_step, end_step):
        z_history = z[step_idx - history_len + 1 : step_idx + 1]
        action_history_base = action_vecs[step_idx - history_len + 1 : step_idx + 1].clone()
        scores = _score_candidate_actions(
            wm_model=wm_model,
            z_history=z_history,
            action_history_base=action_history_base,
            action_dim=action_dim,
            eval_batch_size=eval_batch_size,
        )
        pred_action = max(action_ids, key=lambda action_id: scores.get(action_id, float("-inf")))
        logged_action = logged_actions[step_idx]
        correct = int(pred_action == logged_action)
        if correct and not diverged:
            matched_steps += 1
        elif not diverged:
            diverged = True
            first_mismatch_step = step_idx
        row = {
            "split": split,
            "episode_idx": ep_idx,
            "episode_id": episode.get("episode_id", ""),
            "eval_set": episode.get("eval_set", ""),
            "seed_sequence_index": seed_sequence_index,
            "step_index": step_idx,
            "logged_action_id": logged_action,
            "logged_action_name": ACTION_NAMES.get(logged_action, ""),
            "pred_action_id": pred_action,
            "pred_action_name": ACTION_NAMES.get(pred_action, ""),
            "correct": correct,
            "diverged_before_step": int(diverged and first_mismatch_step != step_idx),
        }
        row.update({f"score_a{action_id}": scores.get(action_id, "") for action_id in action_ids})
        step_rows.append(row)

    evaluated_steps = len(step_rows)
    all_match = evaluated_steps > 0 and matched_steps == evaluated_steps
    logged_success = bool(float(episode.get("success", 0)))
    summary = {
        "split": split,
        "episode_idx": ep_idx,
        "episode_id": episode.get("episode_id", ""),
        "eval_set": episode.get("eval_set", ""),
        "seed_sequence_index": seed_sequence_index,
        "logged_success": float(episode.get("success", 0)),
        "evaluated_steps": evaluated_steps,
        "matched_steps": matched_steps,
        "first_mismatch_step": "" if first_mismatch_step is None else first_mismatch_step,
        "replay_success": int(all_match and logged_success),
        "action_accuracy": sum(int(row["correct"]) for row in step_rows) / evaluated_steps if evaluated_steps else 0.0,
        "prefix_match_ratio": matched_steps / evaluated_steps if evaluated_steps else 0.0,
    }
    del z
    torch.cuda.empty_cache()
    return summary, step_rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(row)

    split_metrics = {}
    for split, split_rows in by_split.items():
        n = len(split_rows)
        split_metrics[split] = {
            "num_rollouts": n,
            "logged_success_rate": sum(float(row["logged_success"]) for row in split_rows) / n if n else 0.0,
            "offline_replay_success_rate": sum(int(row["replay_success"]) for row in split_rows) / n if n else 0.0,
            "mean_action_accuracy": sum(float(row["action_accuracy"]) for row in split_rows) / n if n else 0.0,
            "mean_prefix_match_ratio": sum(float(row["prefix_match_ratio"]) for row in split_rows) / n if n else 0.0,
            "evaluated_steps": sum(int(row["evaluated_steps"]) for row in split_rows),
        }
    return split_metrics


def main() -> None:
    load_project_env()
    args = parse_args()
    cfg = _load_cfg()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    checkpoint_path = _resolve_joint_resume_checkpoint(args.checkpoint, Path("models/wm/joint_qwen"))
    if checkpoint_path is None:
        raise FileNotFoundError(args.checkpoint)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    wm_state = checkpoint["wm_state"]
    if not any(key.startswith("reward_head.") for key in wm_state):
        raise RuntimeError("checkpoint wm_state has no reward_head.* weights; cannot run the reward-head agent policy")
    if isinstance(checkpoint_config, dict):
        reward_cfg = dict(checkpoint_config.get("reward", {}))
        reward_cfg["enabled"] = True
        checkpoint_config = dict(checkpoint_config)
        checkpoint_config["reward"] = reward_cfg
    action_dim = int(checkpoint_config.get("action_dim", getattr(cfg.wm, "action_dim", 3)))
    history_len = int(args.history_len or cfg.wm.history_len)

    selected_rollouts, split_info = _sample_sequence_rollouts(
        dataset_path=args.dataset,
        images_base_dir=args.images_base_dir,
        action_dim=action_dim,
        history_len=history_len,
        temporal_stride=int(args.temporal_stride),
        train_max_samples=int(args.train_max_samples),
        num_train=int(args.num_train_rollouts),
        num_test=int(args.num_test_rollouts),
        seed=int(args.seed),
    )

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
        raise RuntimeError("WM reward/value head is disabled after build")
    wm_model.eval()
    del checkpoint
    gc.collect()

    data = json.load(open(args.dataset))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.image_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    images_base_dir = Path(args.images_base_dir)
    images_zip = zipfile.ZipFile(args.images_zip) if Path(args.images_zip).is_file() else None

    rollout_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(selected_rollouts, start=1):
        episode_idx = int(sample["episode_idx"])
        print(
            f"[{index}/{len(selected_rollouts)}] {sample['split']} "
            f"episode_idx={episode_idx} sequence_index={sample['sequence_index']}"
        )
        summary, episode_steps = _evaluate_episode_offline_replay(
            ep_idx=episode_idx,
            split=str(sample["split"]),
            seed_sequence_index=int(sample["sequence_index"]),
            episode=data[episode_idx],
            qwen_adapter=qwen_adapter,
            wm_model=wm_model,
            cfg=cfg,
            device=device,
            action_dim=action_dim,
            history_len=history_len,
            max_steps_per_rollout=int(args.max_steps_per_rollout),
            images_base_dir=images_base_dir,
            images_zip=images_zip,
            cache_dir=cache_dir,
            encode_micro_batch_size=int(args.encode_micro_batch_size),
            eval_batch_size=int(args.eval_batch_size),
        )
        rollout_rows.append(summary)
        step_rows.extend(episode_steps)
        if not args.keep_image_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

    if images_zip is not None:
        images_zip.close()

    rollout_csv = output_dir / "rollout_summary.csv"
    step_csv = output_dir / "step_predictions.csv"
    if rollout_rows:
        with open(rollout_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rollout_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rollout_rows)
    if step_rows:
        with open(step_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
            writer.writeheader()
            writer.writerows(step_rows)

    summary = {
        "backend": args.backend,
        "backend_limitations": (
            "offline_replay is strict logged-trajectory replay; it cannot measure counterfactual "
            "success after the first action mismatch because no interactive EB-Nav simulator is configured."
        ),
        "checkpoint": str(checkpoint_path),
        "dataset": args.dataset,
        "images_base_dir": args.images_base_dir,
        "planner_lora": args.planner_lora,
        "action_dim": action_dim,
        "history_len": history_len,
        "max_steps_per_rollout": int(args.max_steps_per_rollout),
        "selected_rollouts": selected_rollouts,
        "split_info": split_info,
        "metrics_by_split": _aggregate(rollout_rows),
        "pred_action_distribution": dict(Counter(int(row["pred_action_id"]) for row in step_rows)),
        "logged_action_distribution": dict(Counter(int(row["logged_action_id"]) for row in step_rows)),
        "outputs": {
            "rollout_summary_csv": str(rollout_csv),
            "step_predictions_csv": str(step_csv),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["metrics_by_split"], indent=2, ensure_ascii=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
