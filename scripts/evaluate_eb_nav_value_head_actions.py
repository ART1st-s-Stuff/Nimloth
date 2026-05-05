"""Evaluate EB-Nav next-action prediction by scoring candidate actions with WM reward/value head."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import hydra
import torch

from src.data.eb_nav_dataset import ACTION_MAP, ACTION_NAMES, resolve_eb_nav_image_path


def _action_tensor(action_id: int, *, action_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if action_dim >= len(ACTION_MAP):
        vec = torch.zeros(action_dim, device=device, dtype=dtype)
        vec[int(action_id)] = 1.0
        return vec
    return torch.tensor(ACTION_MAP[int(action_id)][:action_dim], device=device, dtype=dtype)
from src.train.train_wm_joint import _normalize_patch_latent, _resolve_joint_resume_checkpoint
from src.train.visualize_eb_nav_rollout import _build_wm, _load_cfg, _action_vec_tensor
from src.utils.env import load_project_env
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.qwen_planner import build_planner_special_response
from src.wm.encoder.qwen import QwenLLMLatentEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/wm/joint_qwen/checkpoint_step_00002700.pt")
    parser.add_argument("--dataset", default="datasets/EB-Nav/eb-nav_dataset_single_step.json")
    parser.add_argument("--images-base-dir", default="datasets/EB-Nav")
    parser.add_argument("--images-zip", default="datasets/EB-Nav/images.zip")
    parser.add_argument("--image-cache-dir", default="datasets/EB-Nav/.image_cache/value_head_eval")
    parser.add_argument("--planner-lora", default="models/qwen_planner_lora")
    parser.add_argument("--output-dir", default="outputs/dev/eval/value_head_action")
    parser.add_argument("--qwen-dtype", default="bfloat16")
    parser.add_argument("--qwen-device-map", default="auto", choices=["auto", "none"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-micro-batch-size", type=int, default=1)
    parser.add_argument("--max-rollouts", type=int, default=0)
    parser.add_argument("--start-rollout", type=int, default=0)
    parser.add_argument("--history-len", type=int, default=0, help="0 uses cfg.wm.history_len")
    parser.add_argument("--keep-image-cache", action="store_true")
    return parser.parse_args()


def _materialize_image(rel_or_abs: str, images_base_dir: Path, images_zip: zipfile.ZipFile | None, cache_dir: Path) -> str:
    resolved = Path(resolve_eb_nav_image_path(rel_or_abs, images_base_dir))
    if resolved.is_file():
        return str(resolved)
    rel = str(rel_or_abs)
    if Path(rel).is_absolute():
        raise FileNotFoundError(rel)
    cache_path = cache_dir / rel
    if cache_path.is_file():
        return str(cache_path)
    if images_zip is None:
        raise FileNotFoundError(f"missing image and no images.zip available: {rel}")
    try:
        info = images_zip.getinfo(rel)
    except KeyError as exc:
        raise FileNotFoundError(f"{rel} not found in {images_zip.filename}") from exc
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with images_zip.open(info) as src, open(cache_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return str(cache_path)


def _step_action_id(step: dict[str, Any]) -> int:
    plan = (step.get("executable_plan") or [{}])[0]
    action = plan.get("action", [0, ""])
    if isinstance(action, list) and action:
        return int(action[0])
    return 0


def _step_image_path(step: dict[str, Any]) -> str:
    plan = (step.get("executable_plan") or [{}])[0]
    return str(step.get("input_image_path") or plan.get("img_path") or "")


def _step_response(step: dict[str, Any]) -> str:
    return build_planner_special_response(
        cot=str(step.get("reasoning_and_reflection", "")),
        action_id=_step_action_id(step),
    )


def _encode_episode_latents(
    *,
    adapter: QwenVLMAdapter,
    image_paths: list[str],
    prompt: str,
    responses: list[str],
    device: torch.device,
    num_patches: int,
    token_dim: int,
    micro_batch_size: int,
) -> torch.Tensor:
    latents: list[torch.Tensor] = []
    chunk = max(1, int(micro_batch_size))
    for start in range(0, len(image_paths), chunk):
        end = min(len(image_paths), start + chunk)
        extracted = adapter.get_planner_latent_and_action_prior_batch(
            image_paths=image_paths[start:end],
            prompts=[prompt] * (end - start),
            responses=responses[start:end],
            llm_backbone_trainable=False,
        )
        latent_batch = extracted["latent"]
        for latent in latent_batch:
            latents.append(_normalize_patch_latent(latent.to(device), num_patches, token_dim))
    return torch.stack(latents, dim=0)


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    load_project_env()
    args = parse_args()
    cfg = _load_cfg()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    checkpoint_path = _resolve_joint_resume_checkpoint(args.checkpoint, Path("models/wm/joint_qwen"))
    if checkpoint_path is None:
        raise FileNotFoundError(args.checkpoint)

    print(f"loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    wm_state = checkpoint["wm_state"]
    has_reward_head = any(k.startswith("reward_head.") for k in wm_state)
    if not has_reward_head:
        raise RuntimeError("checkpoint wm_state has no reward_head.* weights; cannot evaluate value-head action policy")
    if isinstance(checkpoint_config, dict):
        reward_cfg = dict(checkpoint_config.get("reward", {}))
        reward_cfg["enabled"] = True
        checkpoint_config = dict(checkpoint_config)
        checkpoint_config["reward"] = reward_cfg

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
    for p in qwen_adapter._model.parameters():
        p.requires_grad = False

    wm_model = _build_wm(cfg, checkpoint_config, device)
    missing, unexpected = wm_model.load_state_dict(wm_state, strict=False)
    print(f"loaded WM: missing={len(missing)} unexpected={len(unexpected)}")
    if wm_model.reward_head is None:
        raise RuntimeError("WM reward/value head is disabled after build")
    wm_model.eval()
    del checkpoint
    gc.collect()

    action_dim = int(checkpoint_config.get("action_dim", getattr(cfg.wm, "action_dim", 3)))

    data = json.load(open(args.dataset))
    start_rollout = max(0, int(args.start_rollout))
    end_rollout = len(data) if int(args.max_rollouts) <= 0 else min(len(data), start_rollout + int(args.max_rollouts))
    history_len = int(args.history_len or cfg.wm.history_len)
    images_base_dir = Path(args.images_base_dir)
    cache_dir = Path(args.image_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    images_zip = zipfile.ZipFile(args.images_zip) if Path(args.images_zip).is_file() else None
    if images_zip is None:
        print(f"[warn] images.zip not found at {args.images_zip}; using existing image files only")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "value_head_action_predictions.csv"

    y_true: list[int] = []
    y_pred: list[int] = []
    per_eval_set: dict[str, list[tuple[int, int]]] = defaultdict(list)
    action_ids = sorted(ACTION_MAP.keys())

    with open(pred_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rollout_index", "episode_id", "eval_set", "step_index", "gt_action_id", "pred_action_id", "correct"]
            + [f"score_a{aid}" for aid in action_ids],
        )
        writer.writeheader()
        with torch.no_grad():
            for ep_idx in range(start_rollout, end_rollout):
                ep = data[ep_idx]
                traj = ep.get("trajectory", [])
                if len(traj) <= history_len:
                    continue
                image_paths = [
                    _materialize_image(_step_image_path(step), images_base_dir, images_zip, cache_dir)
                    for step in traj
                ]
                responses = [_step_response(step) for step in traj]
                z = _encode_episode_latents(
                    adapter=qwen_adapter,
                    image_paths=image_paths,
                    prompt=str(ep.get("input") or ep.get("instruction") or ""),
                    responses=responses,
                    device=device,
                    num_patches=int(cfg.wm.num_patches),
                    token_dim=int(cfg.wm.token_dim),
                    micro_batch_size=int(args.encode_micro_batch_size),
                )
                actions = [_step_action_id(step) for step in traj]
                action_vecs = torch.stack([
                    _action_tensor(aid, action_dim=action_dim, device=device, dtype=z.dtype) for aid in actions
                ], dim=0)

                batch_z = []
                batch_a = []
                meta = []
                for step_idx in range(history_len, len(traj)):
                    z_hist = z[step_idx - history_len : step_idx]
                    a_hist_base = action_vecs[step_idx - history_len : step_idx].clone()
                    for aid in action_ids:
                        a_hist = a_hist_base.clone()
                        a_hist[-1] = _action_tensor(aid, action_dim=action_dim, device=device, dtype=z.dtype)
                        batch_z.append(z_hist)
                        batch_a.append(a_hist)
                        meta.append((step_idx, aid))
                if not batch_z:
                    continue
                scores_by_step: dict[int, dict[int, float]] = defaultdict(dict)
                eval_batch = 64
                for start in range(0, len(batch_z), eval_batch):
                    end = min(len(batch_z), start + eval_batch)
                    bz = torch.stack(batch_z[start:end], dim=0)
                    ba = torch.stack(batch_a[start:end], dim=0)
                    pred_z = wm_model.predict_next(bz, ba)
                    scores = wm_model.predict_reward(pred_z).detach().float().cpu().tolist()
                    for (step_idx, aid), score in zip(meta[start:end], scores):
                        scores_by_step[step_idx][aid] = float(score)

                for step_idx in range(history_len, len(traj)):
                    scores = scores_by_step[step_idx]
                    pred_a = max(action_ids, key=lambda aid: scores.get(aid, float("-inf")))
                    true_a = actions[step_idx - 1]
                    y_true.append(true_a)
                    y_pred.append(pred_a)
                    per_eval_set[str(ep.get("eval_set", ""))].append((true_a, pred_a))
                    row = {
                        "rollout_index": ep_idx,
                        "episode_id": ep.get("episode_id", ""),
                        "eval_set": ep.get("eval_set", ""),
                        "step_index": step_idx,
                        "gt_action_id": true_a,
                        "pred_action_id": pred_a,
                        "correct": int(true_a == pred_a),
                    }
                    row.update({f"score_a{aid}": scores.get(aid, "") for aid in action_ids})
                    writer.writerow(row)

                if (ep_idx + 1) % 25 == 0 or ep_idx + 1 == end_rollout:
                    acc = sum(int(t == p) for t, p in zip(y_true, y_pred)) / max(1, len(y_true))
                    print(f"progress rollout={ep_idx + 1}/{end_rollout} examples={len(y_true)} acc={acc:.4f}")
                if not args.keep_image_cache:
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                del z
                torch.cuda.empty_cache()

    if images_zip is not None:
        images_zip.close()

    labels = action_ids
    total = len(y_true)
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    per_class = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        per_class.append({
            "action_id": label,
            "action_name": ACTION_NAMES.get(label, ""),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    macro_precision = sum(x["precision"] for x in per_class) / len(per_class)
    macro_recall = sum(x["recall"] for x in per_class) / len(per_class)
    macro_f1 = sum(x["f1"] for x in per_class) / len(per_class)
    micro_tp = sum(x["tp"] for x in per_class)
    micro_fp = sum(x["fp"] for x in per_class)
    micro_fn = sum(x["fn"] for x in per_class)
    micro_precision, micro_recall, micro_f1 = _precision_recall_f1(micro_tp, micro_fp, micro_fn)

    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset": args.dataset,
        "rollout_start": start_rollout,
        "rollout_end_exclusive": end_rollout,
        "history_len": history_len,
        "num_examples": total,
        "accuracy": correct / total if total else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "true_distribution": dict(Counter(y_true)),
        "pred_distribution": dict(Counter(y_pred)),
        "per_class": per_class,
        "per_eval_set": {},
    }
    for split, pairs in sorted(per_eval_set.items()):
        s_total = len(pairs)
        s_correct = sum(1 for t, p in pairs if t == p)
        summary["per_eval_set"][split] = {"num_examples": s_total, "accuracy": s_correct / s_total if s_total else 0.0}

    (output_dir / "value_head_action_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(output_dir / "value_head_action_per_class.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class[0].keys()))
        writer.writeheader()
        writer.writerows(per_class)

    print(json.dumps({k: summary[k] for k in ["num_examples", "accuracy", "macro_precision", "macro_recall", "macro_f1", "micro_precision", "micro_recall", "micro_f1"]}, indent=2))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
