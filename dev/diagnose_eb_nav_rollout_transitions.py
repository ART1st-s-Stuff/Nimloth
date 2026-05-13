"""Diagnose EB-Nav rollout transitions collected from random/semi-random policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ACTION_NAMES = {
    0: "Move Forward",
    1: "Move Backward",
    2: "Move Right",
    3: "Move Left",
    4: "Rotate Right",
    5: "Rotate Left",
    6: "Look Up",
    7: "Look Down",
}

try:
    from src.data.eb_nav_dataset import ACTION_NAMES as ACTION_NAMES_IMPORTED  # noqa: E402

    ACTION_NAMES = dict(ACTION_NAMES_IMPORTED)
except Exception:
    ACTION_NAMES = dict(DEFAULT_ACTION_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transitions-jsonl",
        nargs="+",
        required=True,
        help="One or more transitions.jsonl files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-transitions", type=int, default=0, help="Max transitions to load across all files (0 = all).")

    parser.add_argument("--no-latents", action="store_true", help="Skip Qwen latent diagnostics.")
    parser.add_argument("--cuda-device", default="0", help="CUDA device index or -1 for CPU.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model-dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--visual-num-tokens", type=int, default=16)
    parser.add_argument("--token-dim", type=int, default=3584)
    return parser.parse_args()


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def load_transitions(paths: list[str], max_transitions: int) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    source_paths: list[str] = []
    cap = int(max_transitions)
    for path_str in paths:
        path = Path(path_str)
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(item)
                source_paths.append(str(path))
                if cap > 0 and len(rows) >= cap:
                    return rows, source_paths
    return rows, source_paths


def build_base_stats(rows: list[dict[str, Any]], source_paths: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    action_counts: Counter[int] = Counter()
    logged_action_counts: Counter[int] = Counter()
    policy_counts: Counter[str] = Counter()

    rewards: list[float] = []
    action_success_values: list[float] = []
    done_values: list[float] = []
    task_success_values: list[float] = []
    distance_values: list[float] = []

    by_action: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "rewards": [],
            "action_success": [],
            "done": [],
            "task_success": [],
            "distance": [],
        }
    )

    for row in rows:
        sampled_action = _safe_int(row.get("sampled_action_id"), default=None)
        logged_action = _safe_int(row.get("logged_action_id"), default=None)
        policy = str(row.get("policy", "unknown"))

        reward = _safe_float(row.get("reward"), default=None)
        action_success = _safe_float(row.get("last_action_success"), default=None)
        done = _safe_float(row.get("done"), default=None)
        task_success = _safe_float(row.get("task_success"), default=None)
        distance = _safe_float(row.get("distance"), default=None)

        policy_counts[policy] += 1
        if sampled_action is not None:
            action_counts[sampled_action] += 1
        if logged_action is not None:
            logged_action_counts[logged_action] += 1

        if reward is not None:
            rewards.append(reward)
        if action_success is not None:
            action_success_values.append(action_success)
        if done is not None:
            done_values.append(done)
        if task_success is not None:
            task_success_values.append(task_success)
        if distance is not None:
            distance_values.append(distance)

        if sampled_action is not None:
            bucket = by_action[sampled_action]
            bucket["count"] = int(bucket["count"]) + 1
            if reward is not None:
                bucket["rewards"].append(reward)
            if action_success is not None:
                bucket["action_success"].append(action_success)
            if done is not None:
                bucket["done"].append(done)
            if task_success is not None:
                bucket["task_success"].append(task_success)
            if distance is not None:
                bucket["distance"].append(distance)

    per_action_rows: list[dict[str, Any]] = []
    for action_id in sorted(by_action.keys()):
        bucket = by_action[action_id]
        per_action_rows.append(
            {
                "action_id": int(action_id),
                "action_name": ACTION_NAMES.get(int(action_id), f"action_{action_id}"),
                "count": int(bucket["count"]),
                "reward_mean": _mean(bucket["rewards"]),
                "action_success_rate": _mean(bucket["action_success"]),
                "done_rate": _mean(bucket["done"]),
                "task_success_rate": _mean(bucket["task_success"]),
                "distance_mean": _mean(bucket["distance"]),
            }
        )

    summary = {
        "num_transitions": len(rows),
        "num_sources": len(set(source_paths)),
        "source_files": sorted(set(source_paths)),
        "policies": dict(sorted(policy_counts.items())),
        "action_distribution": {str(k): int(v) for k, v in sorted(action_counts.items())},
        "logged_action_distribution": {str(k): int(v) for k, v in sorted(logged_action_counts.items())},
        "per_action_counts": {str(r["action_id"]): int(r["count"]) for r in per_action_rows},
        "reward_mean": _mean(rewards),
        "action_success_rate": _mean(action_success_values),
        "done_rate": _mean(done_values),
        "task_success_rate": _mean(task_success_values),
        "distance_mean": _mean(distance_values),
    }
    return summary, per_action_rows


def encode_visual_tokens_cached(
    *,
    encoder: Any,
    image_path: str,
    cache: dict[str, Any],
) -> Any:
    if image_path in cache:
        return cache[image_path]
    encoded = encoder.encode_image_path_with_prompt(image_path=image_path, prompt_override=None)
    z = encoded.z
    if z.dim() == 1:
        z = z.unsqueeze(0)
    z = z.detach().float().cpu()
    cache[image_path] = z
    return z


def compute_latent_stats(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    model_dtype: str,
    device_map: str,
    cuda_device: str,
    visual_num_tokens: int,
    token_dim: int,
) -> tuple[dict[str, Any], dict[int, dict[str, list[float]]]]:
    import torch
    from src.vlm.qwen_adapter import QwenVLMAdapter
    from src.wm.encoder.qwen import QwenLLMLatentEncoder

    if str(cuda_device).strip() not in {"", "-1", "cpu"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device).strip()

    qwen_adapter = QwenVLMAdapter(
        model_name=model_name,
        latent_dim=int(visual_num_tokens) * int(token_dim),
        enabled=True,
        fallback_enabled=False,
        device_map=None if str(device_map).lower() in {"none", ""} else device_map,
        model_dtype=model_dtype,
    )
    qwen_adapter._ensure_model()
    if qwen_adapter._model is None:
        raise RuntimeError(f"Qwen init failed: {qwen_adapter.init_error}")
    qwen_adapter._model.eval()
    for param in qwen_adapter._model.parameters():
        param.requires_grad = False

    encoder = QwenLLMLatentEncoder(
        latent_dim=int(visual_num_tokens) * int(token_dim),
        name="qwen_visual_diag",
        model_name=model_name,
        qwen_adapter=qwen_adapter,
        use_vision_only=True,
        visual_pooling="tokens",
        visual_num_tokens=int(visual_num_tokens),
        cache_latents=True,
        llm_backbone_trainable=False,
    )

    latent_cache: dict[str, Any] = {}
    copy_mse_values: list[float] = []
    delta_l2_values: list[float] = []
    missing_pairs = 0
    usable_pairs = 0
    by_action: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"copy_mse": [], "delta_l2": []})

    for row in rows:
        path_t = str(row.get("image_t", ""))
        path_next = str(row.get("image_next", ""))
        if not path_t or not path_next:
            missing_pairs += 1
            continue
        if not Path(path_t).is_file() or not Path(path_next).is_file():
            missing_pairs += 1
            continue
        try:
            z_t = encode_visual_tokens_cached(encoder=encoder, image_path=path_t, cache=latent_cache)
            z_next = encode_visual_tokens_cached(encoder=encoder, image_path=path_next, cache=latent_cache)
        except Exception:
            missing_pairs += 1
            continue

        copy_mse = float(torch.mean((z_t - z_next) ** 2).item())
        delta_l2 = float(torch.norm((z_next - z_t).reshape(-1), p=2).item())
        copy_mse_values.append(copy_mse)
        delta_l2_values.append(delta_l2)
        usable_pairs += 1

        action_id = _safe_int(row.get("sampled_action_id"), default=None)
        if action_id is not None:
            by_action[action_id]["copy_mse"].append(copy_mse)
            by_action[action_id]["delta_l2"].append(delta_l2)

    stats = {
        "enabled": True,
        "usable_pairs": int(usable_pairs),
        "missing_or_failed_pairs": int(missing_pairs),
        "latent_cache_unique_paths": len(latent_cache),
        "copy_mse_mean": _mean(copy_mse_values),
        "copy_mse_std": float(torch.tensor(copy_mse_values).std(unbiased=False).item()) if copy_mse_values else None,
        "delta_l2_mean": _mean(delta_l2_values),
        "delta_l2_std": float(torch.tensor(delta_l2_values).std(unbiased=False).item()) if delta_l2_values else None,
    }
    return stats, by_action


def merge_latent_per_action(per_action_rows: list[dict[str, Any]], latent_by_action: dict[int, dict[str, list[float]]]) -> None:
    row_by_action = {int(row["action_id"]): row for row in per_action_rows}
    for action_id, values in latent_by_action.items():
        if action_id not in row_by_action:
            row_by_action[action_id] = {
                "action_id": int(action_id),
                "action_name": ACTION_NAMES.get(int(action_id), f"action_{action_id}"),
                "count": 0,
                "reward_mean": None,
                "action_success_rate": None,
                "done_rate": None,
                "task_success_rate": None,
                "distance_mean": None,
            }
            per_action_rows.append(row_by_action[action_id])

        row = row_by_action[action_id]
        row["copy_mse_mean"] = _mean(values.get("copy_mse", []))
        row["delta_l2_mean"] = _mean(values.get("delta_l2", []))
        row["latent_pairs"] = len(values.get("copy_mse", []))

    for row in per_action_rows:
        if "copy_mse_mean" not in row:
            row["copy_mse_mean"] = None
        if "delta_l2_mean" not in row:
            row["delta_l2_mean"] = None
        if "latent_pairs" not in row:
            row["latent_pairs"] = 0

    per_action_rows.sort(key=lambda item: int(item["action_id"]))


def write_outputs(output_dir: Path, summary: dict[str, Any], per_action_rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "action_id",
        "action_name",
        "count",
        "reward_mean",
        "action_success_rate",
        "done_rate",
        "task_success_rate",
        "distance_mean",
        "copy_mse_mean",
        "delta_l2_mean",
        "latent_pairs",
    ]
    with open(output_dir / "per_action_stats.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(per_action_rows, key=lambda item: int(item["action_id"])):
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()

    rows, source_paths = load_transitions(args.transitions_jsonl, args.max_transitions)
    if not rows:
        raise RuntimeError("No transitions loaded. Check --transitions-jsonl paths.")

    summary, per_action_rows = build_base_stats(rows, source_paths)

    latent_diag: dict[str, Any] = {"enabled": False, "reason": "disabled_by_flag"}
    if not args.no_latents:
        latent_diag = {"enabled": False, "reason": "not_run"}
        try:
            latent_stats, latent_by_action = compute_latent_stats(
                rows,
                model_name=str(args.model_name),
                model_dtype=str(args.model_dtype),
                device_map=str(args.device_map),
                cuda_device=str(args.cuda_device),
                visual_num_tokens=int(args.visual_num_tokens),
                token_dim=int(args.token_dim),
            )
            latent_diag = latent_stats
            merge_latent_per_action(per_action_rows, latent_by_action)
        except Exception as exc:
            latent_diag = {
                "enabled": False,
                "reason": "failed",
                "error": str(exc),
            }
    else:
        merge_latent_per_action(per_action_rows, {})

    summary["latents"] = latent_diag
    summary["args"] = vars(args)
    summary["distance_mean_is_nan"] = bool(summary.get("distance_mean") is not None and math.isnan(float(summary["distance_mean"])))

    write_outputs(Path(args.output_dir), summary, per_action_rows)

    print(json.dumps({
        "num_transitions": summary["num_transitions"],
        "output_dir": str(args.output_dir),
        "latents_enabled": bool(summary["latents"].get("enabled", False)),
    }))


if __name__ == "__main__":
    main()
