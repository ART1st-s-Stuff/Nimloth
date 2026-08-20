#!/usr/bin/env python3
"""Validate, summarize, and publish the paired pre-SFT2 policy evaluations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any

from PIL import Image


EVAL_SETS = (
    "base",
    "common_sense",
    "complex_instruction",
    "visual_appearance",
    "long_horizon",
)
SOURCE_BY_EVAL_SET = {
    "base": "navigation_base_test",
    "common_sense": "navigation_common_test",
    "complex_instruction": "navigation_complex_instruction_test",
    "visual_appearance": "navigation_visual_appearance_test",
    "long_horizon": "navigation_long_horizon_test",
}
_ACTION_BLOCK = re.compile(
    r"<\|action_start\|><\|action_\([0-7]\)\|><\|action_end\|>"
)
_XML_ACTION = re.compile(r"<action>\s*[^<]+?\s*</action>", re.DOTALL)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def _validate_images(paths: list[str]) -> dict[str, Any]:
    if not paths:
        raise ValueError("evaluation contains no image references")
    sizes: Counter[tuple[int, int]] = Counter()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing rollout image: {path}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if max(high - low for low, high in rgb.getextrema()) == 0:
                raise ValueError(f"uniform rollout image: {path}")
            sizes[rgb.size] += 1
    return {
        "references": len(paths),
        "sizes": {f"{width}x{height}": count for (width, height), count in sizes.items()},
        "uniform_images": 0,
    }


def _metrics(successes: list[bool], rewards: list[float], steps: list[int]) -> dict[str, float]:
    count = len(successes)
    if count == 0:
        raise ValueError("cannot summarize an empty evaluation")
    values = {
        "success_rate": sum(successes) / count,
        "avg_reward": sum(rewards) / count,
        "avg_steps": sum(steps) / count,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"non-finite metrics: {values}")
    return values


def _finalize_sft1(output_dir: Path, episodes_per_set: int) -> tuple[dict[str, Any], dict[str, Any]]:
    by_eval_set: dict[str, dict[str, float]] = {}
    all_successes: list[bool] = []
    all_rewards: list[float] = []
    all_steps: list[int] = []
    all_images: list[str] = []
    action_blocks = 0
    action_responses = 0
    expected_latent_block = "".join(
        ("<|latent_state|>", *(f"<|latent_state_{index}|>" for index in range(1, 16)))
    )
    for eval_set in EVAL_SETS:
        dataset_dir = output_dir / "eval_sets" / eval_set
        summary = _read_json(dataset_dir / "rollout_summary.json")
        rows = _read_jsonl(dataset_dir / "trajectories.jsonl")
        if summary.get("status") != "ALL_OK" or len(rows) != episodes_per_set:
            raise ValueError(
                f"{eval_set} incomplete: status={summary.get('status')!r} rows={len(rows)}"
            )
        expected_ids = [
            f"rl_{eval_set}_{seed:06d}"
            for seed in range(1, episodes_per_set + 1)
        ]
        actual_ids = [str(row.get("id")) for row in rows]
        if actual_ids != expected_ids:
            raise ValueError(f"{eval_set} does not contain the requested contiguous seed range")
        successes: list[bool] = []
        rewards: list[float] = []
        steps: list[int] = []
        for row in rows:
            if row.get("split") != "test":
                raise ValueError(f"{eval_set} row is not marked as heldout test data")
            prompt_spec = row.get("prompt_template", {})
            config = prompt_spec.get("config", {}) if isinstance(prompt_spec, dict) else {}
            if int(config.get("latent_token_count", -1)) != 16:
                raise ValueError(f"{eval_set} row has the wrong latent token count")
            if float(row.get("sampling_temperature", -1.0)) != 0.0:
                raise ValueError(f"{eval_set} row is not greedy")
            if float(row.get("sampling_top_p", -1.0)) != 1.0:
                raise ValueError(f"{eval_set} row has the wrong top_p")
            system_prompt = str(row.get("system_prompt", ""))
            if (
                "Actions you can take: move_forward" not in system_prompt
                or "<|action_start|><|action_(idx)|><|action_end|>" not in system_prompt
                or "Format correct: +0.5" in system_prompt
            ):
                raise ValueError(f"{eval_set} row is not using the source-compatible Nimloth prompt")
            responses = row.get("assistant_responses")
            if not isinstance(responses, list) or not responses:
                raise ValueError(f"{eval_set} row has no real assistant responses")
            for response in responses:
                text = str(response)
                action_responses += 1
                if expected_latent_block not in text or _ACTION_BLOCK.search(text) is None:
                    raise ValueError(f"{eval_set} row violates the k16 Nimloth action protocol")
                action_blocks += 1
            successes.append(bool(row.get("success")))
            rewards.append(float(row.get("reward", 0.0)))
            steps.append(len(row.get("action_indices", ())))
            image_paths = row.get("image_paths")
            if not isinstance(image_paths, list) or not image_paths:
                raise ValueError(f"{eval_set} row has no images")
            all_images.extend(str(path) for path in image_paths)
        by_eval_set[eval_set] = _metrics(successes, rewards, steps)
        all_successes.extend(successes)
        all_rewards.extend(rewards)
        all_steps.extend(steps)
    images = _validate_images(all_images)
    diagnostics = {
        "action_format": "nimloth_k16_injected",
        "action_responses": action_responses,
        "valid_action_blocks": action_blocks,
        "action_format_rate": action_blocks / action_responses,
        "images": images,
    }
    return {
        "overall": _metrics(all_successes, all_rewards, all_steps),
        "by_eval_set": by_eval_set,
    }, diagnostics


def _finalize_vagen(
    output_dir: Path,
    episodes_per_set: int,
    jsonl_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = _read_jsonl(jsonl_path)
    expected_count = episodes_per_set * len(EVAL_SETS)
    if len(rows) != expected_count:
        raise ValueError(f"VAGEN dump has {len(rows)} rows, expected {expected_count}")
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in EVAL_SETS}
    seen: set[tuple[str, int]] = set()
    images: list[str] = []
    xml_valid = 0
    for index, row in enumerate(rows):
        source = str(row.get("data_source"))
        eval_set = str(row.get("eval_set"))
        seed = int(row.get("env_seed", -1))
        if SOURCE_BY_EVAL_SET.get(eval_set) != source or not 1 <= seed <= episodes_per_set:
            raise ValueError(
                f"VAGEN row {index} has wrong task identity: {source}/{eval_set}/{seed}"
            )
        key = (source, seed)
        if key in seen:
            raise ValueError(f"duplicate VAGEN task identity: {key}")
        seen.add(key)
        expected_uid = f"{source}:{seed}:{eval_set}"
        if row.get("uid") != expected_uid:
            raise ValueError(f"VAGEN row {index} has unstable metadata identity")
        if f"eval_set={eval_set}," not in str(row.get("config_id", "")):
            raise ValueError(f"VAGEN row {index} config_id/eval_set mismatch")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or "success" not in metrics:
            raise ValueError(f"VAGEN row {index} has no success metric")
        output = str(row.get("output_str", ""))
        xml_valid += int(_XML_ACTION.search(output) is not None)
        image_paths = row.get("image_paths")
        if not isinstance(image_paths, list) or not image_paths:
            raise ValueError(f"VAGEN row {index} has no images")
        images.extend(str(path) for path in image_paths)
        grouped[eval_set].append(row)
    expected_keys = {
        (source, seed)
        for source in SOURCE_BY_EVAL_SET.values()
        for seed in range(1, episodes_per_set + 1)
    }
    if seen != expected_keys:
        raise ValueError("VAGEN dump does not cover the exact requested task keys")

    by_eval_set: dict[str, dict[str, float]] = {}
    all_successes: list[bool] = []
    all_rewards: list[float] = []
    all_steps: list[int] = []
    for eval_set, items in grouped.items():
        successes = [bool(item["metrics"]["success"]) for item in items]
        rewards = [float(item["metrics"].get("score", 0.0)) for item in items]
        steps = [int(item["metrics"].get("step", 0)) for item in items]
        by_eval_set[eval_set] = _metrics(successes, rewards, steps)
        all_successes.extend(successes)
        all_rewards.extend(rewards)
        all_steps.extend(steps)
    diagnostics = {
        "action_format": "vagen_eval_mode_xml",
        "trajectories_with_xml_action": xml_valid,
        "action_format_rate": xml_valid / expected_count,
        "metadata_mismatches": 0,
        "images": _validate_images(images),
    }
    return {
        "overall": _metrics(all_successes, all_rewards, all_steps),
        "by_eval_set": by_eval_set,
    }, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("sft1", "vagen"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--vagen-commit", required=True)
    parser.add_argument("--episodes-per-set", type=int, default=60)
    parser.add_argument("--vagen-jsonl", type=Path)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    return parser.parse_args()


def _merge_sft1_rollout_browsers(
    output_dir: Path,
    *,
    expected_rollouts: int,
) -> Path:
    from nimloth.eval.rollout_browser import merge_evaluation_browsers

    sources = [
        output_dir / "eval_sets" / eval_set / "evaluation_browser"
        for eval_set in EVAL_SETS
    ]
    first_manifest = _read_json(sources[0] / "manifest.json")
    destination = output_dir / "evaluation_browser"
    merge_evaluation_browsers(
        destination,
        sources,
        evaluation={
            "evaluation_id": output_dir.name,
            "policy_family": "sft_policy",
            "global_step": first_manifest.get("global_step"),
            "source_step": first_manifest.get("source_step"),
            "checkpoint_identity": first_manifest["checkpoint_identity"],
            "snapshot_identity": first_manifest.get("snapshot_identity"),
        },
        expected_rollouts=expected_rollouts,
    )
    return destination


def main() -> int:
    args = parse_args()
    if args.episodes_per_set < 1:
        raise ValueError("episodes_per_set must be positive")
    if args.arm == "sft1":
        metrics, diagnostics = _finalize_sft1(args.output_dir, args.episodes_per_set)
    else:
        if args.vagen_jsonl is None:
            raise ValueError("--vagen-jsonl is required for the VAGEN arm")
        metrics, diagnostics = _finalize_vagen(
            args.output_dir,
            args.episodes_per_set,
            args.vagen_jsonl,
        )
    contract = {
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "nimloth_commit": args.commit,
        "vagen_commit": args.vagen_commit,
        "frozen_modules": "all",
        "optimizer": None,
        "datasets": {name: {"split": "test", "seeds": [1, args.episodes_per_set]} for name in EVAL_SETS},
        "sampling": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "n": 1,
            "max_response_tokens_per_turn": 512,
            "max_turns": 20,
            "max_actions_per_turn": 1,
        },
        "navigation": {
            "resolution": 255,
            "step_length": 0.3,
            "success_threshold": 1.0,
            "format_reward": 0.0,
            "per_turn_format_reward": 0.01,
            "success_reward": 1.0,
            "effective_format_reward_per_valid_turn": 0.01,
        },
        "resume": (
            "contiguous per-eval-set episode prefix"
            if args.arm == "sft1"
            else "no partial VAGEN dump; retry uses a new attempt directory"
        ),
    }
    summary = {
        "status": "ALL_OK",
        "evaluation": "sft1_and_vagen_parent_checkpoints_pre_sft2",
        "arm": args.arm,
        "num_trajectories": args.episodes_per_set * len(EVAL_SETS),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "contract": contract,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.arm == "sft1":
        browser = _merge_sft1_rollout_browsers(
            args.output_dir,
            expected_rollouts=summary["num_trajectories"],
        )
        summary["rollout_browser"] = str(browser / "index.html")
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    import wandb

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume="never",
        config=contract,
        dir=str(args.output_dir),
    )
    logged = {
        "eval/overall/success_rate": metrics["overall"]["success_rate"],
        "eval/overall/avg_reward": metrics["overall"]["avg_reward"],
        "eval/overall/avg_steps": metrics["overall"]["avg_steps"],
        "eval/action_format_rate": diagnostics["action_format_rate"],
        "eval/num_trajectories": summary["num_trajectories"],
    }
    for eval_set, values in metrics["by_eval_set"].items():
        for key, value in values.items():
            logged[f"eval/{eval_set}/{key}"] = value
    run.log(logged)
    for key, value in logged.items():
        run.summary[key] = value
    run.finish(exit_code=0)
    (args.output_dir / "done.flag").write_text("ALL_OK\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
