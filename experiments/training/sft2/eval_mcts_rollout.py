#!/usr/bin/env python3
"""Evaluate one completed SFT2 checkpoint with real H=1/K-step MCTS rollout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.training.rl.rollout_env import (  # noqa: E402
    _NAV_DATASETS,
    main as rollout_main,
)
from nimloth.training.sft2.mcts_evaluation import (  # noqa: E402
    SFT2MCTSEvaluationContract,
    load_sft2_mcts_evaluation_contract,
)


_HELD_OUT_DATASETS = tuple(
    dataset for dataset in _NAV_DATASETS if not dataset.endswith("_train")
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Pre-RL success evaluation using the SFT2 H=1/K-step MCTS policy"
    )
    ap.add_argument("--sft2-checkpoint", type=Path, required=True)
    ap.add_argument("--env-url", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--eval-sets",
        choices=_HELD_OUT_DATASETS,
        nargs="+",
        required=True,
    )
    ap.add_argument("--split", choices=("val", "test", "eval"), required=True)
    ap.add_argument("--episodes-per-eval-set", type=int, required=True)
    ap.add_argument("--seed-offset", type=int, required=True)
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--top-p", type=float, required=True)
    ap.add_argument("--max-response-tokens", type=int, required=True)
    ap.add_argument("--num-simulations", type=int, required=True)
    ap.add_argument("--exploration-constant", type=float, required=True)
    ap.add_argument("--tensor-parallel-size", type=int, required=True)
    ap.add_argument("--planner-device", required=True)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--vllm-mm-processor-cache-gb", type=float, default=0.0)
    ap.add_argument("--vllm-enable-prefix-caching", action="store_true")
    ap.add_argument(
        "--vllm-distributed-executor-backend",
        choices=("mp", "ray"),
        default=None,
    )
    return ap.parse_args(argv)


def build_rollout_argv(
    args: argparse.Namespace,
    contract: SFT2MCTSEvaluationContract,
) -> list[str]:
    if args.episodes_per_eval_set < 1:
        raise ValueError("episodes_per_eval_set must be positive")
    if len(set(args.eval_sets)) != len(args.eval_sets):
        raise ValueError(f"duplicate eval_sets are not allowed: {args.eval_sets}")
    if args.num_simulations < contract.action_count:
        raise ValueError(
            "num_simulations must visit every root action at least once: "
            f"simulations={args.num_simulations}, actions={contract.action_count}"
        )

    total_episodes = args.episodes_per_eval_set * len(args.eval_sets)
    rollout_args = [
        "--backend",
        "vllm",
        "--planner-enabled",
        "--planning-horizon",
        str(contract.prediction_horizon),
        "--planning-search-mode",
        "mcts",
        "--mcts-num-simulations",
        str(args.num_simulations),
        "--mcts-exploration-constant",
        str(args.exploration_constant),
        "--planner-device",
        args.planner_device,
        "--wm-checkpoint",
        str(contract.wm_checkpoint),
        "--state-proj-checkpoint",
        str(contract.state_proj_checkpoint),
        "--value-head-checkpoint",
        str(contract.value_head_checkpoint),
        "--model",
        str(contract.checkpoint),
        "--env-url",
        args.env_url,
        "--output-dir",
        str(args.output_dir),
        "--num-episodes",
        str(total_episodes),
        "--max-steps",
        str(args.max_steps),
        "--eval-sets",
        *args.eval_sets,
        "--split",
        args.split,
        "--seed-offset",
        str(args.seed_offset),
        "--seed-per-eval-set",
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--credit-assignment",
        "action",
        "--max-response-tokens",
        str(args.max_response_tokens),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--vllm-mm-processor-cache-gb",
        str(args.vllm_mm_processor_cache_gb),
        "--vllm-enforce-eager",
    ]
    if args.max_pixels is not None:
        rollout_args.extend(("--max-pixels", str(args.max_pixels)))
    if args.vllm_enable_prefix_caching:
        rollout_args.append("--vllm-enable-prefix-caching")
    if args.vllm_distributed_executor_backend is not None:
        rollout_args.extend(
            (
                "--vllm-distributed-executor-backend",
                args.vllm_distributed_executor_backend,
            )
        )
    return rollout_args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty evaluation output: {args.output_dir}"
        )
    contract = load_sft2_mcts_evaluation_contract(args.sft2_checkpoint)
    rollout_args = build_rollout_argv(args, contract)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "evaluation": "sft2_pre_rl_mcts_v1",
        "sft2_checkpoint": str(contract.checkpoint),
        "checkpoint_step": contract.step,
        "checkpoint_epoch": contract.epoch,
        "history_size": contract.history_size,
        "prediction_horizon": contract.prediction_horizon,
        "leaf_value": "predicted_state_K_final_simulated_action_mc",
        "num_simulations": args.num_simulations,
        "exploration_constant": args.exploration_constant,
        "eval_sets": args.eval_sets,
        "split": args.split,
        "episodes_per_eval_set": args.episodes_per_eval_set,
        "seed_offset_per_eval_set": args.seed_offset,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_response_tokens": args.max_response_tokens,
    }
    (args.output_dir / "evaluation_contract.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"preflight": metadata}), flush=True)
    return rollout_main(rollout_args)


if __name__ == "__main__":
    raise SystemExit(main())
