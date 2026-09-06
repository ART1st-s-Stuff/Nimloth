"""Explicit direct or WM evaluation using the shared real-environment rollout producer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nimloth.training.sft.evaluation.config import EvaluationConfig
from nimloth.training.sft.evaluation.rollout import _NAV_DATASETS
from nimloth.training.sft.evaluation.rollout import main as rollout_main
from nimloth.training.sft.stage3.mcts_evaluation import (
    SFT2MCTSEvaluationContract,
    load_sft2_mcts_evaluation_contract,
)

_HELD_OUT_DATASETS = tuple(
    dataset for dataset in _NAV_DATASETS if not dataset.endswith("_train")
)


def parse_args(argv: list[str] | None = None) -> EvaluationConfig:
    ap = argparse.ArgumentParser(
        description="SFT direct/WM real-environment rollout evaluation"
    )
    ap.add_argument("--mode", choices=("direct", "wm"), required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--env-url", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
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
    ap.add_argument("--num-simulations", type=int, default=None)
    ap.add_argument("--exploration-constant", type=float, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, required=True)
    ap.add_argument("--planner-device", default=None)
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
    return EvaluationConfig(**vars(ap.parse_args(argv)))


def build_rollout_argv(
    args: EvaluationConfig,
    contract: SFT2MCTSEvaluationContract | None,
) -> list[str]:
    if (args.mode == "wm") != (contract is not None):
        raise ValueError("WM evaluation requires its validated checkpoint contract")
    if contract is not None and args.num_simulations < contract.action_count:
        raise ValueError("num_simulations must visit every root action at least once")

    total_episodes = args.episodes_per_eval_set * len(args.eval_sets)
    rollout_args = [
        "--backend",
        "vllm",
        "--model",
        str(args.checkpoint.resolve()),
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
    if contract is not None:
        rollout_args.extend([
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
        ])
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
    if args.resume:
        rollout_args.append("--resume-existing-rollouts")
    return rollout_args


def write_or_validate_contract(
    output_dir: Path,
    metadata: dict[str, object],
    *,
    resume: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "evaluation_contract.json"
    other_entries = [path for path in output_dir.iterdir() if path != contract_path]
    if contract_path.exists():
        if not resume:
            raise FileExistsError(
                f"refusing to overwrite evaluation output: {output_dir}"
            )
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError("resume evaluation contract does not match requested run")
        return
    if other_entries:
        raise FileExistsError(
            "evaluation output has artifacts but no contract: "
            f"{output_dir}"
        )
    contract_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def run_evaluation(args: EvaluationConfig) -> int:
    """Run real episodes; offline training loss validation is a separate API."""
    contract = (
        load_sft2_mcts_evaluation_contract(args.checkpoint)
        if args.mode == "wm" else None
    )
    rollout_args = build_rollout_argv(args, contract)
    from nimloth.rollout.fresh import (
        auxiliary_artifact_fingerprint,
        policy_artifact_fingerprint,
    )

    planner_artifacts = ({
        "wm_predictor": contract.wm_checkpoint,
        "state_projector": contract.state_proj_checkpoint,
        "value_head": contract.value_head_checkpoint,
    } if contract is not None else {})
    # Record every behavior/runtime argument: resume cannot silently change models,
    # rendering inputs, generation, episode seeds or search.
    metadata = {
        "evaluation": f"sft_eval_{args.mode}_v1",
        "policy_fingerprint": policy_artifact_fingerprint(args.checkpoint),
        "planner_fingerprints": {
            name: auxiliary_artifact_fingerprint(path)
            for name, path in planner_artifacts.items()
        },
        "rollout_argv": [value for value in rollout_args
                         if value != "--resume-existing-rollouts"],
        "leaf_value": ("decision_state_K_minus_1_final_action_mc"
                       if contract is not None else None),
    }
    write_or_validate_contract(args.output_dir, metadata, resume=args.resume)
    print(json.dumps({"preflight": metadata}), flush=True)
    return rollout_main(rollout_args)


def eval_direct(config: EvaluationConfig) -> int:
    """Generate an action for every real observation, without the world model."""
    if config.mode != "direct":
        raise ValueError("eval_direct requires mode='direct'")
    return run_evaluation(config)


def eval_wm(config: EvaluationConfig) -> int:
    """Replan with real observation-aligned Qwen state after every executed action."""
    if config.mode != "wm":
        raise ValueError("eval_wm requires mode='wm'")
    return run_evaluation(config)


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    return eval_direct(config) if config.mode == "direct" else eval_wm(config)


if __name__ == "__main__":
    raise SystemExit(main())
