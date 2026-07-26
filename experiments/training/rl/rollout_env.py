#!/usr/bin/env python3
"""使用 Nimloth policy 采集可供 RL 使用的 navigation trajectory。

这是分布式 JSONL 训练使用的 rollout producer。在线采集与两阶段采集共享同一
套 Agent runner、动作 policy 和 trajectory schema。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch


_NAV_DATASETS = (
    "base",
    "common_sense",
    "complex_instruction",
    "visual_appearance",
    "long_horizon",
    "base_train",
    "common_sense_train",
    "long_horizon_train",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect Nimloth RL trajectories")
    ap.add_argument("--model", type=Path, required=True, help="Full HF policy checkpoint")
    ap.add_argument("--env-url", required=True, help="VAGEN env server base URL")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--num-episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=20)
    dataset_group = ap.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--eval-set", choices=_NAV_DATASETS)
    dataset_group.add_argument(
        "--eval-sets",
        choices=_NAV_DATASETS,
        nargs="+",
        help="Round-robin over multiple datasets while preserving one seed stream",
    )
    ap.add_argument("--split", choices=("train", "val", "test", "eval"), required=True)
    ap.add_argument("--seed-offset", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument(
        "--credit-assignment",
        choices=("action", "turn", "token"),
        default="action",
    )
    ap.add_argument(
        "--action-objective",
        choices=("distillation", "ppo"),
        default="ppo",
    )
    ap.add_argument("--max-response-tokens", type=int, default=64)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--max-pixels", type=int, default=3136)
    ap.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--vllm-enforce-eager", action="store_true")
    ap.add_argument("--vllm-enable-prefix-caching", action="store_true")
    ap.add_argument("--vllm-mm-processor-cache-gb", type=float, default=0.0)
    ap.add_argument(
        "--vllm-distributed-executor-backend",
        choices=("mp", "ray"),
        default=None,
    )
    ap.add_argument("--planner-enabled", action="store_true")
    ap.add_argument("--planning-horizon", type=int, default=None)
    ap.add_argument(
        "--planning-search-mode",
        choices=("greedy", "exhaustive", "beam"),
        default=None,
    )
    ap.add_argument("--planning-beam-width", type=int, default=None)
    ap.add_argument("--planner-device", default=None)
    ap.add_argument("--wm-checkpoint", type=Path, default=None)
    ap.add_argument("--state-proj-checkpoint", type=Path, default=None)
    ap.add_argument("--value-head-checkpoint", type=Path, default=None)
    ap.add_argument(
        "--fresh-manifest",
        type=Path,
        default=None,
        help="记录当前 policy artifact 指纹，供随后唯一一次 RL update 消费",
    )
    return ap.parse_args(argv)


def load_qwen(
    model_path: Path,
    attn_implementation: str,
    max_pixels: int,
    *,
    latent_token_count: int,
):
    """在当前 CUDA device 上加载 rollout policy 及 processor。"""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from nimloth.backbone.qwen25vl.policy import validate_agent_policy_protocol
    from nimloth.latent import add_special_tokens

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = max_pixels
    n_added = add_special_tokens(
        processor.tokenizer,
        latent_token_count=latent_token_count,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    if n_added:
        model.resize_token_embeddings(len(processor.tokenizer))
    loaded_count = validate_agent_policy_protocol(model.config)
    if loaded_count != latent_token_count:
        raise ValueError(
            "checkpoint latent token count changed while loading: "
            f"config={latent_token_count}, model={loaded_count}"
        )
    model.eval().cuda()
    return model, processor


def validate_split(eval_set: str, split: str) -> None:
    """防止 evaluation 数据集被错误标记为训练数据。"""
    if split == "train" and not eval_set.endswith("_train"):
        raise ValueError(f"refusing to label eval dataset {eval_set!r} as training data")
    if split != "train" and eval_set.endswith("_train"):
        raise ValueError(f"training dataset {eval_set!r} must use --split train")


def validate_trajectories(records, *, expected_count: int | None = None) -> None:
    """在 trajectory 进入 FSDP 训练前拒绝不完整记录。"""
    from nimloth.rollout import validate_rollout_trajectory

    if not records:
        raise RuntimeError("rollout produced no complete trajectories")
    if expected_count is not None and len(records) != expected_count:
        raise RuntimeError(
            "rollout produced an incomplete trajectory batch: "
            f"{len(records)} != {expected_count}"
        )
    for record in records:
        if record.split == "train" and not record.image_paths:
            raise RuntimeError(f"training trajectory {record.record_id} has no images")
        try:
            validate_rollout_trajectory(record)
        except ValueError as error:
            raise RuntimeError(str(error)) from error


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    eval_sets = tuple(args.eval_sets or (args.eval_set,))
    for eval_set in eval_sets:
        validate_split(eval_set, args.split)
    if not torch.cuda.is_available():
        raise RuntimeError("rollout_env requires CUDA")
    if args.vllm_mm_processor_cache_gb < 0.0:
        raise ValueError("vllm_mm_processor_cache_gb must be non-negative")
    if args.planner_enabled:
        missing = [
            name
            for name, value in (
                ("planning_horizon", args.planning_horizon),
                ("planning_search_mode", args.planning_search_mode),
                ("planner_device", args.planner_device),
                ("wm_checkpoint", args.wm_checkpoint),
                ("state_proj_checkpoint", args.state_proj_checkpoint),
                ("value_head_checkpoint", args.value_head_checkpoint),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "planner rollout requires explicit arguments: " + ", ".join(missing)
            )
        if args.backend != "vllm":
            raise ValueError("planner rollout requires --backend vllm")
        if args.credit_assignment != "action":
            raise ValueError("planner rollout requires --credit-assignment action")
        if args.action_objective != "distillation":
            raise ValueError(
                "greedy planner rollout requires --action-objective distillation"
            )
        if not args.vllm_enforce_eager:
            raise ValueError("planner rollout requires --vllm-enforce-eager")
        if args.planning_horizon is not None and args.planning_horizon < 1:
            raise ValueError("planning_horizon must be positive")
        if args.planning_search_mode == "beam":
            if args.planning_beam_width is None or args.planning_beam_width < 1:
                raise ValueError("beam planner requires a positive beam width")
        elif args.planning_beam_width is not None:
            raise ValueError("planning_beam_width is only valid for beam search")
    elif any(
        value is not None
        for value in (
            args.planning_horizon,
            args.planning_search_mode,
            args.planning_beam_width,
            args.planner_device,
            args.wm_checkpoint,
            args.state_proj_checkpoint,
            args.value_head_checkpoint,
        )
    ):
        raise ValueError("planner arguments require --planner-enabled")

    repo_root = Path(__file__).resolve().parents[3]
    for path in (repo_root / "src", repo_root / "external" / "VAGEN"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from nimloth.backbone.qwen25vl.policy import QwenAgentPolicy
    from nimloth.backbone.qwen25vl.loading import load_qwen_processor
    from nimloth.backbone.qwen25vl.vllm_policy import QwenVLLMAgentPolicy
    from nimloth.environment.navigation.collector import VAGENNavigationRolloutCollector
    from nimloth.rollout import FreshRolloutManifest

    def report_policy_progress(stage: str) -> None:
        print(
            json.dumps({
                "rl_policy_stage": stage,
                "monotonic_s": round(time.monotonic(), 3),
            }),
            flush=True,
        )

    if args.backend == "vllm":
        from transformers import AutoConfig
        from nimloth.backbone.qwen25vl.policy import validate_agent_policy_protocol

        latent_token_count = validate_agent_policy_protocol(
            AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        )
        processor = load_qwen_processor(
            args.model,
            max_pixels=args.max_pixels,
            latent_token_count=latent_token_count,
        ).processor
        policy = QwenVLLMAgentPolicy.from_model(
            str(args.model),
            processor=processor,
            tensor_parallel_size=args.tensor_parallel_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_model_len=args.max_model_len,
            max_images=args.max_steps + 1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            latent_token_count=latent_token_count,
            credit_assignment=("token" if args.planner_enabled else args.credit_assignment),
            max_response_tokens=args.max_response_tokens,
            distributed_executor_backend=args.vllm_distributed_executor_backend,
            enforce_eager=args.vllm_enforce_eager,
            capture_policy_state=args.planner_enabled,
            enable_prefix_caching=args.vllm_enable_prefix_caching,
            mm_processor_cache_gb=args.vllm_mm_processor_cache_gb,
            progress_callback=report_policy_progress,
        )
        if args.planner_enabled:
            from nimloth.agent import PlanningPolicy
            from nimloth.training.rl.planning_loader import (
                load_planning_world_model,
            )

            assert args.planner_device is not None
            planner_device = torch.device(args.planner_device)
            assert args.wm_checkpoint is not None
            assert args.state_proj_checkpoint is not None
            assert args.value_head_checkpoint is not None
            world_model = load_planning_world_model(
                qwen_config=AutoConfig.from_pretrained(
                    args.model,
                    trust_remote_code=True,
                ),
                wm_checkpoint=args.wm_checkpoint,
                state_proj_checkpoint=args.state_proj_checkpoint,
                value_head_checkpoint=args.value_head_checkpoint,
                device=planner_device,
            )
            assert args.planning_horizon is not None
            assert args.planning_search_mode is not None
            policy = PlanningPolicy(
                turn_policy=policy,
                world_model=world_model,
                horizon=args.planning_horizon,
                search_mode=args.planning_search_mode,
                beam_width=args.planning_beam_width,
                planner_device=planner_device,
                action_objective=args.action_objective,
                progress_callback=report_policy_progress,
            )
    else:
        if args.credit_assignment != "action":
            raise ValueError(
                f"{args.credit_assignment} credit rollout requires --backend vllm"
            )
        from transformers import AutoConfig
        from nimloth.backbone.qwen25vl.policy import validate_agent_policy_protocol

        latent_token_count = validate_agent_policy_protocol(
            AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        )
        model, processor = load_qwen(
            args.model,
            args.attn_implementation,
            args.max_pixels,
            latent_token_count=latent_token_count,
        )
        policy = QwenAgentPolicy(
            model=model,
            processor=processor,
            device=torch.device("cuda"),
            temperature=args.temperature,
            top_p=args.top_p,
            latent_token_count=latent_token_count,
        )
    collector = VAGENNavigationRolloutCollector(
        policy=policy,
        env_url=args.env_url,
        seed_offset=args.seed_offset,
        temperature=args.temperature,
        top_p=args.top_p,
        eval_sets=eval_sets,
        split=args.split,
        latent_token_count=latent_token_count,
    )
    trajectories = collector.collect(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        output_dir=args.output_dir,
    )
    validate_trajectories(trajectories, expected_count=args.num_episodes)
    manifest_path = args.fresh_manifest
    if manifest_path is not None:
        FreshRolloutManifest.create(
            policy_path=args.model,
            trajectory_path=args.output_dir / "trajectories.jsonl",
            num_trajectories=len(trajectories),
            planner_artifacts=(
                {
                    "wm_predictor": args.wm_checkpoint,
                    "state_projector": args.state_proj_checkpoint,
                    "value_head": args.value_head_checkpoint,
                }
                if args.planner_enabled
                else None
            ),
        ).write(manifest_path)
    finish_reasons = Counter(
        reason
        for trajectory in trajectories
        for reason in trajectory.policy_finish_reasons
        if reason is not None
    )
    print(json.dumps({
        "status": "ALL_OK",
        "num_trajectories": len(trajectories),
        "num_transitions": sum(t.num_steps for t in trajectories),
        "jsonl": str(args.output_dir / "trajectories.jsonl"),
        "fresh_manifest": str(manifest_path) if manifest_path else None,
        "eval_sets": eval_sets,
        "reasoning_truncated": sum(
            int(value)
            for trajectory in trajectories
            for value in trajectory.policy_reasoning_truncated
        ),
        "reasoning_finish_reasons": dict(sorted(finish_reasons.items())),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
