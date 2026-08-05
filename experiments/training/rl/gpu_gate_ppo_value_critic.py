"""Real-GPU integration gate for the planner PPO ValueHead critic.

This is a mechanics gate, not a policy-quality experiment.  It reads one real
planner transition whose persisted decision state was produced by the supplied
behavior checkpoint.  ``single_grad`` proves the critic-only graph reaches the
Qwen language body without supervising ``lm_head``.  ``ddp_step`` exercises the
production two-rank, two-GPU-per-rank wrapping and AdamW update for all configured
critic PPO epochs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoConfig

from nimloth.agent import Agent
from nimloth.backbone import (
    build_input_builder,
    load_backbone,
    model_output_device,
)
from nimloth.backbone.qwen25vl.policy import validate_agent_policy_protocol
from nimloth.config.rl import load_rl_config
from nimloth.rollout import (
    FreshJSONLRolloutCollector,
    FreshRolloutManifest,
    RolloutTrajectory,
    validate_rollout_trajectory,
)
from nimloth.training.rl.algorithm import RLAlgorithm
from nimloth.training.rl.episodes import build_episode_training_batches
from nimloth.training.rl.runtime import RLModelRuntime
from nimloth.training.rl.trainer import (
    _build_optimizer,
    _build_world_model,
    _wrap_distributed_modules,
)
from nimloth.util.distributed import (
    broadcast_module_state,
    setup_dist,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single_grad", "ddp_step"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wm-checkpoint", type=Path, required=True)
    parser.add_argument("--state-proj-checkpoint", type=Path, required=True)
    parser.add_argument("--value-head-checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory-jsonl", type=Path, required=True)
    parser.add_argument("--fresh-rollout-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus-per-rank", type=int, required=True)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def _model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        wm_checkpoint=args.wm_checkpoint,
        state_proj_checkpoint=args.state_proj_checkpoint,
        value_head_checkpoint=args.value_head_checkpoint,
        llm_tune="full",
        vision_tune="freeze",
        vision_ema=False,
        lora=False,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        gradient_checkpointing=True,
        max_pixels=None,
        attn_implementation=args.attn_implementation,
        resume=False,
        query_tune="freeze",
    )


def _load_trajectory(path: Path, record_index: int) -> RolloutTrajectory:
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index != record_index:
                continue
            trajectory = RolloutTrajectory.from_record(json.loads(line))
            validate_rollout_trajectory(trajectory)
            return trajectory
    raise IndexError(f"trajectory record {record_index} is absent from {path}")


def _parameter_with_suffix(
    module: torch.nn.Module,
    suffix: str,
) -> tuple[str, torch.nn.Parameter]:
    matches = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        names = [name for name, _ in matches]
        raise RuntimeError(f"expected one parameter ending in {suffix!r}, got {names}")
    return matches[0]


def _grad_max(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().abs().max().item())


def _all_grads_absent(module: torch.nn.Module) -> bool:
    return all(parameter.grad is None for parameter in module.parameters())


def _named_grads_absent(module: torch.nn.Module, marker: str) -> bool:
    matched = [
        parameter
        for name, parameter in module.named_parameters()
        if marker in name
    ]
    if not matched:
        raise RuntimeError(f"model has no parameters matching {marker!r}")
    return all(parameter.grad is None for parameter in matched)


def _gpu_memory(gpus_per_rank: int, local_rank: int) -> dict[str, int]:
    primary = local_rank * gpus_per_rank
    return {
        f"cuda:{index}": int(torch.cuda.max_memory_allocated(index))
        for index in range(primary, primary + gpus_per_rank)
    }


def _write_result(output_dir: Path, mode: str, rank: int, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{mode}_rank_{rank:02d}.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


SYNC_ABSOLUTE_TOLERANCE = 5e-7
SYNC_RELATIVE_TOLERANCE = 1e-6


def _assert_distributed_tensor_sync(
    tensor: torch.Tensor,
    *,
    label: str,
) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return 0.0
    witness = tensor.detach().reshape(-1)[:16].float()
    gathered = [torch.empty_like(witness) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, witness)
    max_difference = max(
        float((current - gathered[0]).abs().max().item()) for current in gathered[1:]
    )
    max_magnitude = max(float(current.abs().max().item()) for current in gathered)
    tolerance = (
        SYNC_ABSOLUTE_TOLERANCE
        + SYNC_RELATIVE_TOLERANCE * max_magnitude
    )
    if max_difference > tolerance:
        raise AssertionError(
            f"DDP {label} replicas diverged: difference={max_difference}, "
            f"tolerance={tolerance}"
        )
    return max_difference


def main() -> int:
    args = _parse_args()
    config = load_rl_config(args.config)
    expected_world = 1 if args.mode == "single_grad" else 2
    expected_gpus_per_rank = 1 if args.mode == "single_grad" else 2
    if args.gpus_per_rank != expected_gpus_per_rank:
        raise ValueError(
            f"{args.mode} requires --gpus-per-rank={expected_gpus_per_rank}"
        )

    rank, world_size, local_rank, device = setup_dist(
        gpu_stride=args.gpus_per_rank
    )
    if world_size != expected_world:
        raise RuntimeError(
            f"{args.mode} requires world_size={expected_world}, got {world_size}"
        )
    torch.manual_seed(config.training.seed)
    torch.cuda.reset_peak_memory_stats()

    try:
        loading_args = _model_args(args)
        manifest = FreshRolloutManifest.read(args.fresh_rollout_manifest)
        if Path(manifest.trajectory_path).resolve() != args.trajectory_jsonl.resolve():
            raise ValueError("manifest trajectory path differs from --trajectory-jsonl")
        freshness_validator = FreshJSONLRolloutCollector(
            args.fresh_rollout_manifest,
            model_path=args.model,
            planner_artifacts={
                "state_projector": args.state_proj_checkpoint,
                "value_head": args.value_head_checkpoint,
                "wm_predictor": args.wm_checkpoint,
            },
        )
        # Hashing the immutable behavior artifacts once is sufficient.  This gate
        # deliberately does not begin/commit consumption of the historical batch.
        if args.mode == "single_grad":
            freshness_validator.validate_policy()
        latent_token_count = validate_agent_policy_protocol(
            AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        )
        loaded = load_backbone(
            loading_args,
            device=device,
            latent_token_count=latent_token_count,
            model_parallel_size=args.gpus_per_rank,
        )
        if args.mode == "single_grad":
            manifest.validate_processor(loaded.processor)
        model = loaded.backbone.model
        training_device = model_output_device(model, default=device)
        world_model = _build_world_model(
            loading_args,
            config,
            llm=model,
            device=training_device,
        )
        # This gate isolates the critic.  The production loop's first-epoch WM/DINO
        # branch already has CPU coverage and would obscure the requested gradient.
        world_model.wm_predictor.requires_grad_(False).eval()
        if world_size > 1:
            broadcast_module_state(world_model.state_proj)
            broadcast_module_state(world_model.wm_predictor)
            broadcast_module_state(world_model.value_head)
        distributed = _wrap_distributed_modules(
            model,
            world_model,
            None,
            world_size=world_size,
            model_parallel=loaded.pair_parallel,
            training_device=training_device,
        )
        model = distributed.model
        world_model = distributed.world_model
        agent = Agent(backbone=loaded.backbone.with_model(model), wm=world_model)
        runtime = RLModelRuntime(
            agent=agent,
            input_builder=build_input_builder(
                loaded,
                max_length=999_999,
                latent_token_count=latent_token_count,
                mask_latent_query_labels=True,
            ),
            state_source="recompute",
            representation_to_backbone=True,
            policy_replay=None,
            max_state_tokens=config.actor.max_state_tokens,
        )
        algorithm = RLAlgorithm(
            history_size=config.predictor.history_size,
            sigreg=None,
            sigreg_weight=0.0,
            value_rank_margin=config.value_head.rank_margin,
            value_rank_weight=0.0,
            value_ppo_clip_range=config.value_head.ppo_clip_range,
            ppo_clip_ratio=config.actor.clip_ratio,
            entropy_weight=0.0,
            train_world_model=False,
            world_model_weight=0.0,
            dino_grid_weight=0.0,
        )

        trajectory = _load_trajectory(args.trajectory_jsonl, rank)
        truncated_bootstrap = (
            0.0 if config.rl.truncated_bootstrap == "zero" else None
        )
        episode = build_episode_training_batches(
            (trajectory,),
            gamma=config.rl.gamma,
            truncated_bootstrap=truncated_bootstrap,
        )[0]
        transition = episode.transitions[args.step_index]
        return_target = episode.return_targets[args.step_index]
        old_action_value = algorithm.planner_old_action_value(runtime, transition)

        qwen_name, qwen_witness = _parameter_with_suffix(
            model, "model.language_model.norm.weight"
        )
        lm_head_name, lm_head = _parameter_with_suffix(model, "lm_head.weight")
        value_name, value_witness = next(
            (name, parameter)
            for name, parameter in world_model.value_head.named_parameters()
            if parameter.requires_grad
        )
        qwen_before = qwen_witness.detach().clone()
        value_before = value_witness.detach().clone()

        optimizer = None
        if args.mode == "ddp_step":
            optimizer = _build_optimizer(model, world_model, None, config)

        epoch_metrics: list[dict[str, float]] = []
        qwen_grad_max = 0.0
        value_grad_max = 0.0
        qwen_grad_replica_max_difference = 0.0
        value_grad_replica_max_difference = 0.0
        qwen_parameter_replica_max_difference = 0.0
        value_parameter_replica_max_difference = 0.0
        epochs = 1 if args.mode == "single_grad" else config.value_head.ppo_epochs
        for ppo_epoch in range(epochs):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)
                world_model.value_head.zero_grad(set_to_none=True)
            output = algorithm.actor_transition_step(
                runtime,
                transition,
                return_target=return_target,
                old_action_value=old_action_value,
                total_transitions=world_size,
                include_world_model=False,
            )
            # Production planner sharding cancels DDP's gradient average this way.
            (output.loss * world_size).backward()
            qwen_grad_max = max(qwen_grad_max, _grad_max(qwen_witness))
            value_grad_max = max(value_grad_max, _grad_max(value_witness))
            if qwen_grad_max <= 0.0:
                raise AssertionError("critic loss did not reach the Qwen language body")
            if value_grad_max <= 0.0:
                raise AssertionError("critic loss did not reach ValueHead")
            if dist.is_available() and dist.is_initialized():
                assert qwen_witness.grad is not None
                assert value_witness.grad is not None
                qwen_grad_replica_max_difference = max(
                    qwen_grad_replica_max_difference,
                    _assert_distributed_tensor_sync(
                        qwen_witness.grad,
                        label="Qwen gradient witness",
                    ),
                )
                value_grad_replica_max_difference = max(
                    value_grad_replica_max_difference,
                    _assert_distributed_tensor_sync(
                        value_witness.grad,
                        label="ValueHead gradient witness",
                    ),
                )
            if lm_head.grad is not None:
                raise AssertionError("critic loss unexpectedly supervised lm_head")
            if not _all_grads_absent(world_model.state_proj):
                raise AssertionError("frozen StateProjector received parameter gradients")
            if not _named_grads_absent(model, ".visual."):
                raise AssertionError("frozen Qwen vision tower received gradients")
            epoch_metrics.append(dict(output.metrics))
            if optimizer is not None:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                    ],
                    max_norm=1.0,
                )
                optimizer.step()
                qwen_parameter_replica_max_difference = max(
                    qwen_parameter_replica_max_difference,
                    _assert_distributed_tensor_sync(
                        qwen_witness,
                        label="Qwen parameter witness",
                    ),
                )
                value_parameter_replica_max_difference = max(
                    value_parameter_replica_max_difference,
                    _assert_distributed_tensor_sync(
                        value_witness,
                        label="ValueHead parameter witness",
                    ),
                )

        qwen_delta = float((qwen_witness.detach() - qwen_before).abs().max().item())
        value_delta = float((value_witness.detach() - value_before).abs().max().item())
        if args.mode == "ddp_step" and value_delta <= 0.0:
            raise AssertionError(
                f"optimizer produced no ValueHead parameter change: value={value_delta}"
            )

        result: dict[str, Any] = {
            "status": "passed",
            "mode": args.mode,
            "rank": rank,
            "world_size": world_size,
            "gpus_per_rank": args.gpus_per_rank,
            "distributed_strategy": distributed.strategy,
            "trajectory_id": trajectory.record_id,
            "transition_step": args.step_index,
            "executed_action_index": transition.action_index,
            "executed_action_name": trajectory.action_names[args.step_index],
            "planner_search_mode": trajectory.planner_policy_traces[
                args.step_index
            ].search_mode,
            "old_action_value": float(old_action_value.item()),
            "return_target": float(return_target.item()),
            "ppo_epochs": epochs,
            "qwen_witness": qwen_name,
            "qwen_grad_max": qwen_grad_max,
            "qwen_parameter_delta_max": qwen_delta,
            "value_witness": value_name,
            "value_grad_max": value_grad_max,
            "value_parameter_delta_max": value_delta,
            "lm_head_witness": lm_head_name,
            "lm_head_grad_is_none": lm_head.grad is None,
            "state_projector_grads_absent": _all_grads_absent(
                world_model.state_proj
            ),
            "vision_grads_absent": _named_grads_absent(model, ".visual."),
            "sync_absolute_tolerance": SYNC_ABSOLUTE_TOLERANCE,
            "sync_relative_tolerance": SYNC_RELATIVE_TOLERANCE,
            "epoch_metrics": epoch_metrics,
            "max_memory_allocated_bytes": _gpu_memory(
                args.gpus_per_rank, local_rank
            ),
        }
        if dist.is_available() and dist.is_initialized():
            result["qwen_grad_replica_max_difference"] = (
                qwen_grad_replica_max_difference
            )
            result["value_grad_replica_max_difference"] = (
                value_grad_replica_max_difference
            )
            result["qwen_parameter_replica_max_difference"] = max(
                qwen_parameter_replica_max_difference,
                _assert_distributed_tensor_sync(
                    qwen_witness,
                    label="Qwen final parameter witness",
                ),
            )
            result["value_parameter_replica_max_difference"] = max(
                value_parameter_replica_max_difference,
                _assert_distributed_tensor_sync(
                    value_witness,
                    label="ValueHead final parameter witness",
                ),
            )
        _write_result(args.output_dir, args.mode, rank, result)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0
    finally:
        # A barrier in an exception path can hide the original failure when one
        # rank has already exited.  Successful synchronization is asserted above.
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
