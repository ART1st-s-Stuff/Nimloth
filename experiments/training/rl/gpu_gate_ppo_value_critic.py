"""Real-GPU integration gate for planner critic or PlannerPolicyHead PPO.

This is a mechanics gate, not a policy-quality experiment.  Ranks select from
the longest qualifying final-step prefixes in the complete set of real,
behavior-checkpoint-matched trajectories.  Distinct qualifying prefixes are
preferred, but a real long prefix is reused deterministically when there are
fewer qualifying prefixes than gate ranks.  ``single_grad`` proves the
selected loss reaches the Qwen language body without supervising ``lm_head``.
``ddp_step`` exercises the production two-rank, two-GPU-per-rank wrapping and
AdamW update for all configured PPO epochs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
from nimloth.backbone.qwen25vl.checkpoint import find_visual_module
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
    _prepare_planner_qwen_training,
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
    parser.add_argument("--planner-policy-head-checkpoint", type=Path)
    parser.add_argument("--trajectory-jsonl", type=Path, required=True)
    parser.add_argument("--fresh-rollout-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus-per-rank", type=int, required=True)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--select-longest-final-transition", action="store_true")
    parser.add_argument("--minimum-state-tokens", type=int, default=0)
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def _model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        wm_checkpoint=args.wm_checkpoint,
        state_proj_checkpoint=args.state_proj_checkpoint,
        value_head_checkpoint=args.value_head_checkpoint,
        planner_policy_head_checkpoint=args.planner_policy_head_checkpoint,
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


def _state_token_count(
    runtime: RLModelRuntime,
    transition: Any,
) -> int:
    prompt = transition.state_prompt
    batch = runtime.input_builder.build(
        [prompt.unbound_messages()],
        [prompt.images],
        include_labels=False,
    )
    input_ids = batch.tensors.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError("GPU gate state prompt did not produce one input_ids row")
    return int(input_ids.shape[-1])


@dataclass(frozen=True)
class _GateTransitionSelection:
    trajectory: RolloutTrajectory
    episode: Any
    step_index: int
    state_tokens: int
    record_index: int
    qualifying_candidate_count: int
    reused_candidate: bool


def _assigned_qualifying_candidate(
    state_tokens_by_record: list[int],
    *,
    rank: int,
    minimum_state_tokens: int,
) -> tuple[int, int, bool]:
    """Assign a real qualifying prefix, reusing only when ranks outnumber it."""

    qualifying = sorted(
        (
            (record_index, state_tokens)
            for record_index, state_tokens in enumerate(state_tokens_by_record)
            if state_tokens >= minimum_state_tokens
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not qualifying:
        maximum = max(state_tokens_by_record, default=0)
        raise RuntimeError(
            "GPU gate has no real final prefix satisfying its memory contract: "
            f"maximum_tokens={maximum}, minimum={minimum_state_tokens}"
        )
    assigned_position = rank % len(qualifying)
    return (
        qualifying[assigned_position][0],
        len(qualifying),
        rank >= len(qualifying),
    )


def _select_longest_final_transition(
    path: Path,
    *,
    runtime: RLModelRuntime,
    rank: int,
    world_size: int,
    gamma: float,
    truncated_bootstrap: float | None,
    minimum_state_tokens: int,
) -> _GateTransitionSelection:
    """Choose a qualifying real final prefix from the global gate sample pool."""

    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world_size {world_size}")
    candidates: list[tuple[RolloutTrajectory, Any, int, int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            trajectory = RolloutTrajectory.from_record(json.loads(line))
            validate_rollout_trajectory(trajectory)
            episode = build_episode_training_batches(
                (trajectory,),
                gamma=gamma,
                truncated_bootstrap=truncated_bootstrap,
            )[0]
            step_index = len(episode.transitions) - 1
            state_tokens = _state_token_count(
                runtime,
                episode.transitions[step_index],
            )
            candidates.append((trajectory, episode, step_index, state_tokens))
    selected_index, qualifying_count, reused_candidate = (
        _assigned_qualifying_candidate(
            [candidate[-1] for candidate in candidates],
            rank=rank,
            minimum_state_tokens=minimum_state_tokens,
        )
    )
    trajectory, episode, step_index, state_tokens = candidates[selected_index]
    return _GateTransitionSelection(
        trajectory=trajectory,
        episode=episode,
        step_index=step_index,
        state_tokens=state_tokens,
        record_index=selected_index,
        qualifying_candidate_count=qualifying_count,
        reused_candidate=reused_candidate,
    )


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
    if (
        config.planner_policy.enabled
        and args.planner_policy_head_checkpoint is None
    ):
        raise ValueError(
            "PlannerPolicyHead GPU gate requires --planner-policy-head-checkpoint"
        )
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
                **(
                    {
                        "planner_policy_head": (
                            args.planner_policy_head_checkpoint
                        )
                    }
                    if args.planner_policy_head_checkpoint is not None
                    else {}
                ),
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
        checkpointed_modules = _prepare_planner_qwen_training(
            model,
            gradient_checkpointing=bool(loading_args.gradient_checkpointing),
            eval_modules=(find_visual_module(model),),
        )
        if world_size > 1:
            broadcast_module_state(world_model.state_proj)
            broadcast_module_state(world_model.wm_predictor)
            broadcast_module_state(world_model.value_head)
            if world_model.planner_policy_head is not None:
                broadcast_module_state(world_model.planner_policy_head)
        distributed = _wrap_distributed_modules(
            loaded.backbone,
            world_model,
            None,
            world_size=world_size,
            model_parallel=loaded.pair_parallel,
            synchronize_backbone_hidden=True,
            training_device=training_device,
        )
        model = distributed.model
        world_model = distributed.world_model
        agent = Agent(backbone=distributed.backbone, wm=world_model)
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
            planner_policy_enabled=config.planner_policy.enabled,
            planner_policy_clip_ratio=config.planner_policy.clip_ratio,
            planner_policy_entropy_weight=config.planner_policy.entropy_coeff,
            planner_policy_temperature=config.planner_policy.temperature,
        )

        truncated_bootstrap = (
            0.0 if config.rl.truncated_bootstrap == "zero" else None
        )
        if args.select_longest_final_transition:
            selection = _select_longest_final_transition(
                args.trajectory_jsonl,
                runtime=runtime,
                rank=rank,
                world_size=world_size,
                gamma=config.rl.gamma,
                truncated_bootstrap=truncated_bootstrap,
                minimum_state_tokens=args.minimum_state_tokens,
            )
            trajectory = selection.trajectory
            episode = selection.episode
            step_index = selection.step_index
            state_tokens = selection.state_tokens
            selection_record_index = selection.record_index
            selection_qualifying_candidate_count = (
                selection.qualifying_candidate_count
            )
            selection_reused_candidate = selection.reused_candidate
            selection_policy = "global-qualified-longest-final"
        else:
            trajectory = _load_trajectory(args.trajectory_jsonl, rank)
            episode = build_episode_training_batches(
                (trajectory,),
                gamma=config.rl.gamma,
                truncated_bootstrap=truncated_bootstrap,
            )[0]
            step_index = args.step_index
            transition = episode.transitions[step_index]
            state_tokens = _state_token_count(runtime, transition)
            selection_record_index = rank
            selection_qualifying_candidate_count = 1
            selection_reused_candidate = False
            selection_policy = "explicit-record-step"
        if state_tokens < args.minimum_state_tokens:
            raise RuntimeError(
                "GPU gate state prefix is shorter than its memory contract: "
                f"tokens={state_tokens}, minimum={args.minimum_state_tokens}"
            )
        transition = episode.transitions[step_index]
        return_target = episode.return_targets[step_index]
        old_policy_log_prob = None
        policy_advantage = None
        if config.planner_policy.enabled:
            old_statistics = algorithm.planner_old_policy_statistics(
                runtime,
                transition,
            )
            old_action_value = old_statistics.selected_action_value
            old_policy_log_prob = old_statistics.selected_log_prob
            policy_advantage = return_target - old_statistics.state_value
        else:
            old_action_value = algorithm.planner_old_action_value(
                runtime,
                transition,
            )

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
        policy_name = None
        policy_witness = None
        policy_before = None
        if world_model.planner_policy_head is not None:
            policy_name, policy_witness = next(
                (name, parameter)
                for name, parameter in world_model.planner_policy_head.named_parameters()
                if parameter.requires_grad
            )
            policy_before = policy_witness.detach().clone()

        optimizer = None
        if args.mode == "ddp_step":
            optimizer = _build_optimizer(model, world_model, None, config)

        epoch_metrics: list[dict[str, float]] = []
        qwen_grad_max = 0.0
        value_grad_max = 0.0
        policy_grad_max = 0.0
        qwen_grad_replica_max_difference = 0.0
        value_grad_replica_max_difference = 0.0
        qwen_parameter_replica_max_difference = 0.0
        value_parameter_replica_max_difference = 0.0
        policy_grad_replica_max_difference = 0.0
        policy_parameter_replica_max_difference = 0.0
        configured_epochs = (
            config.planner_policy.ppo_epochs
            if config.planner_policy.enabled
            else config.value_head.ppo_epochs
        )
        epochs = 1 if args.mode == "single_grad" else configured_epochs
        for ppo_epoch in range(epochs):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)
                world_model.zero_grad(set_to_none=True)
            output = algorithm.actor_transition_step(
                runtime,
                transition,
                return_target=return_target,
                old_action_value=old_action_value,
                old_policy_log_prob=old_policy_log_prob,
                policy_advantage=policy_advantage,
                total_transitions=world_size,
                include_world_model=False,
            )
            # Production planner sharding cancels DDP's gradient average this way.
            (output.loss * world_size).backward()
            qwen_grad_max = max(qwen_grad_max, _grad_max(qwen_witness))
            value_grad_max = max(value_grad_max, _grad_max(value_witness))
            if policy_witness is not None:
                policy_grad_max = max(policy_grad_max, _grad_max(policy_witness))
            if qwen_grad_max <= 0.0:
                raise AssertionError("critic loss did not reach the Qwen language body")
            if value_grad_max <= 0.0:
                raise AssertionError("critic loss did not reach ValueHead")
            if policy_witness is not None and policy_grad_max <= 0.0:
                raise AssertionError(
                    "policy loss did not reach PlannerPolicyHead"
                )
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
                if policy_witness is not None:
                    assert policy_witness.grad is not None
                    policy_grad_replica_max_difference = max(
                        policy_grad_replica_max_difference,
                        _assert_distributed_tensor_sync(
                            policy_witness.grad,
                            label="PlannerPolicyHead gradient witness",
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
                if policy_witness is not None:
                    policy_parameter_replica_max_difference = max(
                        policy_parameter_replica_max_difference,
                        _assert_distributed_tensor_sync(
                            policy_witness,
                            label="PlannerPolicyHead parameter witness",
                        ),
                    )

        qwen_delta = float((qwen_witness.detach() - qwen_before).abs().max().item())
        value_delta = float((value_witness.detach() - value_before).abs().max().item())
        policy_delta = (
            float((policy_witness.detach() - policy_before).abs().max().item())
            if policy_witness is not None and policy_before is not None
            else 0.0
        )
        if args.mode == "ddp_step" and value_delta <= 0.0:
            raise AssertionError(
                f"optimizer produced no ValueHead parameter change: value={value_delta}"
            )
        if (
            args.mode == "ddp_step"
            and policy_witness is not None
            and policy_delta <= 0.0
        ):
            raise AssertionError(
                "optimizer produced no PlannerPolicyHead parameter change"
            )

        result: dict[str, Any] = {
            "status": "passed",
            "mode": args.mode,
            "rank": rank,
            "world_size": world_size,
            "gpus_per_rank": args.gpus_per_rank,
            "distributed_strategy": distributed.strategy,
            "gradient_checkpointing_active_modules": checkpointed_modules,
            "trajectory_id": trajectory.record_id,
            "selection_policy": selection_policy,
            "selection_record_index": selection_record_index,
            "selection_qualifying_candidate_count": (
                selection_qualifying_candidate_count
            ),
            "selection_reused_candidate": selection_reused_candidate,
            "transition_step": step_index,
            "state_tokens": state_tokens,
            "executed_action_index": transition.action_index,
            "executed_action_name": trajectory.action_names[step_index],
            "planner_search_mode": trajectory.planner_policy_traces[
                step_index
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
            "planner_policy_witness": policy_name,
            "planner_policy_grad_max": policy_grad_max,
            "planner_policy_parameter_delta_max": policy_delta,
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
            if policy_witness is not None:
                result["planner_policy_grad_replica_max_difference"] = (
                    policy_grad_replica_max_difference
                )
                result["planner_policy_parameter_replica_max_difference"] = max(
                    policy_parameter_replica_max_difference,
                    _assert_distributed_tensor_sync(
                        policy_witness,
                        label="PlannerPolicyHead final parameter witness",
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
