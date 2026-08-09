"""Real ID147/ID149 complete-objective gate through one Ray/FSDP root.

This diagnostic never consumes the source rollout and never publishes a policy
checkpoint. It verifies one behavior-matched real transition reaches Qwen, WM,
DINO, ValueHead, and PlannerPolicyHead together through the production FSDP
factory. ID151 separately proves sharded checkpoint/reload mechanics.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time
from tempfile import NamedTemporaryFile
from typing import Any

import torch
from transformers import AutoConfig

from nimloth.backbone.dino_grid import (
    DINOV2_LARGE_IDENTITY,
    FrozenDINOGridTargets,
)
from nimloth.backbone.qwen25vl.input import Qwen25VLInputBuilder
from nimloth.backbone.qwen25vl.loading import load_qwen_processor
from nimloth.backbone.qwen25vl.policy import validate_agent_policy_protocol
from nimloth.config.rl import load_rl_config
from nimloth.rollout import (
    FreshJSONLRolloutCollector,
    FreshRolloutManifest,
)
from nimloth.rollout.fresh import (
    auxiliary_artifact_fingerprint,
    file_artifact_fingerprint,
    policy_artifact_fingerprint,
)
from nimloth.training.rl.episodes import build_episode_training_batches
from nimloth.training.rl.planner_verl_batch import (
    build_replicated_planner_gate_round,
    load_planner_behavior_heads,
    prepare_planner_behavior_row,
    prepare_planner_nonbehavior_diagnostic_row,
)
from nimloth.training.rl.planner_verl_worker import PlannerVERLFSDPWorker


_TRAINABLE_COMPONENTS = (
    "qwen_language",
    "wm_predictor",
    "value_head",
    "planner_policy_head",
)
_FROZEN_COMPONENTS = ("state_projector", "vision", "lm_head")
_REQUIRED_METRICS = (
    "wm_mse",
    "dino_grid_mse",
    "value_loss",
    "planner_policy_loss",
    "planner_policy_entropy",
    "total_loss",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wm-checkpoint", type=Path, required=True)
    parser.add_argument("--state-proj-checkpoint", type=Path, required=True)
    parser.add_argument("--value-head-checkpoint", type=Path, required=True)
    parser.add_argument("--planner-policy-head-checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory-jsonl", type=Path, required=True)
    parser.add_argument("--fresh-rollout-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--minimum-state-tokens", type=int, default=6000)
    parser.add_argument("--diagnostic-nonbehavior-prefix", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-trajectory-count", type=int)
    parser.add_argument("--expected-selected-record-id")
    parser.add_argument("--expected-selected-state-tokens", type=int)
    parser.add_argument("--expected-current-policy-sha256")
    parser.add_argument("--expected-current-state-projector-sha256")
    parser.add_argument("--expected-current-wm-predictor-sha256")
    parser.add_argument("--expected-current-value-head-sha256")
    parser.add_argument("--expected-current-planner-policy-head-sha256")
    parser.add_argument("--wandb-project", default="nimloth-rl")
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    return parser.parse_args()


def _state_token_count(
    input_builder: Qwen25VLInputBuilder,
    transition: Any,
) -> int:
    prompt = transition.state_prompt
    batch = input_builder.build(
        [prompt.unbound_messages()],
        [prompt.images],
        include_labels=False,
    )
    input_ids = batch.tensors.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError("complete-objective gate did not build one token row")
    return int(input_ids.shape[-1])


def _source_policy_provenance(
    manifest: FreshRolloutManifest,
) -> dict[str, Any]:
    actual_policy = policy_artifact_fingerprint(Path(manifest.policy_path))
    if actual_policy != manifest.policy_fingerprint:
        raise ValueError("diagnostic source policy fingerprint mismatch")
    actual_planner = {
        name: auxiliary_artifact_fingerprint(Path(manifest.planner_paths[name]))
        for name in sorted(manifest.planner_fingerprints)
    }
    if actual_planner != {
        name: manifest.planner_fingerprints[name]
        for name in sorted(manifest.planner_fingerprints)
    }:
        raise ValueError("diagnostic source planner fingerprint mismatch")
    return {
        "policy_path": manifest.policy_path,
        "policy_fingerprint": actual_policy,
        "planner_paths": {
            name: manifest.planner_paths[name] for name in sorted(actual_planner)
        },
        "planner_fingerprints": actual_planner,
    }


def _current_objective_provenance(args: argparse.Namespace) -> dict[str, Any]:
    provenance = {
        "policy_path": str(args.model.resolve()),
        "policy_fingerprint": policy_artifact_fingerprint(args.model),
        "planner_paths": {
            "state_projector": str(args.state_proj_checkpoint.resolve()),
            "wm_predictor": str(args.wm_checkpoint.resolve()),
            "value_head": str(args.value_head_checkpoint.resolve()),
            "planner_policy_head": str(
                args.planner_policy_head_checkpoint.resolve()
            ),
        },
        "planner_fingerprints": {
            "state_projector": auxiliary_artifact_fingerprint(
                args.state_proj_checkpoint
            ),
            "wm_predictor": auxiliary_artifact_fingerprint(args.wm_checkpoint),
            "value_head": auxiliary_artifact_fingerprint(
                args.value_head_checkpoint
            ),
            "planner_policy_head": auxiliary_artifact_fingerprint(
                args.planner_policy_head_checkpoint
            ),
        },
    }
    if args.diagnostic_nonbehavior_prefix:
        expected = {
            "policy_fingerprint": args.expected_current_policy_sha256,
            "planner_fingerprints": {
                "state_projector": (
                    args.expected_current_state_projector_sha256
                ),
                "wm_predictor": args.expected_current_wm_predictor_sha256,
                "value_head": args.expected_current_value_head_sha256,
                "planner_policy_head": (
                    args.expected_current_planner_policy_head_sha256
                ),
            },
        }
        if any(
            value in (None, "")
            for value in (
                expected["policy_fingerprint"],
                *expected["planner_fingerprints"].values(),
            )
        ):
            raise ValueError(
                "nonbehavior diagnostic requires expected current artifact hashes"
            )
        if provenance["policy_fingerprint"] != expected["policy_fingerprint"]:
            raise ValueError("current policy fingerprint differs from contract")
        if provenance["planner_fingerprints"] != expected["planner_fingerprints"]:
            raise ValueError(
                "current planner artifact fingerprints differ from contract"
            )
    return provenance


def _revalidate_gate_source(
    args: argparse.Namespace,
    source_metadata: dict[str, Any],
) -> None:
    manifest = FreshRolloutManifest.read(args.fresh_rollout_manifest)
    manifest.validate_trajectory_artifacts()
    if file_artifact_fingerprint(args.fresh_rollout_manifest) != source_metadata[
        "manifest_fingerprint"
    ]:
        raise RuntimeError("gate source manifest changed during execution")
    if manifest.trajectory_fingerprint != source_metadata["trajectory_fingerprint"]:
        raise RuntimeError("gate source trajectory changed during execution")
    if args.fresh_rollout_manifest.with_suffix(
        args.fresh_rollout_manifest.suffix + ".consumption.json"
    ).exists():
        raise RuntimeError("gate source was consumed during diagnostic execution")
    if _source_policy_provenance(manifest) != source_metadata[
        "source_policy_provenance"
    ]:
        raise RuntimeError("gate source policy artifacts changed during execution")
    if _current_objective_provenance(args) != source_metadata[
        "current_objective_provenance"
    ]:
        raise RuntimeError("gate objective artifacts changed during execution")


def _prepare_gate_row(args: argparse.Namespace) -> tuple[Any, int, torch.Tensor, dict[str, Any]]:
    config = load_rl_config(args.config)
    if config.distributed.world_size != args.world_size:
        raise ValueError("gate world size differs from RL config")
    if config.distributed.gpus_per_rank != 1:
        raise ValueError("complete-objective FSDP gate requires one GPU per rank")
    if config.planner_policy.ppo_epochs != 1:
        raise ValueError("complete-objective gate requires one optimizer epoch")
    if not config.predictor.train_wm or config.predictor.lambda_dino <= 0.0:
        raise ValueError("complete-objective gate requires both WM and DINO losses")

    manifest = FreshRolloutManifest.read(args.fresh_rollout_manifest)
    if Path(manifest.trajectory_path).resolve() != args.trajectory_jsonl.resolve():
        raise ValueError("manifest trajectory path differs from gate trajectory")
    manifest_fingerprint = file_artifact_fingerprint(args.fresh_rollout_manifest)
    if args.diagnostic_nonbehavior_prefix:
        expected = {
            "manifest_sha256": args.expected_manifest_sha256,
            "trajectory_count": args.expected_trajectory_count,
            "selected_record_id": args.expected_selected_record_id,
            "selected_state_tokens": args.expected_selected_state_tokens,
            "current_policy_sha256": args.expected_current_policy_sha256,
            "current_state_projector_sha256": (
                args.expected_current_state_projector_sha256
            ),
            "current_wm_predictor_sha256": (
                args.expected_current_wm_predictor_sha256
            ),
            "current_value_head_sha256": args.expected_current_value_head_sha256,
            "current_planner_policy_head_sha256": (
                args.expected_current_planner_policy_head_sha256
            ),
        }
        if any(value in (None, "") for value in expected.values()):
            raise ValueError(
                "nonbehavior diagnostic requires every explicit source expectation"
            )
        if manifest_fingerprint != args.expected_manifest_sha256:
            raise ValueError("diagnostic manifest SHA256 differs from contract")
        if manifest.num_trajectories != args.expected_trajectory_count:
            raise ValueError("diagnostic trajectory count differs from contract")
        freshness = FreshJSONLRolloutCollector(
            args.fresh_rollout_manifest,
            model_path=Path(manifest.policy_path),
            planner_artifacts={
                name: Path(manifest.planner_paths[name])
                for name in manifest.planner_fingerprints
            },
        )
    else:
        freshness = FreshJSONLRolloutCollector(
            args.fresh_rollout_manifest,
            model_path=args.model,
            planner_artifacts={
                "state_projector": args.state_proj_checkpoint,
                "wm_predictor": args.wm_checkpoint,
                "value_head": args.value_head_checkpoint,
                "planner_policy_head": args.planner_policy_head_checkpoint,
            },
        )
    freshness.validate_policy()
    if freshness.consumption_path.exists():
        raise RuntimeError(
            "complete-objective diagnostic requires an unconsumed source manifest"
        )
    trajectories = freshness.collect(
        num_episodes=manifest.num_trajectories,
        max_steps_per_episode=config.rl.max_steps_per_episode,
    )

    latent_token_count = validate_agent_policy_protocol(
        AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    )
    processor_bundle = load_qwen_processor(
        args.model,
        max_pixels=None,
        latent_token_count=latent_token_count,
    )
    manifest.validate_processor(processor_bundle.processor)
    input_builder = Qwen25VLInputBuilder(
        processor=processor_bundle.processor,
        max_length=999_999,
        latent_token_count=latent_token_count,
        mask_latent_query_labels=True,
    )

    truncated_bootstrap = 0.0 if config.rl.truncated_bootstrap == "zero" else None
    candidates: list[tuple[Any, torch.Tensor, int]] = []
    for trajectory in trajectories:
        episode = build_episode_training_batches(
            (trajectory,),
            gamma=config.rl.gamma,
            truncated_bootstrap=truncated_bootstrap,
        )[0]
        transition = episode.transitions[-1]
        candidates.append(
            (
                transition,
                episode.return_targets[-1],
                _state_token_count(input_builder, transition),
            )
        )
    qualifying = sorted(
        (candidate for candidate in candidates if candidate[2] >= args.minimum_state_tokens),
        key=lambda candidate: (-candidate[2], candidate[0].trajectory.record_id),
    )
    if not qualifying:
        maximum = max((candidate[2] for candidate in candidates), default=0)
        raise RuntimeError(
            "complete-objective gate has no real prefix meeting its contract: "
            f"maximum={maximum}, minimum={args.minimum_state_tokens}"
        )
    transition, return_target, token_count = qualifying[0]
    if args.diagnostic_nonbehavior_prefix and (
        transition.trajectory.record_id != args.expected_selected_record_id
        or token_count != args.expected_selected_state_tokens
    ):
        raise RuntimeError(
            "selected long-prefix record/tokens differ from explicit contract: "
            f"record={transition.trajectory.record_id}, tokens={token_count}"
        )
    if token_count > config.actor.max_state_tokens:
        raise RuntimeError(
            "selected real prefix exceeds actor.max_state_tokens: "
            f"tokens={token_count}, maximum={config.actor.max_state_tokens}"
        )

    value_head, policy_head = load_planner_behavior_heads(
        value_head_checkpoint=args.value_head_checkpoint,
        planner_policy_head_checkpoint=args.planner_policy_head_checkpoint,
        emb_dim=config.predictor.emb_dim,
    )
    prepare_row = (
        prepare_planner_nonbehavior_diagnostic_row
        if args.diagnostic_nonbehavior_prefix
        else prepare_planner_behavior_row
    )
    row = prepare_row(
        transition,
        return_target=return_target,
        value_head=value_head,
        planner_policy_head=policy_head,
        temperature=config.planner_policy.temperature,
    )
    del value_head, policy_head

    dino_source = FrozenDINOGridTargets.from_pretrained(
        DINOV2_LARGE_IDENTITY,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )
    dino_target = dino_source.load(
        (transition.next_image_path,),
        device=torch.device("cpu"),
    )[0].unsqueeze(0)
    del dino_source
    gc.collect()
    metadata = {
        "trajectory_id": transition.trajectory.record_id,
        "behavior_matched": not args.diagnostic_nonbehavior_prefix,
        "source_policy_path": manifest.policy_path,
        "diagnostic_only": bool(args.diagnostic_nonbehavior_prefix),
        "manifest_path": str(args.fresh_rollout_manifest.resolve()),
        "manifest_fingerprint": manifest_fingerprint,
        "trajectory_fingerprint": manifest.trajectory_fingerprint,
        "trajectory_count": manifest.num_trajectories,
        "source_policy_provenance": _source_policy_provenance(manifest),
        "current_objective_provenance": _current_objective_provenance(args),
        "transition_step": transition.step_index,
        "executed_action_index": transition.action_index,
        "state_tokens": token_count,
        "minimum_state_tokens": args.minimum_state_tokens,
        "candidate_final_state_tokens": {
            candidate[0].trajectory.record_id: candidate[2] for candidate in candidates
        },
        **(
            {
                "current_head_action_value": float(row.old_action_value.item()),
                "current_head_policy_log_prob": float(
                    row.old_policy_log_prob.item()
                ),
                "current_head_diagnostic_advantage": float(
                    row.policy_advantage.item()
                ),
            }
            if args.diagnostic_nonbehavior_prefix
            else {
                "old_action_value": float(row.old_action_value.item()),
                "old_policy_log_prob": float(row.old_policy_log_prob.item()),
                "policy_advantage": float(row.policy_advantage.item()),
            }
        ),
        "return_target": float(row.return_target.item()),
        "dino_identity": {
            "source": DINOV2_LARGE_IDENTITY.source,
            "revision": DINOV2_LARGE_IDENTITY.revision,
            "processor_fingerprint": DINOV2_LARGE_IDENTITY.processor_fingerprint,
            "gate_device": "cpu",
            "gate_dtype": "float32",
        },
    }
    if freshness.consumption_path.exists():
        raise RuntimeError("diagnostic unexpectedly created a consumption sidecar")
    _revalidate_gate_source(args, metadata)
    return row, token_count, dino_target, metadata


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _update_finished_wandb_status(
    *,
    entity: str,
    project: str,
    run_id: str,
    status: str,
) -> None:
    import wandb

    last_error: Exception | None = None
    for _ in range(6):
        try:
            remote = wandb.Api().run(f"{entity}/{project}/{run_id}")
            remote.summary["gate/status"] = status
            remote.update()
            confirmed = wandb.Api().run(f"{entity}/{project}/{run_id}")
            if confirmed.summary.get("gate/status") != status:
                raise RuntimeError("W&B summary status update did not persist")
            return
        except Exception as error:  # pragma: no cover - network retry
            last_error = error
            time.sleep(5)
    raise RuntimeError("could not finalize W&B gate status") from last_error


def _validate_worker_placement(rows: list[dict[str, Any]], world_size: int) -> None:
    if len(rows) != world_size:
        raise RuntimeError("complete-objective gate did not initialize every rank")
    if sorted(int(row["rank"]) for row in rows) != list(range(world_size)):
        raise RuntimeError("complete-objective gate rank identities are invalid")
    if len({row["hostname"] for row in rows}) != 1:
        raise RuntimeError("complete-objective workers span multiple Slurm nodes")
    if len({row["ray_node_id"] for row in rows}) != 1:
        raise RuntimeError("complete-objective workers span multiple Ray nodes")
    if len({row["cuda_device_uuid"] for row in rows}) != world_size:
        raise RuntimeError("complete-objective workers do not own distinct GPUs")


def _replicated_dict(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if len(rows) < 2 or any(row != rows[0] for row in rows[1:]):
        raise RuntimeError(f"{label} differs across FSDP ranks")
    return rows[0]


def _finite_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if len(rows) < 2:
        raise RuntimeError("complete-objective metrics require distributed ranks")
    metric_names = tuple(sorted(rows[0]))
    if any(tuple(sorted(row)) != metric_names for row in rows[1:]):
        raise RuntimeError("complete-objective metric keys differ across ranks")
    missing = sorted(set(_REQUIRED_METRICS) - set(metric_names))
    if missing:
        raise RuntimeError(
            "complete-objective metrics are missing: " + ", ".join(missing)
        )
    if any(
        not math.isfinite(float(row[name]))
        for row in rows
        for name in _REQUIRED_METRICS
    ):
        raise RuntimeError("complete-objective metrics are non-finite")
    averaged = {
        name: sum(float(row[name]) for row in rows) / len(rows)
        for name in metric_names
    }
    if averaged["wm_mse"] <= 0.0 or averaged["dino_grid_mse"] <= 0.0:
        raise RuntimeError("complete-objective WM/DINO losses must be positive")
    return averaged


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    import ray
    from verl.single_controller.ray import (
        RayClassWithInitArgs,
        RayResourcePool,
        RayWorkerGroup,
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != 1:
        raise RuntimeError("complete-objective gate requires one Slurm node")
    if torch.cuda.device_count() != args.world_size:
        raise RuntimeError("driver-visible GPU count differs from gate world size")

    row, token_count, dino_target, source_metadata = _prepare_gate_row(args)
    rank_rounds = build_replicated_planner_gate_round(
        row,
        token_count=token_count,
        dino_grid_target=dino_target,
        world_size=args.world_size,
        provisional_update_id="complete-objective-gate",
        allow_nonbehavior_diagnostic=args.diagnostic_nonbehavior_prefix,
    )
    worker_config = {
        "model_factory": (
            "nimloth.training.rl.planner_verl_factory:"
            "build_planner_worker_components"
        ),
        "rl_config_path": str(args.config.resolve()),
        "trainer_args": {
            "model": str(args.model.resolve()),
            "max_pixels": None,
            "attn_implementation": "sdpa",
            "llm_tune": "full",
            "vision_tune": "freeze",
            "gradient_checkpointing": True,
            "wm_checkpoint": str(args.wm_checkpoint.resolve()),
            "state_proj_checkpoint": str(args.state_proj_checkpoint.resolve()),
            "value_head_checkpoint": str(args.value_head_checkpoint.resolve()),
            "planner_policy_head_checkpoint": str(
                args.planner_policy_head_checkpoint.resolve()
            ),
            "resume": False,
            "resume_checkpoint": None,
        },
    }

    ray.init(
        num_cpus=max(8, args.world_size * 4),
        num_gpus=args.world_size,
        include_dashboard=False,
    )
    try:
        remote_worker = ray.remote(PlannerVERLFSDPWorker)
        resource_pool = RayResourcePool(
            process_on_nodes=[args.world_size],
            use_gpu=True,
            name_prefix="planner-complete-objective-",
            max_colocate_count=1,
        )
        workers = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=RayClassWithInitArgs(remote_worker, worker_config),
            name_prefix="planner_complete_objective",
        )
        init_rows = workers.init_model()
        _validate_worker_placement(init_rows, args.world_size)
        before = _replicated_dict(
            workers.planner_component_fingerprints(),
            "initial component fingerprints",
        )
        update_id = "complete-objective-gate"
        if workers.begin_planner_update(update_id) != [True] * args.world_size:
            raise RuntimeError("not every worker began the complete-objective gate")
        backward_metrics = workers.backward_planner_micro_batch(list(rank_rounds[0]))
        metrics = _finite_metrics(backward_metrics)
        gradient_norms = _replicated_dict(
            workers.planner_component_gradient_norms(),
            "component gradient norms",
        )
        for component in _TRAINABLE_COMPONENTS:
            value = float(gradient_norms[component])
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError(
                    f"complete objective did not reach {component}: norm={value}"
                )
        for component in _FROZEN_COMPONENTS:
            if float(gradient_norms[component]) != 0.0:
                raise RuntimeError(
                    f"frozen component {component} received a gradient"
                )
        finish_metrics = workers.finish_planner_update(update_id)
        if len(finish_metrics) != args.world_size:
            raise RuntimeError("complete-objective optimizer did not finish every rank")
        after = _replicated_dict(
            workers.planner_component_fingerprints(),
            "updated component fingerprints",
        )
        for component in _TRAINABLE_COMPONENTS:
            if before[component] == after[component]:
                raise RuntimeError(
                    f"optimizer did not change trainable component {component}"
                )
        for component in _FROZEN_COMPONENTS:
            if before[component] != after[component]:
                raise RuntimeError(
                    f"optimizer changed frozen component {component}"
                )
        peak_memory = workers.planner_peak_memory_allocated()
        if len(peak_memory) != args.world_size or any(
            int(value) <= 0 for value in peak_memory
        ):
            raise RuntimeError("complete-objective peak-memory rows are invalid")
        _revalidate_gate_source(args, source_metadata)
        return {
            "objective_status": "passed",
            "world_size": args.world_size,
            "source": source_metadata,
            "init_rows": init_rows,
            "metrics": metrics,
            "gradient_norms": gradient_norms,
            "component_fingerprints_before": before,
            "component_fingerprints_after": after,
            "peak_memory_allocated_bytes": peak_memory,
            "optimizer_epochs": 1,
            "source_rollout_consumed": False,
            "policy_checkpoint_published": False,
        }
    finally:
        ray.shutdown()


def main() -> int:
    args = _parse_args()
    if args.world_size < 2:
        raise ValueError("complete-objective Ray/FSDP gate requires at least two GPUs")
    if args.minimum_state_tokens < 1:
        raise ValueError("minimum state tokens must be positive")
    if (
        args.diagnostic_nonbehavior_prefix
        and args.minimum_state_tokens < 14_000
    ):
        raise ValueError(
            "nonbehavior long-prefix diagnostic requires at least 14000 tokens"
        )

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume="never",
        config={
            "gate": (
                "planner_verl_complete_objective_nonbehavior_long_prefix"
                if args.diagnostic_nonbehavior_prefix
                else "planner_verl_complete_objective_id147_id149"
            ),
            "world_size": args.world_size,
            "minimum_state_tokens": args.minimum_state_tokens,
            "diagnostic_nonbehavior_prefix": args.diagnostic_nonbehavior_prefix,
        },
    )
    output = args.output_dir.resolve()
    wandb_finished = False
    wandb_entity = str(run.entity)
    try:
        result = run_gate(args)
        run.log(
            {
                "gate/status": 1,
                "gate/state_tokens": result["source"]["state_tokens"],
                "gate/peak_memory_max_bytes": max(
                    result["peak_memory_allocated_bytes"]
                ),
                **{
                    f"gate/grad_norm_{name}": value
                    for name, value in result["gradient_norms"].items()
                },
            }
        )
        payload = {
            "status": "COMPUTE_OK_PENDING_FINAL_PROVENANCE",
            "wandb_project": args.wandb_project,
            "wandb_run_name": args.wandb_run_name,
            "wandb_run_id": args.wandb_run_id,
            **result,
        }
        _write_json_atomic(output / "result.json", payload)
        _revalidate_gate_source(args, result["source"])
        run.summary["gate/status"] = "COMPUTE_OK_PENDING_FINAL_PROVENANCE"
        run.finish(exit_code=0)
        wandb_finished = True
        _revalidate_gate_source(args, result["source"])
        _update_finished_wandb_status(
            entity=wandb_entity,
            project=args.wandb_project,
            run_id=args.wandb_run_id,
            status="ALL_OK",
        )
        payload["status"] = "ALL_OK"
        _write_json_atomic(output / "result.json", payload)
        return 0
    except Exception as error:
        try:
            if wandb_finished:
                _update_finished_wandb_status(
                    entity=wandb_entity,
                    project=args.wandb_project,
                    run_id=args.wandb_run_id,
                    status="FAILED_FINALIZATION",
                )
            else:
                run.summary["gate/status"] = "FAILED"
                run.finish(exit_code=1)
        finally:
            _write_json_atomic(
                output / "terminal.json",
                {
                    "status": "FAILED",
                    "wandb_project": args.wandb_project,
                    "wandb_run_name": args.wandb_run_name,
                    "wandb_run_id": args.wandb_run_id,
                    "error": repr(error),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
