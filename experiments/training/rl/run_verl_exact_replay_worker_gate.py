#!/usr/bin/env python3
"""One exact-transcript PPO update through VERL full actor/ref/critic workers."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from nimloth.training.rl.rollout import RolloutTrajectory
from nimloth.training.rl.verl_adapter import (
    build_verl_replay_dataproto,
    build_verl_replay_row_from_trajectory,
    finalize_verl_exact_replay_batch,
)
from nimloth.training.rl.verl_critic_455 import (
    install_verl_transformers455_critic_patch as _install_transformers455_critic_patch,
)
from nimloth.training.rl.verl_gate import (
    build_exact_replay_worker_config,
    configure_nimloth_wm_auxiliary,
    install_verl_zero_warmup_scheduler_patch,
)


EXPECTED_TRANSFORMERS = "4.55.4"
EXPECTED_TORCH_PREFIX = "2.8.0"
EXPECTED_VAGEN = "e00131c2555720b104d225cc8dd3b1a582f11ed6"
EXPECTED_VERL = "490a3cb557b6e8237d1b551d309e45dd7a6e0a99"


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _module_fingerprint(module) -> dict[str, float | int]:
    device = torch.device("cuda", torch.cuda.current_device())
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    for parameter in module.parameters():
        values = parameter.detach().to(device=device, dtype=torch.float64)
        totals[0] += values.sum()
        totals[1] += values.square().sum()
        totals[2] += parameter.numel()
        if parameter.requires_grad:
            totals[3] += parameter.numel()
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return {
        "sum": float(totals[0].item()),
        "sum_sq": float(totals[1].item()),
        "parameter_numel": int(totals[2].item()),
        "trainable_numel": int(totals[3].item()),
    }


def _fingerprint_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return before["sum"] != after["sum"] or before["sum_sq"] != after["sum_sq"]


def _local_optimizer_fingerprint(optimizer) -> dict[str, float | int]:
    device = torch.device("cuda", torch.cuda.current_device())
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            values = parameter.detach()
            totals[0] += values.sum(dtype=torch.float64)
            totals[1] += values.square().sum(dtype=torch.float64)
            totals[2] += parameter.numel()
            if parameter.grad is not None:
                totals[3] += parameter.grad.detach().square().sum(dtype=torch.float64)
                totals[4] += parameter.grad.numel()
    return {
        "sum": float(totals[0].item()),
        "sum_sq": float(totals[1].item()),
        "parameter_numel": int(totals[2].item()),
        "grad_sum_sq": float(totals[3].item()),
        "grad_numel": int(totals[4].item()),
    }


def _assert_tied_embeddings(module, *, role: str) -> None:
    input_weight = module.get_input_embeddings().weight
    output_weight = module.get_output_embeddings().weight
    if input_weight is not output_weight and (
        input_weight.data_ptr() != output_weight.data_ptr()
        or input_weight.storage_offset() != output_weight.storage_offset()
    ):
        raise RuntimeError(
            f"{role} lm_head is not tied to input embeddings; random/missing head forbidden"
        )


def _load_trajectory(path: Path, index: int) -> RolloutTrajectory:
    records = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 0 <= index < len(records):
        raise IndexError(f"trajectory index {index} outside JSONL size {len(records)}")
    return RolloutTrajectory.from_record(json.loads(records[index]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trajectory-jsonl", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-token-length", type=int, default=8192)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--resume-checkpoint-root", type=Path)
    parser.add_argument("--resume-result", type=Path)
    parser.add_argument("--save-global-step", type=int, default=1)
    parser.add_argument("--enable-wm-aux-mechanics", action="store_true")
    args = parser.parse_args()

    if (args.resume_checkpoint_root is None) != (args.resume_result is None):
        raise RuntimeError(
            "resume checkpoint root and source result must be provided together"
        )
    if args.save_global_step <= 0:
        raise RuntimeError("save global step must be positive")
    if args.resume_checkpoint_root is None and args.save_global_step != 1:
        raise RuntimeError("fresh exact replay gate must save global_step_1")

    if args.wandb_project != "nimloth-rl":
        raise RuntimeError(
            f"exact replay gate requires W&B project nimloth-rl, got {args.wandb_project!r}"
        )
    if os.environ.get("WANDB_PROJECT") != args.wandb_project:
        raise RuntimeError("W&B project argument/environment mismatch")
    if os.environ.get("WANDB_RUN_NAME") != args.wandb_run_name:
        raise RuntimeError("W&B run-name argument/environment mismatch")
    if os.environ.get("WANDB_RUN_ID") != args.wandb_run_id:
        raise RuntimeError("W&B run-id argument/environment mismatch")

    import transformers
    from transformers import AutoProcessor

    install_verl_zero_warmup_scheduler_patch()
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

    if transformers.__version__ != EXPECTED_TRANSFORMERS:
        raise RuntimeError(
            f"exact replay gate requires transformers={EXPECTED_TRANSFORMERS}, "
            f"got {transformers.__version__}"
        )
    if not torch.__version__.startswith(EXPECTED_TORCH_PREFIX):
        raise RuntimeError(
            f"exact replay gate requires torch {EXPECTED_TORCH_PREFIX}.x, got {torch.__version__}"
        )
    repo = args.repo.resolve()
    if _git_head(repo) != args.expected_commit:
        raise RuntimeError("Nimloth worktree commit mismatch")
    if _git_head(repo / "external/VAGEN") != EXPECTED_VAGEN:
        raise RuntimeError("VAGEN commit mismatch")
    verl_path = repo / "external/VAGEN/verl"
    if not (verl_path / "verl/__init__.py").is_file():
        shared_verl = Path(
            "/project/peilab/atst/nimloth/.worktree/vagen-legacy-wm-k8/external/VAGEN/verl"
        )
        verl_path = shared_verl
    if not (verl_path / "verl/__init__.py").is_file():
        raise RuntimeError("VERL source tree is missing or uninitialized")
    if _git_head(verl_path) != EXPECTED_VERL:
        raise RuntimeError("VERL commit mismatch")

    if not torch.cuda.is_available():
        raise RuntimeError("exact replay worker gate requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    visible_cuda_devices = torch.cuda.device_count()
    if visible_cuda_devices != world_size or not 0 <= local_rank < visible_cuda_devices:
        raise RuntimeError(
            "exact replay single-node torchrun requires every process to see all "
            f"world GPUs; device_count={visible_cuda_devices}, world_size={world_size}, "
            f"local_rank={local_rank}"
        )
    torch.cuda.set_device(local_rank)
    if world_size < 2:
        raise RuntimeError("exact replay full-worker gate requires distributed FSDP")
    process_device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=process_device)
    dist.barrier(device_ids=[local_rank])

    output = args.output_dir.resolve()
    if not output.is_dir() or not (output / "README.md").is_file():
        raise RuntimeError("output directory and preflight README.md must exist")
    for forbidden in ("result.json", "checkpoints"):
        if (output / forbidden).exists():
            raise RuntimeError(f"exact replay output already contains {forbidden}")
    model = args.model.resolve()
    trajectory_path = args.trajectory_jsonl.resolve()
    config = build_exact_replay_worker_config(
        repo / "external/VAGEN/vagen/trainer/config/ppo_trainer.yaml",
        model_path=model,
        world_size=world_size,
        max_token_length=args.max_token_length,
    )
    processor = AutoProcessor.from_pretrained(
        model, use_fast=True, trust_remote_code=False
    )
    trajectory = _load_trajectory(trajectory_path, args.trajectory_index)
    if args.enable_wm_aux_mechanics:
        configure_nimloth_wm_auxiliary(
            config,
            latent_token_count=int(trajectory.latent_token_count),
            checkpoint_dir=None,
            loss_coef=0.1,
            learning_rate=1e-5,
            allow_random_init=True,
        )
    row = build_verl_replay_row_from_trajectory(processor, trajectory)
    if row.input_ids.numel() > args.max_token_length:
        raise RuntimeError(
            f"trajectory has {row.input_ids.numel()} tokens, exceeds gate budget"
        )
    replay = build_verl_replay_dataproto(
        [row],
        pad_token_id=int(processor.tokenizer.pad_token_id),
        temperature=1.0,
        micro_batch_size=1,
    )

    actor = ActorRolloutRefWorker(
        config=copy.deepcopy(config.actor_rollout_ref), role="actor"
    )
    actor.init_model()
    _assert_tied_embeddings(actor.actor_module, role="actor")
    actor_fresh = _module_fingerprint(actor.actor_module_fsdp)
    resume_report = None
    resume_root = None
    if args.resume_checkpoint_root is not None:
        resume_root = args.resume_checkpoint_root.resolve()
        resume_result = args.resume_result.resolve()
        if not resume_root.is_dir() or not resume_result.is_file():
            raise RuntimeError("resume checkpoint/result path is missing")
        resume_report = json.loads(resume_result.read_text(encoding="utf-8"))
        if resume_report.get("status") != "VERL_EXACT_REPLAY_ALL_OK":
            raise RuntimeError("resume source result is not a successful exact replay")
        if Path(resume_report.get("checkpoint_root", "")).resolve() != resume_root:
            raise RuntimeError("resume source result/checkpoint root mismatch")
        if int(resume_report.get("world_size", -1)) != world_size:
            raise RuntimeError("resume source world size mismatch")
        try:
            source_global_step = int(resume_root.name.removeprefix("global_step_"))
        except ValueError as error:
            raise RuntimeError("resume checkpoint root must be named global_step_<n>") from error
        if args.save_global_step != source_global_step + 1:
            raise RuntimeError(
                "resumed exact replay must save the next global step: "
                f"source={source_global_step}, requested={args.save_global_step}"
            )
        for role in ("actor", "critic"):
            for prefix in ("model", "optim", "extra_state"):
                for source_rank in range(world_size):
                    source_file = (
                        resume_root
                        / role
                        / f"{prefix}_world_size_{world_size}_rank_{source_rank}.pt"
                    )
                    if not source_file.is_file():
                        raise RuntimeError(f"resume checkpoint is incomplete: {source_file}")
        actor.load_checkpoint(str(resume_root / "actor"), del_local_after_load=False)
    actor_before = _module_fingerprint(actor.actor_module_fsdp)
    wm_before = (
        _module_fingerprint(actor.wm_auxiliary_module)
        if actor.wm_auxiliary_module is not None
        else None
    )
    if resume_report is not None and actor_before != resume_report["actor_after"]:
        raise RuntimeError("resumed actor fingerprint does not match source result")
    if resume_report is not None and wm_before != resume_report.get("wm_aux_after"):
        raise RuntimeError("resumed WM auxiliary fingerprint does not match source result")
    if actor_before["trainable_numel"] != actor_before["parameter_numel"]:
        raise RuntimeError("full actor gate found frozen actor parameters")
    old_output = actor.compute_log_prob(copy.deepcopy(replay))
    actor_before_update = _module_fingerprint(actor.actor_module_fsdp)
    if actor_before_update != actor_before:
        raise RuntimeError("PPO-old recompute changed actor parameters")

    reference = ActorRolloutRefWorker(
        config=copy.deepcopy(config.actor_rollout_ref), role="ref"
    )
    reference.init_model()
    _assert_tied_embeddings(
        reference.ref_module_fsdp._fsdp_wrapped_module, role="reference"
    )
    for parameter in reference.ref_module_fsdp.parameters():
        parameter.requires_grad_(False)
    reference_before = _module_fingerprint(reference.ref_module_fsdp)
    ref_output = reference.compute_ref_log_prob(copy.deepcopy(replay))

    _install_transformers455_critic_patch()
    critic = CriticWorker(config=copy.deepcopy(config.critic))
    critic.init_model()
    critic_fresh = _module_fingerprint(critic.critic_module)
    if resume_root is not None:
        critic.load_checkpoint(str(resume_root / "critic"), del_local_after_load=False)
    critic_before = _module_fingerprint(critic.critic_module)
    if resume_report is not None and critic_before != resume_report["critic_after"]:
        raise RuntimeError("resumed critic fingerprint does not match source result")
    if critic_before["trainable_numel"] != critic_before["parameter_numel"]:
        raise RuntimeError("full critic gate found frozen critic parameters")
    values_output = critic.compute_values(copy.deepcopy(replay))

    ppo_batch, audit = finalize_verl_exact_replay_batch(
        replay,
        old_log_prob_output=old_output,
        reference_log_prob_output=ref_output,
        values_output=values_output,
        gamma=1.0,
        lam=1.0,
    )
    # VERL intentionally loads the trainable actor in fp32 and the immutable
    # reference in bf16. Judge initialization parity by the actual PPO KL
    # estimator, not bitwise log-prob equality across those precision paths.
    if audit["max_abs_old_ref_delta"] > 0.5 or audit["mean_low_var_kl"] > 0.01:
        raise RuntimeError(
            "actor/reference initialization KL is unexpectedly large: "
            f"max_delta={audit['max_abs_old_ref_delta']}, "
            f"mean_low_var_kl={audit['mean_low_var_kl']}"
        )

    critic_step_audit: list[dict[str, Any]] = []
    original_critic_step = critic.critic_optimizer.step

    def audited_critic_step(*args, **kwargs):
        before_step = _local_optimizer_fingerprint(critic.critic_optimizer)
        output = original_critic_step(*args, **kwargs)
        after_step = _local_optimizer_fingerprint(critic.critic_optimizer)
        state_steps = [
            float(state["step"].item())
            for state in critic.critic_optimizer.state.values()
            if "step" in state
        ]
        critic_step_audit.append(
            {
                "before": before_step,
                "after": after_step,
                "state_entries": len(critic.critic_optimizer.state),
                "state_step_min": min(state_steps) if state_steps else None,
                "state_step_max": max(state_steps) if state_steps else None,
            }
        )
        return output

    critic.critic_optimizer.step = audited_critic_step
    critic_metrics = critic.update_critic(copy.deepcopy(ppo_batch))
    critic.critic_optimizer.step = original_critic_step
    critic_after = _module_fingerprint(critic.critic_module)
    gathered_step_audits: list[Any] = [None] * world_size
    dist.all_gather_object(gathered_step_audits, critic_step_audit)
    for source_rank, rank_audit in enumerate(gathered_step_audits):
        if len(rank_audit) != 1:
            raise RuntimeError(
                f"rank{source_rank} expected one critic optimizer step, got {rank_audit}"
            )
        step_audit = rank_audit[0]
        if (
            step_audit["state_step_min"] != float(args.save_global_step)
            or step_audit["state_step_max"] != float(args.save_global_step)
        ):
            raise RuntimeError(
                f"rank{source_rank} critic optimizer state did not reach "
                f"step{args.save_global_step}: {step_audit}"
            )
    if rank == 0:
        print(
            "CRITIC_UPDATE_AUDIT="
            + json.dumps(
                {
                    "before": critic_before,
                    "after": critic_after,
                    "meta_info": _json_value(critic_metrics.meta_info),
                    "optimizer_steps_by_rank": gathered_step_audits,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not _fingerprint_changed(critic_before, critic_after):
        raise RuntimeError("critic optimizer update did not change parameters")

    actor_metrics = actor.update_actor(copy.deepcopy(ppo_batch))
    actor_after = _module_fingerprint(actor.actor_module_fsdp)
    wm_after = (
        _module_fingerprint(actor.wm_auxiliary_module)
        if actor.wm_auxiliary_module is not None
        else None
    )
    if not _fingerprint_changed(actor_before_update, actor_after):
        raise RuntimeError("actor optimizer update did not change parameters")
    if wm_before is not None and not _fingerprint_changed(wm_before, wm_after):
        raise RuntimeError("WM auxiliary optimizer update did not change parameters")
    post_output = actor.compute_log_prob(copy.deepcopy(replay))
    response_length = replay.batch["responses"].shape[1]
    policy_mask = replay.batch["loss_mask"][:, -response_length:].bool()
    actor_log_prob_change = float(
        (
            post_output.batch["old_log_probs"].masked_select(policy_mask)
            - old_output.batch["old_log_probs"].masked_select(policy_mask)
        ).abs().max().item()
    )
    if not math.isfinite(actor_log_prob_change) or actor_log_prob_change <= 0:
        raise RuntimeError("actor policy log-probs did not change after update")

    reference_after = _module_fingerprint(reference.ref_module_fsdp)
    if reference_after != reference_before:
        raise RuntimeError("immutable reference parameters changed")

    checkpoint_root = output / "checkpoints" / f"global_step_{args.save_global_step}"
    actor.save_checkpoint(
        str(checkpoint_root / "actor"),
        hdfs_path=None,
        global_step=args.save_global_step,
        remove_previous_ckpt=False,
    )
    critic.save_checkpoint(
        str(checkpoint_root / "critic"),
        hdfs_path=None,
        global_step=args.save_global_step,
        remove_previous_ckpt=False,
    )
    dist.barrier()

    report = {
        "status": "VERL_EXACT_REPLAY_ALL_OK",
        "commit": args.expected_commit,
        "vagen_commit": EXPECTED_VAGEN,
        "verl_commit": EXPECTED_VERL,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "world_size": world_size,
        "model": str(model),
        "trajectory_jsonl": str(trajectory_path),
        "trajectory_id": trajectory.record_id,
        "trajectory_turns": trajectory.num_steps,
        "sequence_tokens": int(row.input_ids.numel()),
        "actor_fresh_before_resume": actor_fresh,
        "actor_before": actor_before,
        "actor_after": actor_after,
        "wm_aux_before": wm_before,
        "wm_aux_after": wm_after,
        "critic_fresh_before_resume": critic_fresh,
        "critic_before": critic_before,
        "critic_after": critic_after,
        "reference_before": reference_before,
        "reference_after": reference_after,
        "actor_log_prob_change": actor_log_prob_change,
        "replay_audit": audit,
        "actor_metrics": _json_value(actor_metrics.meta_info.get("metrics", {})),
        "critic_metrics": _json_value(critic_metrics.meta_info.get("metrics", {})),
        "critic_optimizer_step_audit": gathered_step_audits,
        "checkpoint_root": str(checkpoint_root),
        "resume_checkpoint_root": str(resume_root) if resume_root is not None else None,
        "resume_source_result": str(args.resume_result.resolve()) if args.resume_result else None,
        "save_global_step": args.save_global_step,
    }
    if rank == 0:
        (output / "result.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=args.wandb_run_id,
            resume="never",
            config={
                "commit": args.expected_commit,
                "vagen_commit": EXPECTED_VAGEN,
                "verl_commit": EXPECTED_VERL,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "world_size": world_size,
                "model": str(model),
                "trajectory_id": trajectory.record_id,
                "quality_valid": False,
                "mechanics_only": True,
                "wm_aux_enabled": bool(args.enable_wm_aux_mechanics),
            },
        )
        run.log(
            {
                "global_step": args.save_global_step,
                "gate/policy_tokens": audit["policy_tokens"],
                "gate/sequence_tokens": int(row.input_ids.numel()),
                "gate/old_ref_max_delta": audit["max_abs_old_ref_delta"],
                "gate/old_ref_mean_low_var_kl": audit["mean_low_var_kl"],
                "gate/actor_log_prob_change": actor_log_prob_change,
            },
            step=args.save_global_step,
        )
        run.summary["status"] = report["status"]
        run.finish()
        print(json.dumps(report, sort_keys=True), flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
