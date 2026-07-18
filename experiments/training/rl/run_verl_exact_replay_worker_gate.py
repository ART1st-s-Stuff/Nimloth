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
from nimloth.training.rl.verl_gate import build_exact_replay_worker_config


EXPECTED_TRANSFORMERS = "4.55.4"
EXPECTED_TORCH_PREFIX = "2.8.0"
EXPECTED_VAGEN = "e7cc2d01584abcab1e49ba4a6b18ba2067fb6762"
EXPECTED_VERL = "65316156d1011d71d62e0542e4b954f9499e872e"


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
    args = parser.parse_args()

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
    if visible_cuda_devices != 1 or local_rank != 0:
        raise RuntimeError(
            "exact replay Slurm tasks require one remapped CUDA device at ordinal0; "
            f"device_count={visible_cuda_devices}, local_rank={local_rank}"
        )
    torch.cuda.set_device(0)
    if world_size < 2:
        raise RuntimeError("exact replay full-worker gate requires distributed FSDP")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda:0"))
    dist.barrier(device_ids=[0])

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
    actor_before = _module_fingerprint(actor.actor_module_fsdp)
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

    critic = CriticWorker(config=copy.deepcopy(config.critic))
    critic.init_model()
    critic_before = _module_fingerprint(critic.critic_module)
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
    if audit["max_abs_old_ref_delta"] > 1e-4:
        raise RuntimeError(
            "actor/reference initialization mismatch: "
            f"{audit['max_abs_old_ref_delta']}"
        )

    critic_metrics = critic.update_critic(copy.deepcopy(ppo_batch))
    critic_after = _module_fingerprint(critic.critic_module)
    if not _fingerprint_changed(critic_before, critic_after):
        raise RuntimeError("critic optimizer update did not change parameters")

    actor_metrics = actor.update_actor(copy.deepcopy(ppo_batch))
    actor_after = _module_fingerprint(actor.actor_module_fsdp)
    if not _fingerprint_changed(actor_before_update, actor_after):
        raise RuntimeError("actor optimizer update did not change parameters")
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

    checkpoint_root = output / "checkpoints" / "global_step_1"
    actor.save_checkpoint(
        str(checkpoint_root / "actor"),
        hdfs_path=None,
        global_step=1,
        remove_previous_ckpt=False,
    )
    critic.save_checkpoint(
        str(checkpoint_root / "critic"),
        hdfs_path=None,
        global_step=1,
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
        "actor_before": actor_before,
        "actor_after": actor_after,
        "critic_before": critic_before,
        "critic_after": critic_after,
        "reference_before": reference_before,
        "reference_after": reference_after,
        "actor_log_prob_change": actor_log_prob_change,
        "replay_audit": audit,
        "actor_metrics": _json_value(actor_metrics.meta_info.get("metrics", {})),
        "critic_metrics": _json_value(critic_metrics.meta_info.get("metrics", {})),
        "checkpoint_root": str(checkpoint_root),
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
            },
        )
        run.log(
            {
                "global_step": 1,
                "gate/policy_tokens": audit["policy_tokens"],
                "gate/sequence_tokens": int(row.input_ids.numel()),
                "gate/old_ref_max_delta": audit["max_abs_old_ref_delta"],
                "gate/actor_log_prob_change": actor_log_prob_change,
            },
            step=1,
        )
        run.summary["status"] = report["status"]
        run.finish()
        print(json.dumps(report, sort_keys=True), flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
