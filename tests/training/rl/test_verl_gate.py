from __future__ import annotations

from pathlib import Path


def test_transformers455_token_critic_avoids_removed_verl_constants() -> None:
    from nimloth.training.rl.verl_critic_455 import (
        Qwen2_5_VLForTokenClassification,
    )

    assert Qwen2_5_VLForTokenClassification.__name__ == (
        "Qwen2_5_VLForTokenClassification"
    )
    source = Path(
        "src/nimloth/training/rl/verl_critic_455.py"
    ).read_text(encoding="utf-8")
    assert "Qwen2_5_VL_START_DOCSTRING" not in source
    assert "self.model = Qwen2_5_VLModel(config)" in source
    assert "config.text_config.hidden_size" in source


def test_exact_replay_worker_config_is_full_actor_ref_critic() -> None:
    from nimloth.training.rl.verl_gate import build_exact_replay_worker_config

    config = build_exact_replay_worker_config(
        Path("external/VAGEN/vagen/trainer/config/ppo_trainer.yaml"),
        model_path=Path("/tmp/nimloth-k8-model"),
        world_size=8,
        max_token_length=8192,
    )
    assert config.algorithm.adv_estimator == "masked_gae"
    assert config.algorithm.gamma == 1.0
    assert config.algorithm.lam == 1.0
    assert config.actor_rollout_ref.model.path == "/tmp/nimloth-k8-model"
    assert config.actor_rollout_ref.model.enable_gradient_checkpointing is True
    assert config.actor_rollout_ref.model.use_remove_padding is False
    assert config.actor_rollout_ref.model.override_config.tie_word_embeddings is True
    assert config.actor_rollout_ref.actor.strategy == "fsdp"
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 8
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert config.actor_rollout_ref.actor.use_kl_loss is True
    assert config.actor_rollout_ref.actor.kl_loss_type == "low_var_kl"
    assert config.actor_rollout_ref.actor.kl_loss_coef == 0.001
    assert config.actor_rollout_ref.actor.fsdp_config.param_offload is True
    assert config.actor_rollout_ref.actor.fsdp_config.optimizer_offload is True
    assert config.actor_rollout_ref.ref.fsdp_config.param_offload is True
    assert config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu == 1
    assert config.critic.model.path == "/tmp/nimloth-k8-model"
    assert config.critic.model.tokenizer_path == "/tmp/nimloth-k8-model"
    assert config.critic.model.enable_gradient_checkpointing is True
    assert config.critic.model.use_remove_padding is False
    assert config.critic.ppo_mini_batch_size == 8
    assert config.critic.ppo_micro_batch_size_per_gpu == 1
    assert config.critic.forward_micro_batch_size_per_gpu == 1
    assert config.critic.model.fsdp_config.param_offload is True
    assert config.critic.model.fsdp_config.optimizer_offload is True
    assert config.critic.model.fsdp_config.model_dtype == "fp32"
    assert config.rollout_manager.use_loss_mask is True
    assert config.rollout_manager.use_gae_mask is True
    assert config.rollout_manager.use_multi_turn_reward is True


def test_exact_replay_runner_uses_real_full_verl_workers() -> None:
    runner = Path(
        "experiments/training/rl/run_verl_exact_replay_worker_gate.py"
    ).read_text(encoding="utf-8")
    assert 'EXPECTED_TRANSFORMERS = "4.55.4"' in runner
    assert 'args.wandb_project != "nimloth-rl"' in runner
    assert "W&B run-id argument/environment mismatch" in runner
    assert '(verl_path / "verl/__init__.py").is_file()' in runner
    assert "VERL source tree is missing or uninitialized" in runner
    assert "ActorRolloutRefWorker" in runner
    assert 'role="actor"' in runner
    assert 'role="ref"' in runner
    assert "CriticWorker" in runner
    assert "_install_transformers455_critic_patch()" in runner
    assert "build_verl_replay_row_from_trajectory" in runner
    assert "compute_log_prob" in runner
    assert "compute_ref_log_prob" in runner
    assert "compute_values" in runner
    assert "finalize_verl_exact_replay_batch" in runner
    assert "update_critic" in runner
    assert "update_actor" in runner
    assert "immutable reference parameters changed" in runner
    assert "save_checkpoint" in runner
    assert "VERL_EXACT_REPLAY_ALL_OK" in runner
    rank_runner = Path(
        "experiments/training/rl/run_verl_exact_replay_rank.sh"
    ).read_text(encoding="utf-8")
    assert ".venv-vagen-main/bin/python3" in rank_runner
    assert "SLURM_PROCID" in rank_runner
    assert "SLURM_TASK_LOCAL_RANK" in rank_runner
    assert "export LOCAL_RANK=0" in rank_runner
    assert "torch.cuda.set_device(local_rank)" in runner
    assert 'dist.init_process_group(backend="nccl", device_id=process_device)' in runner
    assert "dist.barrier(device_ids=[local_rank])" in runner
    assert "random/missing head forbidden" in runner
    assert "single-node torchrun requires every process" in runner
    assert "MASTER_ADDR" in rank_runner
    assert "MASTER_PORT" in rank_runner
    assert "external/VAGEN/verl" in rank_runner
    source_index = rank_runner.index("source /project/peilab/atst/flower/.env")
    restore_index = rank_runner.index(
        "export WANDB_PROJECT=${_REQUESTED_WANDB_PROJECT}"
    )
    assert source_index < restore_index
    assert "run_verl_exact_replay_worker_gate.py" in rank_runner
    hold = Path(
        "experiments/training/rl/hold_verl_exact_replay_normal8.slurm"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=normal" in hold
    assert "#SBATCH --gres=gpu:8" in hold
    assert "#SBATCH --nodelist=" not in hold
    launcher = Path(
        "experiments/training/rl/launch_verl_exact_replay_in_hold.sh"
    ).read_text(encoding="utf-8")
    assert "sbatch" not in launcher
    assert 'JobState=RUNNING' in launcher
    assert 'Partition=normal' in launcher
    assert "--ntasks=1" in launcher
    assert "--gpus=8" in launcher
    assert "--gpus-per-task=1" not in launcher
    assert "run_verl_exact_replay_torchrun.sh" in launcher
    assert "VERL_EXACT_REPLAY_ARTIFACTS_OK" in launcher
    torchrun = Path(
        "experiments/training/rl/run_verl_exact_replay_torchrun.sh"
    ).read_text(encoding="utf-8")
    assert ".venv-vagen-main/bin/python3" in torchrun
    assert "-m torch.distributed.run" in torchrun
    assert "--nproc-per-node=8" in torchrun
    assert "run_verl_exact_replay_worker_gate.py" in torchrun


def test_exact_replay_worker_config_rejects_invalid_world_or_budget() -> None:
    import pytest

    from nimloth.training.rl.verl_gate import build_exact_replay_worker_config

    path = Path("external/VAGEN/vagen/trainer/config/ppo_trainer.yaml")
    with pytest.raises(ValueError, match="world_size"):
        build_exact_replay_worker_config(
            path, model_path=Path("/tmp/model"), world_size=0
        )
    with pytest.raises(ValueError, match="max_token_length"):
        build_exact_replay_worker_config(
            path,
            model_path=Path("/tmp/model"),
            world_size=8,
            max_token_length=0,
        )
