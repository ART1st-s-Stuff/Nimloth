from __future__ import annotations

from pathlib import Path

import pytest


def test_verl_runtime_patch_installs_scheduler_and_critic_compatibility() -> None:
    import sys

    from nimloth.training.rl.verl_runtime_patch import (
        install_nimloth_verl_runtime_patches,
    )
    from verl.utils import torch_functional

    install_nimloth_verl_runtime_patches()
    critic_patch = sys.modules[
        "verl.models.transformers.modeling_qwen_2_5_vl_patch"
    ]
    assert critic_patch._nimloth_transformers455 is True
    assert critic_patch.Qwen2_5_VLForTokenClassification.__module__ == (
        "nimloth.training.rl.verl_critic_455"
    )
    assert getattr(
        torch_functional.get_constant_schedule_with_warmup,
        "_nimloth_zero_warmup_fixed",
        False,
    )


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
    assert "_checkpoint_conversion_mapping = dict(" in source
    worker = Path(
        "external/VAGEN/verl/verl/workers/fsdp_workers.py"
    ).read_text(encoding="utf-8")
    assert "output_loading_info=True" in worker
    assert "Qwen2.5-VL critic checkpoint coverage failed" in worker


def test_zero_warmup_scheduler_patch_preserves_first_optimizer_lr(
    monkeypatch,
) -> None:
    import pytest
    import torch

    from nimloth.training.rl.verl_gate import install_verl_zero_warmup_scheduler_patch
    from verl.utils import torch_functional

    from torch.optim.lr_scheduler import LambdaLR

    def pinned_broken_schedule(optimizer, num_warmup_steps, last_epoch=-1):
        return LambdaLR(
            optimizer,
            lambda step: min(
                1, float(step) / float(max(1, num_warmup_steps))
            ),
            last_epoch,
        )

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    broken_optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    pinned_broken_schedule(broken_optimizer, num_warmup_steps=0)
    assert broken_optimizer.param_groups[0]["lr"] == 0.0

    monkeypatch.setattr(
        torch_functional,
        "get_constant_schedule_with_warmup",
        pinned_broken_schedule,
    )
    install_verl_zero_warmup_scheduler_patch()
    fixed_optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    torch_functional.get_constant_schedule_with_warmup(
        fixed_optimizer, num_warmup_steps=0
    )
    assert fixed_optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_wm_aux_config_requires_explicit_checkpoint_or_mechanics_override() -> None:
    from omegaconf import OmegaConf

    from nimloth.training.rl.verl_gate import configure_nimloth_wm_auxiliary

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "nimloth_wm_aux": {
                        "enabled": False,
                        "checkpoint_dir": None,
                        "allow_random_init": False,
                        "latent_token_count": None,
                        "loss_coef": 1.0,
                        "lr": 1e-5,
                    }
                }
            }
        }
    )
    with pytest.raises(ValueError, match="requires an SFT2 checkpoint"):
        configure_nimloth_wm_auxiliary(
            config, latent_token_count=8, checkpoint_dir=None
        )
    configure_nimloth_wm_auxiliary(
        config,
        latent_token_count=8,
        checkpoint_dir=None,
        loss_coef=0.3,
        learning_rate=2e-5,
        allow_random_init=True,
    )
    wm = config.actor_rollout_ref.actor.nimloth_wm_aux
    assert wm.enabled is True
    assert wm.latent_token_count == 8
    assert wm.latent_query_mode == "inject"
    assert wm.allow_random_init is True
    assert wm.loss_coef == 0.3
    assert wm.lr == 2e-5


def test_online_rollout_config_selects_strict_service_manager() -> None:
    from omegaconf import OmegaConf

    from nimloth.training.rl.verl_gate import configure_nimloth_online_rollout

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"temperature": 0.7}},
            "rollout_manager": {},
        }
    )
    configure_nimloth_online_rollout(
        config, latent_token_count=8, max_think_tokens=256, use_service=True
    )
    assert config.rollout_manager.latent_token_count == 8
    assert config.rollout_manager.latent_query_mode == "inject"
    assert config.rollout_manager.max_think_tokens == 256
    assert config.rollout_manager.temperature == 0.7
    assert config.rollout_manager.manager_class is None
    assert config.rollout_manager.service_manager_class.endswith(
        "NimlothQwenVLRolloutManagerService"
    )


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
    assert config.actor_rollout_ref.model.external_lib == (
        "nimloth.training.rl.verl_runtime_patch"
    )
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
    assert config.critic.model.external_lib == "nimloth.training.rl.verl_runtime_patch"
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
    assert 'EXPECTED_VAGEN = "e00131c2' in runner
    assert 'EXPECTED_VERL = "490a3cb5' in runner
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
    assert "--resume-checkpoint-root" in runner
    assert "--resume-result" in runner
    assert "--enable-wm-aux-mechanics" in runner
    assert "WM auxiliary optimizer update did not change parameters" in runner
    assert "resumed actor fingerprint does not match source result" in runner
    assert "resumed critic fingerprint does not match source result" in runner
    assert "del_local_after_load=False" in runner
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
    assert 'RESUME_CHECKPOINT_ROOT:?set both resume paths' in torchrun
    assert '--save-global-step "${SAVE_GLOBAL_STEP:-1}"' in torchrun
    assert "SAVE_GLOBAL_STEP=${SAVE_GLOBAL_STEP:-1}" in launcher
    assert "WM_AUX_MECHANICS=${WM_AUX_MECHANICS:-0}" in launcher
    assert "nimloth_wm_aux.pt" in launcher
    assert 'f"global_step_{global_step}"' in launcher


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
