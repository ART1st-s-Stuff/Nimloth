from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[3]
VAGEN = ROOT / "external" / "VAGEN"
RUNNER = ROOT / "experiments" / "training" / "rl" / "run_vagen_joint_update_gate_phase.sh"
LAUNCHER = ROOT / "experiments" / "training" / "rl" / "launch_vagen_joint_update_gate_on_hold.sh"
HOLD = ROOT / "experiments" / "training" / "rl" / "hold_eight_gpu_60m.slurm"
CONFIG = VAGEN / "vagen" / "configs" / "joint_id171_gate.yaml"
TRAIN_DATA = VAGEN / "examples" / "train" / "navigation" / "train_navigation_joint_id171.yaml"
VAL_DATA = VAGEN / "examples" / "train" / "navigation" / "val_navigation_joint_id171.yaml"


def test_shell_contracts_parse_and_use_target_allocation() -> None:
    for path in (RUNNER, LAUNCHER, HOLD):
        subprocess.run(["bash", "-n", str(path)], check=True)
    hold = HOLD.read_text()
    assert "#SBATCH --partition=normal" in hold
    assert "#SBATCH --gres=gpu:8" in hold
    assert "#SBATCH --time=01:00:00" in hold
    assert "dgx-13,dgx-32,dgx-37,dgx-51" in hold
    launcher = LAUNCHER.read_text()
    assert "PHASE=update_1" in launcher
    assert "PHASE=resume_update_2" in launcher
    assert "--gres=gpu:8" in launcher
    assert "TimeLimit=01:00:00" in launcher
    assert "171_smoke_vagenlite_jointupdate_dp8_tp8_" in launcher
    assert "nimloth-id171-dp8-hold" in hold
    assert "sleep infinity" not in hold
    assert "launch_vagen_joint_update_gate_on_hold.sh" in hold
    assert '"${SLURM_JOB_ID}"' in hold
    assert "ReqTRES=[^ ]*mem=256G" in launcher
    assert "MinMemoryNode=256G" in launcher
    assert "AllocTRES=[^ ]*mem=256G" not in launcher
    assert "ReqTRES=[^ ]*mem=256G" in RUNNER.read_text()
    assert "MinMemoryNode=256G" in RUNNER.read_text()
    assert "AllocTRES=[^ ]*mem=256G" not in RUNNER.read_text()
    expected_verl = "494f264494b2525f2c13595f63ac4912963e6d2f"
    assert expected_verl in launcher
    assert expected_verl in RUNNER.read_text()


def test_embedded_python_blocks_compile_independently() -> None:
    source = RUNNER.read_text()
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(blocks) == 4
    for index, block in enumerate(blocks):
        compile(block, f"{RUNNER}:inline-{index}", "exec")


def test_runner_pins_exact_git_checkpoint_and_clean_worktrees() -> None:
    source = RUNNER.read_text()
    for name in (
        "EXPECTED_PARENT_COMMIT",
        "EXPECTED_VAGEN_COMMIT",
        "EXPECTED_VERL_COMMIT",
    ):
        assert name in source
    assert "status --porcelain --untracked-files=all" in source
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert "checkpoint_preflight.json" in source
    assert "joint_checkpoint_complete.json" in source
    assert "load_complete_joint_checkpoint" in source
    assert "Setting global step to 1" in source
    assert "rsync" not in source
    assert "scp " not in source
    assert 'COMMAND=(' in source
    assert '"${COMMAND[@]}"' in source
    assert "PHASE_TAG=$([[ \"${PHASE}\" == update_1 ]] && echo p1 || echo p2)" in source
    assert "RUNTIME_ROOT=/tmp/i171-${SLURM_JOB_ID}-${PHASE_TAG}" in source
    assert "RAY_TMPDIR=${RUNTIME_ROOT}" in source
    worst_case_plasma_socket = (
        "/tmp/i171-99999999-p2/ray/session_2026-08-15_00-40-07_"
        "477970_1313372/sockets/plasma_store"
    )
    assert len(worst_case_plasma_socket.encode()) <= 107


def test_human_approved_values_are_explicit_and_test_only() -> None:
    source = CONFIG.read_text()
    exact = (
        "implementation: id171_dp8_resume_smoke_v1",
        "run_seed: 42001",
        "gamma: 0.99",
        "gae_lambda: 0.95",
        "ppo_clip_ratio: 0.2",
        "token_kl_coefficient: 0.01",
        "guided_entropy_coefficient: 0.01",
        "checkpoint_frequency: 1",
        "lr: 1.0e-7",
        "lr: 1.0e-4",
        "initial_snapshot_source_step: 776",
        "tensor_model_parallel_size: 8",
        "data_parallel_size: 1",
        "mm_encoder_tp_mode: data",
        "train_batch_size: 8",
        "ppo_mini_batch_size: 8",
        "ppo_micro_batch_size_per_gpu: 1",
        "freeze_vision_tower: true",
        "huggingface_hub:",
        "hf_save_freq: null",
    )
    for value in exact:
        assert value in source
    assert "non-production" in source
    train_data = TRAIN_DATA.read_text()
    val_data = VAL_DATA.read_text()
    assert "base_train" in train_data
    assert "n_envs: 8" in train_data
    assert "max_turns: 2" in train_data
    assert "eval_set: base" in val_data
    for data_config in (train_data, val_data):
        assert "per_turn_format_reward: 0.0" in data_config
        assert "format_reward: 0.0" in data_config
        assert "success_reward: 1.0" in data_config


def test_production_stays_closed_without_id171_escape_hatch() -> None:
    trainer = (VAGEN / "vagen" / "ray_trainer.py").read_text()
    gate = (VAGEN / "vagen" / "joint_policy" / "integration_gate.py").read_text()
    assert "if self.joint_integration_gate is None:" in trainer
    assert "refusing production training" in trainer
    assert 'experiment_id != 171' in gate
    assert '{"update_1", "resume_update_2"}' in gate
