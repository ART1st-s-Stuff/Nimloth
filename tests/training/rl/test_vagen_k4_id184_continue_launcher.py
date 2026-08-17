from __future__ import annotations

from pathlib import Path
import re
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[3]
VAGEN = ROOT / "external/VAGEN"
RUNNER = ROOT / "experiments/training/rl/run_vagen_k4_id184_continue_to20.sh"
LAUNCHER = ROOT / "experiments/training/rl/launch_vagen_k4_id184_continue_to20_on_hold.sh"
SLURM = ROOT / "experiments/training/rl/id184_k4_continue_to20.slurm"
CONFIG = VAGEN / "vagen/configs/joint_id184_continue.yaml"
TRAIN = VAGEN / "examples/train/navigation/train_navigation_joint_id184.yaml"
VAL = VAGEN / "examples/train/navigation/val_navigation_joint_id184.yaml"


def test_id184_shells_parse_and_request_exact_four_by_two() -> None:
    for path in (RUNNER, LAUNCHER, SLURM):
        subprocess.run(["bash", "-n", str(path)], check=True)
    slurm = SLURM.read_text()
    for value in (
        "#SBATCH --partition=normal",
        "#SBATCH --nodes=4",
        "#SBATCH --ntasks=4",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gres=gpu:2",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --mem=64G",
        "#SBATCH --time=05:00:00",
        "dgx-13,dgx-32,dgx-51",
    ):
        assert value in slurm
    launcher = LAUNCHER.read_text()
    assert "NAVIGATION_HEAD_EXCLUSIONS=(dgx-13 dgx-23 dgx-32 dgx-37 dgx-51)" in launcher
    assert "ID184_EXPECTED_NNODES=4" in launcher
    assert "ID184_EXPECTED_GPUS_PER_NODE=2" in launcher
    assert "ID184_RAY_4X2_OK" in RUNNER
    assert "persist_ray_logs pre_cleanup" in launcher
    assert "persist_ray_logs post_cleanup" in launcher
    assert "capture_manifest.json" in launcher
    assert "max_total_bytes=16*1024*1024" in launcher


def test_id184_runner_has_exact_source_reset_and_output_boundaries() -> None:
    source = RUNNER.read_text()
    for value in (
        "global_step_10/joint_checkpoint_complete.json",
        "trainer.resume_mode=resume_path",
        "trainer.total_training_steps=20",
        "trainer.total_epochs=20",
        "trainer.val_before_train=true",
        "trainer.test_freq=5",
        "trainer.save_freq=5",
        "joint_dataloader_resume_policy=reset",
        "ID184_K4_CONTINUE_RESUME_OK global_step=10",
        "ID184_DATALOADER_RESET_OK global_step=10",
        "ID184_TRAINING_CONTRACT_PATH_MIGRATION_OK",
        "EXPECTED_CHECKPOINT_STEPS=(15 20)",
        "EXPECTED_VALIDATION_STEPS=(10 15 20)",
        "EXPECTED_SOURCE=796",
        "summarize_canary_validation_rows",
    ):
        assert value in source
    assert "WANDB_RUN_ID=nimloth-id184-k4-continue-to20" in source
    assert "WANDB_RESUME=never" in source
    assert "WANDB_RESUME=must" not in source
    assert "steps==list(range(10,21))" in source
    assert "ID183_SOURCE_RUN_OUT" in source
    assert "source_step_786" in source
    for forbidden in ("global_step_11", "global_step_12", "global_step_13", "global_step_14", "global_step_16", "global_step_17", "global_step_18", "global_step_19"):
        assert forbidden not in source


def test_id184_data_and_config_are_expanded_pool_with_fixed_update_batch() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    assert config["algorithm"]["adv_estimator"] == "joint_frozen_v_gae"
    assert config["joint_integration_gate"] == {
        "enabled": True,
        "implementation": "id184_k4_continue_to20_v1",
        "experiment_id": 184,
        "phase": "resume_10_to_20",
    }
    assert config["data"]["train_batch_size"] == 24
    assert config["data"]["gen_batch_size"] == 24
    assert config["data"]["val_batch_size"] == 40
    assert config["data"]["shuffle"] is True
    assert config["data"]["seed"] == 42184
    assert config["trainer"]["total_training_steps"] == 20
    assert config["trainer"]["resume_mode"] == "resume_path"
    assert config["trainer"]["joint_dataloader_resume_policy"] == "reset"
    assert config["trainer"]["test_freq"] == 5
    assert config["trainer"]["save_freq"] == 5

    train = yaml.safe_load(TRAIN.read_text())["envs"]
    val = yaml.safe_load(VAL.read_text())["envs"]
    assert [row["n_envs"] for row in train] == [60, 60, 60]
    assert all(row["seed"] == [0, 1199, 1] for row in train)
    assert [row["n_envs"] for row in val] == [8, 8, 8, 8, 8]


def test_id184_embedded_python_compiles() -> None:
    for path in (RUNNER, LAUNCHER):
        blocks = re.findall(r"<<'PY'.*?\n(.*?)\nPY", path.read_text(), flags=re.DOTALL)
        assert blocks
        for index, block in enumerate(blocks):
            compile(block, f"{path}:inline-{index}", "exec")
