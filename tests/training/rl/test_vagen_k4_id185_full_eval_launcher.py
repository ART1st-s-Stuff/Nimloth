from __future__ import annotations

from pathlib import Path
import re
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[3]
VAGEN = ROOT / "external/VAGEN"
RUNNER = ROOT / "experiments/training/rl/run_vagen_k4_id185_full_eval_test300.sh"
LAUNCHER = ROOT / "experiments/training/rl/launch_vagen_k4_id185_full_eval_test300_on_hold.sh"
SLURM = ROOT / "experiments/training/rl/id185_k4_full_eval_test300.slurm"
CONFIG = VAGEN / "vagen/configs/joint_id185_full_eval.yaml"
TRAIN = VAGEN / "examples/train/navigation/train_navigation_joint_id185.yaml"
VAL = VAGEN / "examples/train/navigation/val_navigation_joint_id185.yaml"


def test_id185_shells_parse_and_request_exact_four_by_two() -> None:
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
        "dgx-09,dgx-13,dgx-32,dgx-51",
    ):
        assert value in slurm
    launcher = LAUNCHER.read_text()
    assert "#SBATCH --job-name=nimloth-id185-k4-eval-test300-r4" in slurm
    assert "NAVIGATION_HEAD_EXCLUSIONS=(dgx-09 dgx-10 dgx-13 dgx-23 dgx-32 dgx-37 dgx-51)" in launcher
    assert "ID185_DYNAMIC_HEAD_RENDER_OK" in launcher
    assert "head_candidate_results.tsv" in launcher
    assert "--kill-after=10s 150s" in launcher
    assert 'for candidate in "${HEAD_CANDIDATES[@]}"' in launcher
    assert "ID185_EXPECTED_NNODES=4" in launcher
    assert "ID185_EXPECTED_GPUS_PER_NODE=2" in launcher
    assert "persist_ray_logs pre_cleanup" in launcher
    assert "persist_ray_logs post_cleanup" in launcher
    assert "max_total_bytes=16*1024*1024" in launcher


def test_id185_runner_is_full_eval_only_from_step20() -> None:
    source = RUNNER.read_text()
    for value in (
        "SOURCE_CHECKPOINT=${ID184_SOURCE_RUN_OUT}/checkpoints/global_step_20",
        "trainer.resume_mode=resume_path",
        "trainer.total_training_steps=20",
        "trainer.total_epochs=20",
        "trainer.val_before_train=true",
        "trainer.val_only=true",
        "trainer.test_freq=-1",
        "trainer.joint_dataloader_resume_policy=exact",
        "ID185_K4_FULL_EVAL_RESTORE_OK global_step=20",
        "ID185_TRAINING_CONTRACT_PATH_MIGRATION_OK",
        "EXPECTED_VALIDATION_ROWS=300",
        "expected_rows_per_source=60",
        "validation_seeds':'explicit historical VAGEN seeds 1..60",
    ):
        assert value in source
    assert "RUN_NAME=185_eval_k4schemeb_dp8_tp8_source20_test5x60_t20_s100_c1_a1_b85p78297006578457_t1_cot07p095_retry4" in LAUNCHER.read_text()
    assert "WANDB_RUN_ID=nimloth-id185-k4-full-eval-test300-retry4" in source
    assert "VALIDATION_BATCH_JOURNAL_COMPLETE" in source
    assert "validation_batch_journal" in source
    assert "steps==[20]" in source
    assert "global_step_15" not in source
    assert "global_step_10" not in source
    assert "rollout_data" not in source


def test_id185_data_and_config_are_full_test300() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    assert config["joint_integration_gate"] == {
        "enabled": True,
        "implementation": "id185_k4_full_eval_test300_v1",
        "experiment_id": 185,
        "phase": "full_eval_test300",
    }
    assert config["data"]["val_batch_size"] == 40
    assert config["trainer"]["val_before_train"] is True
    assert config["trainer"]["val_only"] is True
    assert config["trainer"]["test_freq"] == -1
    assert config["trainer"]["resume_mode"] == "resume_path"
    assert config["trainer"]["joint_dataloader_resume_policy"] == "exact"

    train = yaml.safe_load(TRAIN.read_text())["envs"]
    val = yaml.safe_load(VAL.read_text())["envs"]
    assert [row["n_envs"] for row in train] == [60, 60, 60]
    assert all(row["seed"] == [0, 1199, 1] for row in train)
    assert [row["n_envs"] for row in val] == [60, 60, 60, 60, 60]
    assert all(row["seed_list"][:60] == list(range(1, 61)) for row in val)


def test_id185_embedded_python_compiles() -> None:
    for path in (RUNNER, LAUNCHER):
        blocks = re.findall(r"<<'PY'.*?\n(.*?)\nPY", path.read_text(), flags=re.DOTALL)
        assert blocks
        for index, block in enumerate(blocks):
            compile(block, f"{path}:inline-{index}", "exec")
