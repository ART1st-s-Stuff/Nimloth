from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / "experiments/training/rl/id189_source20_base_common120.slurm"
RETRY1_SLURM = ROOT / "experiments/training/rl/id189_source20_base_common120_retry1.slurm"
RUNNER = ROOT / "experiments/training/rl/run_vagen_k4_id189_source20_base_common120.sh"
LAUNCHER = ROOT / "experiments/training/rl/launch_vagen_k4_1x8_browser_on_hold.sh"
CONFIG = ROOT / "external/VAGEN/vagen/configs/joint_id189_source20_base_common120.yaml"
VAL_SOURCE = ROOT / "external/VAGEN/examples/train/navigation/val_navigation_joint_id185.yaml"


def test_id189_is_frozen_one_node_source20_full_browser() -> None:
    slurm = SLURM.read_text()
    runner = RUNNER.read_text()
    launcher = LAUNCHER.read_text()
    config = CONFIG.read_text()
    assert "#SBATCH --nodes=1" in slurm
    assert "#SBATCH --gres=gpu:8" in slurm
    assert "ID185_VIS_PHASE_NAME_OVERRIDE=base_common120" in slurm
    assert "run_vagen_k4_id189_source20_base_common120.sh" in slurm
    assert "joint_id189_source20_base_common120" in runner
    assert "ID189_K4_SOURCE20_BASE_COMMON120_RESTORE_OK global_step=20" in runner
    assert "len(rows)==120" in runner
    assert "complete['batch_count']==3" in runner
    assert "complete['rollout_count']==120" in runner
    assert "len(rollout_files)==120" in runner
    assert "len(sims)==100" in runner
    assert "shape==(16,1024)" in runner
    assert "not list((run/'checkpoints').glob('global_step_*'))" in runner
    assert "PHASE_NAME=${ID185_VIS_PHASE_NAME_OVERRIDE:-visualization}" in launcher
    assert "phase: source20_base_common120" in config
    assert "validation_batch_journal_expected_rows: 120" in config
    assert "validation_rollout_browser_expected_rows: 120" in config
    assert "validation_rollout_browser_capture_mcts_process: true" in config
    assert "validation_visualization_data_source" not in config


def test_id189_retry1_uses_fresh_non_resume_identities() -> None:
    attempt0 = SLURM.read_text()
    retry1 = RETRY1_SLURM.read_text()
    assert "source20_base_common120_t20_s100_preempt_retry1" in retry1
    assert "source20-base-common120-preempt-r1" in retry1
    assert "WANDB_RESUME=never" not in retry1  # launcher owns the fixed never-resume gate
    assert "source20_base_common120_t20_s100_preempt_retry1" not in attempt0
    assert "source20-base-common120-preempt-r1" not in attempt0


def test_id189_val_source_filters_to_exact_base_and_common_60_each() -> None:
    payload = yaml.safe_load(VAL_SOURCE.read_text())
    envs = [
        item
        for item in payload["envs"]
        if item["config"]["eval_set"] in {"base", "common_sense"}
    ]
    assert [item["config"]["eval_set"] for item in envs] == [
        "base",
        "common_sense",
    ]
    assert all(item["n_envs"] == 60 for item in envs)
    assert all(item["seed_list"][:60] == list(range(1, 61)) for item in envs)
    assert all(item["seed_list"][60:] == [61] for item in envs)
    assert sum(item["n_envs"] for item in envs) == 120
