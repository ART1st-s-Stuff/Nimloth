from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / "experiments/training/rl/id188_step0_base_common120_normal4x2.slurm"
RUNNER = ROOT / "experiments/training/rl/run_vagen_k4_id188_step0_base_common120_normal4x2.sh"
LAUNCHER = ROOT / "experiments/training/rl/launch_vagen_k4_id185_visualize_base_failure_on_hold.sh"
CONFIG = ROOT / "external/VAGEN/vagen/configs/joint_id188_step0_base_common120.yaml"


def test_id188_step0_full_browser_uses_frozen_normal_4x2_contract() -> None:
    slurm = SLURM.read_text()
    runner = RUNNER.read_text()
    config = yaml.safe_load(CONFIG.read_text())

    assert "#SBATCH --account=peilab" in slurm
    assert "#SBATCH --partition=normal" in slurm
    assert "#SBATCH --nodes=4" in slurm
    assert "#SBATCH --gres=gpu:2" in slurm
    assert "ID185_VIS_SOURCE_BOUNDARY=0" in slurm
    assert "ID185_VIS_EXPECTED_OUTCOME=any" in slurm
    assert "ID185_VIS_PHASE_NAME_OVERRIDE=base_common120" in slurm
    assert "nimloth-id188-eval-rollout-browser-k4-step0-base-common120-normal-4x2" in slurm

    assert "ID188_K4_STEP0_BASE_COMMON120_BOOTSTRAP_OK global_step=0" in runner
    assert "--config-name=joint_id188_step0_base_common120" in runner
    assert "trainer.resume_mode=disable" in runner
    assert "trainer.val_only=true" in runner
    assert "len(rows)==120" in runner
    assert "global_step_0" in runner
    assert "source_step':776" in runner
    assert "not list((run/'checkpoints').glob('global_step_*'))" in runner
    assert "workers=8" in runner
    assert "SOURCE_CHECKPOINT" not in runner

    assert config["joint_integration_gate"] == {
        "enabled": True,
        "implementation": "id188_k4_step0_browser_v1",
        "experiment_id": 188,
        "phase": "step0_base_common120",
    }
    assert config["trainer"]["resume_mode"] == "disable"
    assert config["trainer"]["val_only"] is True
    assert config["trainer"]["nnodes"] == 4
    assert config["trainer"]["n_gpus_per_node"] == 2
    assert config["trainer"]["validation_batch_journal_expected_rows"] == 120
    assert config["trainer"]["validation_rollout_browser_expected_rows"] == 120
    assert config["trainer"]["validation_rollout_browser_capture_mcts_process"] is True
    assert config["trainer"]["validation_rollout_browser_source_step"] == 776
    assert "validation_visualization_data_source" not in config["trainer"]


def test_launcher_propagates_effective_browser_pack_workers_to_ray() -> None:
    launcher = LAUNCHER.read_text()
    assert "VAGEN_ROLLOUT_BROWSER_PACK_WORKERS=${VAGEN_ROLLOUT_BROWSER_PACK_WORKERS:-8}" in launcher
    assert 'VAGEN_ROLLOUT_BROWSER_PACK_WORKERS="${VAGEN_ROLLOUT_BROWSER_PACK_WORKERS}"' in launcher
    assert "'rollout_browser_pack_workers':os.environ.get('VAGEN_ROLLOUT_BROWSER_PACK_WORKERS')" in launcher
    assert "row['rollout_browser_pack_workers']=='8'" in launcher
