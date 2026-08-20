from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / "experiments/training/rl/id185_k4_visualize_base_failure.slurm"
LAUNCHER = ROOT / "experiments/training/rl/launch_vagen_k4_id185_visualize_base_failure_on_hold.sh"
RUNNER = ROOT / "experiments/training/rl/run_vagen_k4_id185_visualize_base_failure.sh"
SAMPLE_ID = "sha256:babf2d061be39ece523d4d688e10f04a8a11f6dcda0d56e6d1e705c8b3a2a779"


def test_visualization_launcher_keeps_exact_tp8_topology() -> None:
    slurm = SLURM.read_text()
    launcher = LAUNCHER.read_text()
    assert "#SBATCH --nodes=4" in slurm
    assert "#SBATCH --gres=gpu:2" in slurm
    assert "#SBATCH --cpus-per-task=16" in slurm
    assert "#SBATCH --mem=64G" in slurm
    assert "--exclude=dgx-09,dgx-13,dgx-32,dgx-51" in slurm
    assert "NAVIGATION_HEAD_EXCLUSIONS=(dgx-09 dgx-10 dgx-13 dgx-23 dgx-32 dgx-37 dgx-51)" in launcher
    assert "ID185_DYNAMIC_HEAD_RENDER_OK" in launcher
    assert "--tensor_parallel_size" not in launcher
    assert 'ID185_VIS_ROLLOUT_SAMPLE_ID="' + SAMPLE_ID + '"' in launcher


def test_visualization_runner_is_one_read_only_failure_rollout() -> None:
    source = RUNNER.read_text()
    assert "--config-name=joint_id185_visualize_one" in source
    assert "ID185_K4_VISUALIZATION_RESTORE_OK global_step=20" in source
    assert "VALIDATION_BATCH_JOURNAL_COMPLETE batches=1 rows=1" in source
    assert "audit['success'] is False" in source
    assert "assert not list((run/'checkpoints').glob('global_step_*'))" in source
    assert SAMPLE_ID in source
    assert "render_id185_rollout_visualization.py" in source
    assert "rollout_audit/index.html" in source
    assert "optimizer update" in source


def test_visualization_scripts_are_executable() -> None:
    for path in (SLURM, LAUNCHER, RUNNER):
        assert path.stat().st_mode & 0o111
