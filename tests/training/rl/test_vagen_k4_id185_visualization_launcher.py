from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / "experiments/training/rl/id185_k4_visualize_base_failure.slurm"
LAUNCHER = ROOT / "experiments/training/rl/launch_vagen_k4_id185_visualize_base_failure_on_hold.sh"
RUNNER = ROOT / "experiments/training/rl/run_vagen_k4_id185_visualize_base_failure.sh"
CANARY_SLURM = ROOT / "experiments/training/rl/id187_rollout_browser_canary.slurm"
STEP0_SLURM = ROOT / "experiments/training/rl/id188_step0_rollout_browser_canary.slurm"
STEP0_RUNNER = ROOT / "experiments/training/rl/run_vagen_k4_id188_step0_browser_canary.sh"
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
    assert "ID185_VIS_ROLLOUT_SAMPLE_ID" not in launcher
    assert "ID185_VIS_SEED=\"${VIS_SEED}\"" in launcher


def test_visualization_runner_is_one_read_only_failure_rollout() -> None:
    source = RUNNER.read_text()
    assert "--config-name=joint_id185_visualize_one" in source
    assert "ID185_K4_VISUALIZATION_RESTORE_OK global_step=20" in source
    assert "VALIDATION_BATCH_JOURNAL_COMPLETE batches=1 rows=1" in source
    assert "if expected_outcome == 'failure':" in source
    assert "elif expected_outcome == 'success':" in source
    assert "assert not list((run/'checkpoints').glob('global_step_*'))" in source
    assert "row['data_source']=='navigation_base_test_id185'" in source
    assert "row['seed']==expected_seed" in source
    assert "export ID185_VIS_SEED=${ID185_VIS_SEED:-2}" in source
    assert "evaluation_browser/global_step_20" in source
    assert "manifest_sha256" in source
    assert "render_id185_rollout_visualization.py" in source
    assert "rollout_audit/index.html" in source
    assert "optimizer update" in source


def test_id187_browser_canary_has_unique_identity_and_no_training() -> None:
    source = CANARY_SLURM.read_text()
    assert "#SBATCH --job-name=nimloth-id187-browser-canary" in source
    assert "#SBATCH --nodes=4" in source
    assert "#SBATCH --gres=gpu:2" in source
    assert "#SBATCH --partition=preempt" in source
    assert "ID185_VIS_RUN_NAME_OVERRIDE=187_smoke_rollout_browser" in source
    assert "ID185_VIS_EXPECTED_OUTCOME=any" in source
    assert "ID185_VIS_ENABLE_WANDB=true" in source
    assert "ID185_VIS_WANDB_RUN_ID=nimloth-id187-smoke-rollout-browser" in source
    assert "#SBATCH --partition=preempt" in source
    assert "ID185_VIS_EXPECTED_PARTITION=preempt" in source
    assert "ID185_VIS_SOURCE_BOUNDARY=20" in source
    assert "preempt_retry2" in source
    launcher = LAUNCHER.read_text()
    runner = RUNNER.read_text()
    assert "VIS_PARTITION=${ID185_VIS_EXPECTED_PARTITION:-normal}" in launcher
    assert "ROLLOUT_BROWSER_LAUNCHER_ERROR" in launcher
    assert '"${SLURM_JOB_PARTITION:-}" == "${ID185_VIS_EXPECTED_PARTITION}"' in runner


def test_id188_step0_canary_uses_sft2_initialization_without_resume() -> None:
    slurm = STEP0_SLURM.read_text()
    runner = STEP0_RUNNER.read_text()
    assert "#SBATCH --partition=preempt" in slurm
    assert "#SBATCH --nodes=4" in slurm
    assert "#SBATCH --gres=gpu:2" in slurm
    assert "188_smoke_rollout_browser_k4_dp8_tp8_step0_" in slurm
    assert "run_vagen_k4_id188_step0_browser_canary.sh" in slurm
    assert "ID185_VIS_SOURCE_BOUNDARY=0" in slurm
    assert "--config-name=joint_id188_step0_visualize_one" in runner
    assert "trainer.resume_mode=disable" in runner
    assert "ID188_K4_STEP0_BOOTSTRAP_OK global_step=0" in runner
    assert "evaluation_browser/global_step_0" in runner
    assert "validation/0.jsonl" in runner
    assert "load_complete_joint_checkpoint" not in runner
    assert "global_step_*" in runner


def test_visualization_scripts_are_executable() -> None:
    for path in (SLURM, LAUNCHER, RUNNER, CANARY_SLURM, STEP0_SLURM, STEP0_RUNNER):
        assert path.stat().st_mode & 0o111
