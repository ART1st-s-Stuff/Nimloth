from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "experiments/training/rl/run_vagen_k4_id186_continue_phase.sh"
LAUNCHER = ROOT / "experiments/training/rl/launch_vagen_k4_id186_continue_on_hold.sh"
SLURM1 = ROOT / "experiments/training/rl/id186_k4_continue_phase1.slurm"
SLURM2 = ROOT / "experiments/training/rl/id186_k4_continue_phase2.slurm"


def test_id186_uses_two_strict_exact_resume_phases() -> None:
    runner = RUNNER.read_text()
    launcher = LAUNCHER.read_text()
    assert "resume_20_to_30" in runner and "resume_30_to_40" in runner
    assert "trainer.joint_dataloader_resume_policy=exact" in runner
    assert "trainer.joint_dataloader_resume_policy=reset" not in runner
    assert "trainer.total_training_steps=${TARGET_STEP}" in runner
    assert "trainer.val_before_train=${VAL_BEFORE_TRAIN}" in runner
    assert "global_step_20" in launcher and "global_step_30" in launcher
    assert "nimloth-id186-k4-continue-20-to30" in launcher
    assert "nimloth-id186-k4-continue-30-to40" in launcher


def test_id186_preserves_dataset_ids_and_migrates_only_transport() -> None:
    runner = RUNNER.read_text()
    launcher = LAUNCHER.read_text()
    assert "runtime_train_path.write_bytes(source_train.read_bytes())" in runner
    assert "assert train_rows==source_manifest['train_rows']" in runner
    assert "VAGEN_REMOTE_ENV_BASE_URL_OVERRIDE=\"${ENV_URL}\"" in launcher
    assert "VAGEN_REMOTE_ENV_BASE_URL_OVERRIDE_SCOPE=id186_exact_continuation_v1" in launcher
    assert "val_text.replace(" not in runner


def test_id186_keeps_exact4x2_tp8_and_dynamic_head_qualification() -> None:
    launcher = LAUNCHER.read_text()
    runner = RUNNER.read_text()
    assert "HEAD_CANDIDATES=()" in launcher
    assert "ID186_DYNAMIC_HEAD_RENDER_OK" in launcher
    assert "timeout --signal=TERM --kill-after=10s 150s" in launcher
    assert "NAVIGATION_HEAD_EXCLUSIONS=(dgx-09 dgx-10 dgx-13 dgx-23 dgx-32 dgx-37 dgx-51)" in launcher
    assert "ID186_RAY_4X2_OK" in runner
    assert "tensor_model_parallel_size: 8" in (
        ROOT / "external/VAGEN/vagen/configs/joint_id186_continue.yaml"
    ).read_text()
    for path in (SLURM1, SLURM2):
        source = path.read_text()
        assert "#SBATCH --nodes=4" in source
        assert "#SBATCH --gres=gpu:2" in source
        assert "#SBATCH --cpus-per-task=16" in source
        assert "#SBATCH --mem=64G" in source
        assert "#SBATCH --time=05:00:00" in source
        assert "#SBATCH --exclude=dgx-09,dgx-13,dgx-32,dgx-51" in source


def test_id186_validates_every_five_steps_without_overwrite() -> None:
    runner = RUNNER.read_text()
    assert "expected_validation_steps=({20,25,30}" in runner
    assert "else {20,25,30,35,40})" in runner
    assert "phase2 disables validation-before-train" in runner
    assert "actual_checkpoint_steps=={target-5,target}" in runner
    assert "ID186_DATALOADER_RESET_OK' not in log" in runner
    assert "phase=='resume_30_to_40'" in runner
    assert "atomic_json(run_out/'final_status.json'" in runner
