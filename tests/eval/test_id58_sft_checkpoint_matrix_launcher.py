from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / "experiments/training/sft2/id58_sft_checkpoint_state_matrix.slurm"
RUNNER = ROOT / "experiments/training/sft2/run_id58_sft_checkpoint_state_matrix.sh"


def test_id58_slurm_contract_is_one_gpu_readonly_and_time_bounded() -> None:
    text = SLURM.read_text(encoding="utf-8")
    assert "#SBATCH --account=peilab" in text
    assert "#SBATCH --partition=normal" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --time=01:45:00" in text
    assert "run_id58_sft_checkpoint_state_matrix.sh" in text


def test_id58_runner_uses_fresh_output_and_explicit_server_python() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3" in text
    assert "VAGEN_ROLLOUT_BROWSER" not in text
    assert "sft_checkpoint_state_matrix" in text
    assert "--samples-per-source 32" in text
    assert "--max-step-index 3" in text
    assert "--batch-size 2" in text
    assert "--output-dir" in text
    assert "training is forbidden" in text
    assert "mkdir \"${RUN_OUT}\"" in text
    assert "mkdir -p \"${RUN_OUT}\"" not in text
