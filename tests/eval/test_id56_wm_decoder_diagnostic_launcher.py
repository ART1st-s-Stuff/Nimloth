from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / "experiments/training/reconstruction/id56_id189_wm_decoder_diagnostic.slurm"
RUNNER = ROOT / "experiments/training/reconstruction/run_id56_id189_wm_decoder_diagnostic.sh"
CONTRACT = ROOT / "ai_tasks/id56_id189_wm_decoder_diagnostic_contract.md"


def test_id56_is_readonly_full_wm_decoder_separation() -> None:
    slurm = SLURM.read_text()
    runner = RUNNER.read_text()
    contract = CONTRACT.read_text()
    assert "#SBATCH --account=peilab" in slurm
    assert "#SBATCH --partition=normal" in slurm
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "#SBATCH --time=02:00:00" in slurm
    assert "nimloth.eval.id189_wm_decoder_diagnostic" in runner
    assert "--expected-rollouts 120" in runner
    assert "--expected-turns 1862" in runner
    assert "--expected-transitions 1742" in runner
    assert "--noise-seed-count 4" in runner
    assert "state_depth1_action_count']==8" in runner
    assert "training_or_optimizer_update'] is False" in runner
    assert "not list(run.rglob('checkpoint*.pt'))" in runner
    assert "No terminal state or CoT is fabricated" in contract
    assert "No optimizer" in contract
