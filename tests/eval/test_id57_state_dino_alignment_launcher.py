from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / "experiments/training/reconstruction/id57_id189_state_dino_alignment.slurm"
RUNNER = ROOT / "experiments/training/reconstruction/run_id57_id189_state_dino_alignment.sh"
CONTRACT = ROOT / "ai_tasks/id57_id189_state_dino_alignment_contract.md"


def test_id57_is_readonly_state_dino_alignment() -> None:
    slurm = SLURM.read_text()
    runner = RUNNER.read_text()
    contract = CONTRACT.read_text()
    assert "#SBATCH --account=peilab" in slurm
    assert "#SBATCH --partition=normal" in slurm
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "nimloth.eval.id189_state_dino_alignment" in runner
    assert "--expected-rollouts 120" in runner
    assert "--expected-turns 1862" in runner
    assert "--expected-transitions 1742" in runner
    assert "training_or_optimizer_update'] is False" in runner
    assert "model_replay'] is False" in runner
    assert "not list(run.rglob('checkpoint*.pt'))" in runner
    assert "actual current state" in contract
    assert "actual next state" in contract
    assert "No optimizer" in contract
    assert "does not fabricate" in contract
