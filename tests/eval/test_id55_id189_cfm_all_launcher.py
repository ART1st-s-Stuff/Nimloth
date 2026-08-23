from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / "experiments/training/reconstruction/id55_id189_source20_all120_id45cfm.slurm"
RUNNER = ROOT / "experiments/training/reconstruction/run_id55_id189_source20_all120_id45cfm.sh"
CONTRACT = ROOT / "ai_tasks/id55_id189_source20_all120_id45cfm_contract.md"


def test_id55_full_cfm_is_frozen_exact_and_complete() -> None:
    slurm = SLURM.read_text()
    runner = RUNNER.read_text()
    contract = CONTRACT.read_text()
    assert "#SBATCH --account=peilab" in slurm
    assert "#SBATCH --partition=normal" in slurm
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "#SBATCH --time=00:30:00" in slurm
    assert "nimloth.eval.id189_cfm_all" in runner
    assert "--expected-rollouts 120" in runner
    assert "--expected-turns 1862" in runner
    assert "--steps 50" in runner and "--cfg-scale 2" in runner
    assert "training_uses_rl_data'] is False" in runner
    assert "len(list(run.glob('reconstructions/batches/batch_*/rollouts/*/turn_*_comparison.png')))==1862" in runner
    assert "not list(run.rglob('checkpoint*.pt'))" in runner
    assert "trained before RL" in contract
    assert "No optimizer" in contract
