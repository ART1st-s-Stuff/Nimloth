from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / "experiments/training/reconstruction/id54_id189_source20_base2_id45cfm.slurm"
RUNNER = ROOT / "experiments/training/reconstruction/run_id54_id189_source20_base2_id45cfm.sh"
CONTRACT = ROOT / "ai_tasks/id54_id189_source20_base2_id45cfm_contract.md"


def test_id54_is_frozen_pre_rl_cfm_derived_browser() -> None:
    slurm = SLURM.read_text()
    runner = RUNNER.read_text()
    contract = CONTRACT.read_text()
    assert "#SBATCH --partition=normal" in slurm
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "#SBATCH --time=00:30:00" in slurm
    assert "navigation_base_test_id187" in runner
    assert "--seed 2" in runner
    assert "--steps 50" in runner
    assert "--cfg-scale 2" in runner
    assert "5f029ba4cdf1077d49377100c43d9ac836d89386e0ac049c4b92e0b0a7744dfa" in runner
    assert "6d555cd81141f280d3b7b1de5ad1972cea5456c13c2c0334ac4861dabb27de60" in runner
    assert "training_uses_rl_data'] is False" in runner
    assert "not list(run.rglob('checkpoint*.pt'))" in runner
    assert "No RL or post-RL data" in contract
    assert "All modules are frozen" in contract
