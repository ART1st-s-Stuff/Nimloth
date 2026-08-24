from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / "experiments/training/sft2/id191_state_interface_direction_canary.slurm"
RUNNER = ROOT / "experiments/training/sft2/run_id191_state_interface_direction_canary.sh"


def test_id191_launcher_locks_direction_canary_contract() -> None:
    slurm = SLURM.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "#SBATCH --account=peilab" in slurm
    assert "#SBATCH --partition=normal" in slurm
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "#SBATCH --time=00:45:00" in slurm
    assert "dgx-09,dgx-13,dgx-32,dgx-51" in slurm
    assert "nimloth.training.sft2.id191_state_interface_canary" in runner
    assert "191_state_interface_direction_canary" in runner
    assert "nimloth-sft2-id191-state-interface-canary" in runner
    assert '[[ ! -e "${OUT}" ]]' in runner
    assert "--adapter-rank 64" in runner
    assert "--max-residual-fraction 0.1" in runner
    assert "--anchor-weight 0.25" in runner
    assert "2e2e1675317d252bc6e503ac78507328c81bf1925aceb487ed8e506f8b70c113" in runner
    assert runner.index("source /project/peilab/atst/flower/.env") < runner.index(
        'export WANDB_PROJECT="${RUN_WANDB_PROJECT}"'
    )
