from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / "experiments/training/sft2/id71_action_outcome_predictability_probe.slurm"
RUNNER = ROOT / "experiments/training/sft2/run_id71_action_outcome_predictability_probe.sh"


def test_id71_launcher_locks_readout_only_single_h800_contract() -> None:
    slurm = SLURM.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "#SBATCH --account=peilab" in slurm
    assert "#SBATCH --partition=normal" in slurm
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "#SBATCH --time=00:10:00" in slurm
    assert "dgx-09,dgx-13,dgx-32,dgx-51" in slurm
    assert "nimloth.eval.action_outcome_predictability_probe" in runner
    assert "71_frozen_state_action_outcome_probe" in runner
    assert "nimloth-recon-id71-action-outcome-probe" in runner
    assert '[[ ! -e "${OUT}" ]]' in runner
    assert "RUN_WANDB_PROJECT=nimloth-recon" in runner
    assert runner.index("source /project/peilab/atst/flower/.env") < runner.index(
        'export WANDB_PROJECT="${RUN_WANDB_PROJECT}"'
    )
    assert "--bootstrap-draws 10000" in runner
    assert "bace6fcbc5ec85fdeed59e6ba30ff61b58bbe382f88af51f9dd591a8105a28e4" in runner
