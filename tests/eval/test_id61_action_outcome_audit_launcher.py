from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / "experiments/training/sft2/id61_id75_action_outcome_audit.slurm"
RUNNER = ROOT / "experiments/training/sft2/run_id61_id75_action_outcome_audit.sh"


def test_id61_launcher_is_read_only_fresh_single_h800() -> None:
    slurm = SLURM.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "#SBATCH --account=peilab" in slurm
    assert "#SBATCH --partition=normal" in slurm
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "#SBATCH --time=00:15:00" in slurm
    assert "dgx-09,dgx-13,dgx-32,dgx-51" in slurm
    assert "nimloth.eval.id75_action_outcome_audit" in runner
    assert "nimloth-recon-id61-id75-action-outcome-audit" in runner
    assert "61_id75_action_outcome_audit" in runner
    assert '[[ ! -e "${OUT}" ]]' in runner
    assert "--bootstrap-draws 10000" in runner
    assert "0fa994139d038d7f89b5a02a83d9036f9367b34a25f25e6b8cb84204f0daf8b6" in runner
    assert "7c79cb06b8a349ab96cda9064541917ef97702be32e7d1b8f7eb2392122804ad" in runner
    assert "optimizer" not in runner.lower().replace("no optimizer update", "")
