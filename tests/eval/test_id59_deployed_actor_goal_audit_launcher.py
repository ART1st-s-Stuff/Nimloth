from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT / "experiments/training/sft2/id59_deployed_actor_sft1_goal_audit.slurm"
RUNNER = ROOT / "experiments/training/sft2/run_id59_deployed_actor_sft1_goal_audit.sh"


def test_id59_slurm_contract() -> None:
    text = SLURM.read_text(encoding="utf-8")
    assert "#SBATCH --account=peilab" in text
    assert "#SBATCH --partition=normal" in text
    assert "#SBATCH --exclude=dgx-09,dgx-13,dgx-32,dgx-51" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --time=00:30:00" in text


def test_id59_runner_is_readonly_and_locks_wandb_identity() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3" in text
    assert "RUN_WANDB_PROJECT=nimloth-recon" in text
    assert '--wandb-project "${RUN_WANDB_PROJECT}"' in text
    assert 'export WANDB_PROJECT="${RUN_WANDB_PROJECT}"' in text
    assert "--actor-checkpoint" in text
    assert "176_id74_action_head_repair" in text
    assert "--train-jsonl" in text and "--val-jsonl" in text
    assert "training is forbidden" in text
    assert "mkdir \"${RUN_OUT}\"" in text
    assert "mkdir -p \"${RUN_OUT}\"" not in text
