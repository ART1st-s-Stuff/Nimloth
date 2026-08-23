from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / "experiments/training/sft2/id60_goal_probe_id75_residual_t1_canary.slurm"
RUNNER = ROOT / "experiments/training/sft2/run_id60_goal_probe_id75_residual_t1_canary.sh"


def test_slurm_uses_approved_single_h800_contract() -> None:
    text = SLURM.read_text(encoding="utf-8")
    assert "#SBATCH --account=peilab" in text
    assert "#SBATCH --partition=normal" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --time=01:00:00" in text
    assert "dgx-09,dgx-13,dgx-32,dgx-51" in text


def test_runner_locks_sources_freeze_boundary_and_sequential_identity() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3" in text
    assert "176_id74_action_head_repair_balanced271x8_val40x8/checkpoint" in text
    assert "3_k16_ep5_untied_lm_head_restore/hf_merged" in text
    assert "train_terminal_cot_migrated.jsonl" in text
    assert "val_terminal_cot_migrated.jsonl" in text
    assert "nimloth.eval.frozen_state_goal_probe" in text
    assert "nimloth.training.sft2.residual_t1_canary" in text
    assert text.index("nimloth.eval.frozen_state_goal_probe") < text.index(
        "nimloth.training.sft2.residual_t1_canary"
    )
    assert "nimloth-recon-id60-frozen-state-goal-probe" in text
    assert "nimloth-sft2-id75-frozen-sft1-residual-t1-canary" in text
    assert "WANDB_PROJECT=\"${PROBE_WANDB_PROJECT}\"" in text
    assert "WANDB_PROJECT=\"${CANARY_WANDB_PROJECT}\"" in text
    assert "[[ ! -e \"${PROBE_OUT}\" ]]" in text
    assert "[[ ! -e \"${CANARY_OUT}\" ]]" in text
    assert "--state-cache \"${PROBE_OUT}/frozen_state_cache.npz\"" in text
    assert "--minimum-primary-count 20" in text
    assert "--raw-dino-loss-weight 0" in text
