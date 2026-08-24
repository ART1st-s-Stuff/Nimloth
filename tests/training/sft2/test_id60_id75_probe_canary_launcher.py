from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SLURM = ROOT / "experiments/training/sft2/id60_goal_probe_id75_residual_t1_canary.slurm"
RUNNER = ROOT / "experiments/training/sft2/run_id60_goal_probe_id75_residual_t1_canary.sh"
RETRY_SLURM = ROOT / "experiments/training/sft2/id75_residual_t1_canary_retry1.slurm"
RETRY_RUNNER = ROOT / "experiments/training/sft2/run_id75_residual_t1_canary_retry1.sh"


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
    assert "mkdir -p" in text
    assert '"$(dirname "${CANARY_OUT}")"' in text


def test_retry_runs_only_id75_from_exact_completed_id60_cache() -> None:
    slurm = RETRY_SLURM.read_text(encoding="utf-8")
    runner = RETRY_RUNNER.read_text(encoding="utf-8")
    assert "#SBATCH --account=peilab" in slurm
    assert "#SBATCH --partition=normal" in slurm
    assert "#SBATCH --gres=gpu:1" in slurm
    assert "#SBATCH --time=00:30:00" in slurm
    assert "dgx-09,dgx-13,dgx-32,dgx-51" in slurm
    assert "nimloth.eval.frozen_state_goal_probe" not in runner
    assert "nimloth.training.sft2.residual_t1_canary" in runner
    assert "0fa994139d038d7f89b5a02a83d9036f9367b34a25f25e6b8cb84204f0daf8b6" in runner
    assert "b25163d390930d1ccdc172e4f4401a97cbea3dba561a8ec25ef33b9a09911682" in runner
    assert "37243c37e265691cc0cd3acdbc03a35241661a765ce8efc5fb2b6a7995bcd0ea" in runner
    assert "nimloth-sft2-id75-frozen-sft1-residual-t1-canary-retry1" in runner
    assert "75_frozen_sft1_residual_t1_canary_early4_retry1" in runner
    assert 'mkdir -p "$(dirname "${CANARY_OUT}")"' in runner
    assert '[[ ! -e "${CANARY_OUT}" ]]' in runner
    assert "--raw-dino-loss-weight 0" in runner
