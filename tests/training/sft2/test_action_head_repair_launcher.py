from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "experiments/training/sft2/run_id74_action_head_repair.sh"
SLURM = ROOT / "experiments/training/sft2/id176_action_head_repair.slurm"
ENTRYPOINT = ROOT / "src/nimloth/training/sft2/action_head_repair_cli.py"


def test_id176_launcher_locks_low_cost_repair_contract() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    slurm = SLURM.read_text(encoding="utf-8")
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")

    assert "--time=01:30:00" in slurm
    assert "--gres=gpu:8" in slurm
    assert "--cpus-per-task=64" in slurm
    assert "--mem=256G" in slurm
    assert "#SBATCH --requeue" in slurm
    assert "MinMemoryNode=256G" in slurm
    assert "AllocTRES=" in slurm
    assert "EXPECTED_COMMIT" in slurm
    assert "RUN_NAME=176_id74_action_head_repair_balanced271x8_val40x8" in slurm

    for value in (
        "TRAIN_EXAMPLES_PER_ACTION=271",
        "VALIDATION_EXAMPLES_PER_ACTION=40",
        "SELECTION_SEED=42002",
        "FIT_LEARNING_RATE=0.0001",
        "FIT_WEIGHT_DECAY=0.0",
        "FIT_MAX_EPOCHS=500",
        "FIT_EARLY_STOPPING_PATIENCE=50",
        "MINIMUM_VALIDATION_NLL_IMPROVEMENT=0.05",
        "MINIMUM_BF16_MEDIAN_SPREAD=0.001",
        "EXPECTED_WORLD_SIZE=8",
    ):
        assert value in slurm

    assert "resume-mode allow-exact" in runner
    assert "--extraction-batch-size 1" in runner
    assert "--latent-token-count 16" in runner
    assert "--expected-action-count 8" in runner
    assert "model.safetensors.index.json" in entrypoint
    assert "train_terminal_cot_migrated.jsonl" in runner
    assert "val_terminal_cot_migrated.jsonl" in runner
    assert "3e501a0ccee9193676d69dd3590ae0d592c4fdee298810df2abff47d9f36a943" in runner
    assert "acd10994cff947c365f95da69d81219fde0e97a30a7f574bb395e8169b93da58" in runner
    assert "optimizer" not in slurm.lower()
    assert "wandb" not in slurm.lower()


def test_id176_validator_preserves_id74_non_action_components() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    for component in (
        "qwen_transformer",
        "qwen_vision",
        "all_non_action_lm_head_rows",
        "state_proj",
        "wm_predictor",
        "value_head",
    ):
        assert component in runner
    assert "validation_nll_improvement_fp32" in runner
    assert "validation_bfloat16_median_action_spread" in runner
    assert "train_step_log.csv" in runner
    assert "complete.marker" in runner
