from pathlib import Path

import pytest

from nimloth.training.sft1.experiment_config import load_sft1_v2_config
from nimloth.training.sft1.launch_config import (
    SFT1V2LaunchResolution,
    resolve_launch_config,
)


ROOT = Path(__file__).resolve().parents[3]


def test_early4_report_first_yaml_is_strict_and_deliberately_launch_locked() -> None:
    config = load_sft1_v2_config(
        ROOT / "configs/training/sft1/state_interface_v2_early4_report_first.yaml"
    )
    assert config.selection.steps == (0, 1, 2, 3)
    assert (config.selection.train_rows, config.selection.raw_validation_rows) == (12836, 1420)
    assert config.selection.external_validation_rows == 1413
    assert config.selection.excluded_train_empty_cot_rows == 5
    assert config.selection.excluded_validation_empty_cot_rows == 0
    assert (
        config.data.overlap_key
        == "record_initial_and_current_next_original_image_sha256"
    )
    assert config.optimizer.scheduler == "none"
    assert config.runtime.epochs == 3
    assert config.validation.report_first is True
    assert config.output.overwrite is False
    assert config.runtime.launch_locked is False
    assert "LOCK_BEFORE_LAUNCH" in config.output.run_dir
    assert config.runtime.max_rows_per_micro_batch == 2
    assert config.runtime.max_padded_tokens >= 2 * config.runtime.max_sequence_length


def test_launch_resolution_requires_every_identity_and_preserves_contrastive_batch(
    tmp_path: Path,
) -> None:
    template = ROOT / "configs/training/sft1/state_interface_v2_early4_report_first.yaml"
    values = dict(
        repo="/server/worktree",
        expected_commit="a" * 40,
        interpreter="/server/.venv-vagen-main/bin/python3",
        cache_output_dir="/outputs/experiments/training/sft1_state_interface_v2/193/cache",
        run_dir="/outputs/experiments/training/sft1_state_interface_v2/193/formal",
        wandb_run_name="193_sft1v2_early4",
        wandb_run_id="nimloth-sft1-id193-sft1v2-early4",
        minimum_free_bytes=100_000_000_000,
        processor_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        prompt_template_sha256="3" * 64,
        token_table_sha256="4" * 64,
        world_size=4,
        max_sequence_length=12000,
        max_padded_tokens=24000,
        max_rows_per_micro_batch=2,
        rows_per_rank_update=2,
        teacher_batch_size=8,
        checkpoint_cadence_steps=100,
    )
    config = resolve_launch_config(
        template,
        tmp_path / "resolved.json",
        SFT1V2LaunchResolution(**values),
    )
    assert config.runtime.launch_locked
    assert config.source.expected_commit == "a" * 40
    assert config.runtime.world_size == 4
    assert config.output.wandb_run_id == values["wandb_run_id"]
    persisted = load_sft1_v2_config(tmp_path / "resolved.json")
    assert persisted.identity == config.identity

    values["max_rows_per_micro_batch"] = 1
    with pytest.raises(ValueError, match="at least two rows"):
        resolve_launch_config(
            template,
            tmp_path / "invalid.json",
            SFT1V2LaunchResolution(**values),
        )
