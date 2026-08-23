from __future__ import annotations

import numpy as np
import pytest

from nimloth.eval.frozen_state_goal_probe import (
    aggregate_task_probe_features,
    goal_probe_gate,
    parse_source_row_metadata,
)


def test_parse_source_row_metadata_is_diagnostic_and_validates_seed() -> None:
    identity = parse_source_row_metadata(
        config_id=(
            "NavigationEnvConfig(eval_set=common_sense_train,"
            "render_mode=vision,max_actions_per_step=1)"
        ),
        migrated_seed=1082,
        source_seed=1082,
    )
    assert identity == ("common_sense_train", 1082)
    with pytest.raises(ValueError, match="seed"):
        parse_source_row_metadata(
            config_id="NavigationEnvConfig(eval_set=base_train)",
            migrated_seed=1,
            source_seed=2,
        )


def test_aggregate_task_probe_features_averages_duplicate_task_rows() -> None:
    features = np.asarray([[1.0, 3.0], [3.0, 5.0], [9.0, 7.0]], dtype=np.float32)
    result = aggregate_task_probe_features(
        features=features,
        task_keys=np.asarray(["image1:Mug", "image1:Mug", "image2:Chair"]),
        labels=np.asarray(["Mug", "Mug", "Chair"]),
        leakage_group_keys=np.asarray(["image1", "image1", "image2"]),
    )
    assert result.task_keys.tolist() == ["image1:Mug", "image2:Chair"]
    assert result.leakage_group_keys.tolist() == ["image1", "image2"]
    np.testing.assert_allclose(result.features, [[2.0, 4.0], [9.0, 7.0]])
    assert result.labels.tolist() == ["Mug", "Chair"]


def test_aggregate_task_probe_features_rejects_inconsistent_goal() -> None:
    with pytest.raises(ValueError, match="multiple goal labels"):
        aggregate_task_probe_features(
            features=np.ones((2, 3), dtype=np.float32),
            task_keys=np.asarray(["same-observation-task", "same-observation-task"]),
            labels=np.asarray(["Mug", "Chair"]),
        )


def test_goal_probe_gate_requires_state_to_beat_visual_baseline() -> None:
    passing = goal_probe_gate(
        state_micro_top1=0.42,
        state_macro_top1=0.38,
        dino_micro_top1=0.37,
        dino_macro_top1=0.34,
        majority_top1=0.20,
        paired_bootstrap_lower=0.01,
    )
    assert passing["passed"] is True
    failing = goal_probe_gate(
        state_micro_top1=0.38,
        state_macro_top1=0.35,
        dino_micro_top1=0.37,
        dino_macro_top1=0.34,
        majority_top1=0.20,
        paired_bootstrap_lower=-0.01,
    )
    assert failing["passed"] is False
    assert failing["micro_margin"] is False
