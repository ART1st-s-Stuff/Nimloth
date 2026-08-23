from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nimloth.eval.deployed_actor_sft1_goal_audit import (
    build_instruction_goal_map,
    goal_retrieval_metrics,
    parse_actual_eval_set,
)


def test_parse_actual_eval_set_uses_environment_config() -> None:
    config = "NavigationEnvConfig(eval_set=common_sense_train,render_mode=vision,max_actions_per_step=1)"
    assert parse_actual_eval_set(config) == "common_sense_train"


def test_instruction_goal_map_requires_unique_ground_truth(tmp_path: Path) -> None:
    (tmp_path / "base_train.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {"instruction": "find the mug", "targetObjectType": "Mug"},
                    {"instruction": "find the mug", "targetObjectType": "Cup"},
                ]
            }
        ),
        encoding="utf-8",
    )
    try:
        build_instruction_goal_map(tmp_path, ("base_train",))
    except ValueError as error:
        assert "ambiguous" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ambiguous-label failure")


def test_goal_retrieval_excludes_exact_image_matches() -> None:
    gallery = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    result = goal_retrieval_metrics(
        gallery_embeddings=gallery,
        query_embeddings=query,
        gallery_labels=np.asarray(["wrong", "right", "other"]),
        query_labels=np.asarray(["right"]),
        gallery_image_sha256=np.asarray(["same", "different", "third"]),
        query_image_sha256=np.asarray(["same"]),
    )
    assert result["query_count"] == 1
    assert result["exact_image_candidates_excluded"] == 1
    assert result["top1_accuracy"] == 1.0
    assert result["top5_recall"] == 1.0
    assert result["mean_reciprocal_rank"] == 1.0
