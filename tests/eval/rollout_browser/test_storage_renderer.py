from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from nimloth.eval.rollout_browser.sft_adapter import rollout_trajectory_artifact
from nimloth.eval.rollout_browser.storage import (
    finalize_evaluation_browser,
    merge_evaluation_browsers,
    write_evaluation_browser,
    write_evaluation_browser_batch,
)
from nimloth.rollout.schema import RolloutTrajectory


def _artifact(tmp_path: Path, record_id: str, success: bool):
    for name, color in ((f"{record_id}-0.png", "red"), (f"{record_id}-1.png", "blue")):
        Image.new("RGB", (8, 8), color=color).save(tmp_path / name)
    trajectory = RolloutTrajectory(
        record_id=record_id,
        reward_provenance="navigation",
        image_paths=[f"{record_id}-0.png", f"{record_id}-1.png"],
        action_indices=[0],
        action_names=["move_forward"],
        action_log_probs=[[-0.1, -2.35, -3.0, -3.1, -3.2, -3.3, -3.4, -3.5]],
        instruction=f"task for {record_id}",
        success=success,
        reward=1.01 if success else 0.01,
        rewards=[1.01 if success else 0.01],
        terminated=success,
        truncated=not success,
        split="test",
        observation_texts=["initial", "terminal"],
        assistant_responses=["<think>real cot</think><action>"],
    )
    return rollout_trajectory_artifact(
        trajectory,
        policy_family="sft1",
        image_root=tmp_path,
    )


def test_writer_atomically_builds_selectable_offline_browser(tmp_path: Path) -> None:
    destination = tmp_path / "browser"
    artifacts = [
        _artifact(tmp_path, "episode-a", False),
        _artifact(tmp_path, "episode-b", True),
    ]
    manifest = write_evaluation_browser(
        destination,
        artifacts,
        evaluation={
            "evaluation_id": "sft1-test",
            "policy_family": "sft1",
            "global_step": None,
            "source_step": None,
            "checkpoint_identity": "sha256:checkpoint",
            "snapshot_identity": None,
        },
        expected_rollouts=2,
    )
    assert manifest["status"] == "complete"
    assert manifest["rollout_count"] == 2
    assert manifest["summary"]["success_count"] == 1
    assert (destination / "complete.json").is_file()
    index = (destination / "index.html").read_text()
    assert "Rollout Browser" in index
    assert "search" in index
    assert "iframe" in index
    assert "task for episode-a" in index
    rollout_pages = sorted(
        (destination / "batches" / "batch_0000" / "rollouts").glob(
            "*/index.html"
        )
    )
    assert len(rollout_pages) == 2
    page = rollout_pages[0].read_text()
    assert "Task" in page
    assert "task for episode-" in page
    assert "Current state value" not in page
    stored = json.loads((rollout_pages[0].parent / "rollout.json").read_text())
    assert stored["task"].startswith("task for")
    assert (rollout_pages[0].parent / "step_00_observation.png").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        write_evaluation_browser(
            destination,
            artifacts,
            evaluation=manifest,
            expected_rollouts=2,
        )


def test_multi_batch_writer_publishes_only_after_global_identity_gate(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "browser"
    write_evaluation_browser_batch(
        destination, [_artifact(tmp_path, "episode-a", False)], batch_index=0
    )
    assert not (destination / "complete.json").exists()
    write_evaluation_browser_batch(
        destination, [_artifact(tmp_path, "episode-b", True)], batch_index=1
    )
    manifest = finalize_evaluation_browser(
        destination,
        evaluation={
            "evaluation_id": "two-batches",
            "policy_family": "sft1",
            "global_step": None,
            "source_step": None,
            "checkpoint_identity": "sha256:checkpoint",
            "snapshot_identity": None,
        },
        expected_rollouts=2,
        expected_batches=2,
    )
    assert manifest["rollout_count"] == 2
    assert (destination / "complete.json").is_file()


def test_merge_builds_one_selector_over_complete_shard_browsers(
    tmp_path: Path,
) -> None:
    sources = []
    for index in range(2):
        source = tmp_path / "eval_sets" / f"shard-{index}" / "evaluation_browser"
        write_evaluation_browser(
            source,
            [_artifact(tmp_path, f"episode-{index}", index == 1)],
            evaluation={
                "evaluation_id": f"shard-{index}",
                "policy_family": "sft2_mcts",
                "global_step": None,
                "source_step": None,
                "checkpoint_identity": "sha256:checkpoint",
                "snapshot_identity": "sha256:planner",
            },
            expected_rollouts=1,
        )
        sources.append(source)
    destination = tmp_path / "evaluation_browser"
    manifest = merge_evaluation_browsers(
        destination,
        sources,
        evaluation={
            "evaluation_id": "combined",
            "policy_family": "sft2_mcts",
            "global_step": None,
            "source_step": None,
            "checkpoint_identity": "sha256:checkpoint",
            "snapshot_identity": "sha256:planner",
        },
        expected_rollouts=2,
    )
    assert manifest["rollout_count"] == 2
    assert "../eval_sets/" in (destination / "index.html").read_text()
    assert (destination / "complete.json").is_file()


def test_writer_does_not_publish_partial_root_on_count_failure(tmp_path: Path) -> None:
    destination = tmp_path / "browser"
    with pytest.raises(ValueError, match="rollout count"):
        write_evaluation_browser(
            destination,
            [_artifact(tmp_path, "episode-a", False)],
            evaluation={
                "evaluation_id": "bad",
                "policy_family": "sft1",
                "global_step": None,
                "source_step": None,
                "checkpoint_identity": "sha256:checkpoint",
                "snapshot_identity": None,
            },
            expected_rollouts=2,
        )
    assert not destination.exists()
