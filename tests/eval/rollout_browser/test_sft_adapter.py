from __future__ import annotations

from pathlib import Path

from nimloth.agent.policy import PlannerPolicyTrace
from nimloth.eval.rollout_browser.sft_adapter import rollout_trajectory_artifact
from nimloth.rollout.schema import RolloutTrajectory


def _trajectory(tmp_path: Path, *, planner: bool) -> RolloutTrajectory:
    traces = []
    if planner:
        traces = [
            PlannerPolicyTrace(
                candidate_sequences=tuple((i, (i + 1) % 8) for i in range(8)),
                candidate_scores=(0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0),
                root_action_scores=(0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0),
                executed_action_index=0,
                horizon=2,
                search_mode="mcts",
                candidate_visit_counts=(3, 1, 1, 1, 1, 1, 1, 1),
                root_visit_counts=(3, 1, 1, 1, 1, 1, 1, 1),
                num_simulations=10,
                exploration_constant=1.0,
            )
        ]
    return RolloutTrajectory(
        record_id="sft-episode-1",
        reward_provenance="navigation",
        image_paths=["step0.png", "terminal.png"],
        action_indices=[0],
        action_names=["move_forward"],
        action_log_probs=[[-0.1, -2.35, -3.0, -3.1, -3.2, -3.3, -3.4, -3.5]],
        instruction="navigate to the toaster",
        success=False,
        reward=0.2,
        rewards=[0.01],
        terminated=False,
        truncated=True,
        split="test",
        observation_texts=["initial", "terminal"],
        assistant_responses=["<think>move forward</think><action>"],
        planner_policy_traces=traces,
        action_space_id="navigation",
        action_space_version=1,
    )


def test_sft_adapter_preserves_task_images_and_capability_absence(
    tmp_path: Path,
) -> None:
    artifact = rollout_trajectory_artifact(
        _trajectory(tmp_path, planner=False),
        policy_family="sft1",
        image_root=tmp_path,
    )
    audit = artifact.audit
    assert audit["task"] == "navigate to the toaster"
    assert audit["capabilities"]["planner"] is False
    assert audit["capabilities"]["direct_q"] is False
    assert "planner" not in audit["turns"][0]
    assert artifact.image_sources == {
        "step_00_observation.png": tmp_path / "step0.png",
        "terminal_observation.png": tmp_path / "terminal.png",
    }


def test_sft_action_only_response_is_preserved_without_fake_cot(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory(tmp_path, planner=False)
    trajectory.assistant_responses = ["<|action_start|><|action_(0)|><|action_end|>"]
    artifact = rollout_trajectory_artifact(
        trajectory,
        policy_family="sft1_action_only",
        image_root=tmp_path,
    )
    assert artifact.audit["capabilities"]["cot"] is False
    assert artifact.audit["turns"][0]["cot"] is None
    assert artifact.audit["turns"][0]["raw_response"].startswith(
        "<|action_start|>"
    )


def test_sft_mcts_adapter_preserves_every_candidate_and_visit(
    tmp_path: Path,
) -> None:
    artifact = rollout_trajectory_artifact(
        _trajectory(tmp_path, planner=True),
        policy_family="sft2_mcts",
        image_root=tmp_path,
    )
    audit = artifact.audit
    assert audit["capabilities"]["planner"] is True
    assert audit["capabilities"]["mcts"] is True
    planner = audit["turns"][0]["planner"]
    assert len(planner["candidates"]) == 8
    assert [row["visits"] for row in planner["candidates"]] == [3, 1, 1, 1, 1, 1, 1, 1]
    assert planner["root_visits"] == [3, 1, 1, 1, 1, 1, 1, 1]
