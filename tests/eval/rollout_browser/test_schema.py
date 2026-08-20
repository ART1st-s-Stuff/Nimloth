from __future__ import annotations

import copy

import pytest

from nimloth.eval.rollout_browser.schema import (
    ROLLOUT_AUDIT_SCHEMA,
    validate_rollout_audit,
)


def _audit() -> dict:
    return {
        "schema": ROLLOUT_AUDIT_SCHEMA,
        "identity": {
            "rollout_sample_id": "sha256:sample",
            "rollout_repeat_index": 0,
            "record_id": "episode-1",
        },
        "policy_family": "vagen_k4_joint",
        "capabilities": {
            "task": True,
            "observations": True,
            "terminal_observation": True,
            "cot": True,
            "token_trace": False,
            "action_distribution": True,
            "direct_q": True,
            "state_value": True,
            "planner": True,
            "mcts": True,
            "model_state": False,
            "mcts_process": False,
        },
        "task": "navigate to the toaster",
        "data_source": "navigation_base",
        "seed": 2,
        "split": "test",
        "success": False,
        "reward": 0.2,
        "terminated": False,
        "truncated": True,
        "stop_reason": "task_failure",
        "turn_count": 1,
        "provenance": {"global_step": 20, "source_step": 796},
        "turns": [
            {
                "turn_index": 0,
                "observation": {
                    "text": "initial observation",
                    "image": "step_00_observation.png",
                    "sha256": "sha256:image",
                },
                "raw_response": "<think>go forward</think><action/>",
                "cot": "<think>go forward</think>",
                "executed_action": {"id": 0, "name": "move_forward"},
                "environment": {
                    "reward": 0.01,
                    "terminated": False,
                    "truncated": True,
                    "stop_reason": "task_failure",
                },
                "action_distribution": {
                    "kind": "guided_policy",
                    "log_probabilities": [-0.1, -2.35],
                },
                "direct_q": {
                    "values": [0.4, 0.1],
                    "state_value": 0.37,
                },
                "planner": {
                    "search_mode": "mcts",
                    "horizon": 4,
                    "num_simulations": 2,
                    "exploration_constant": 1.0,
                    "root_scores": [0.5, 0.2],
                    "root_visits": [1, 1],
                    "candidates": [
                        {
                            "action_ids": [0, 1, 0, 1],
                            "actions": [
                                "move_forward",
                                "turn_left",
                                "move_forward",
                                "turn_left",
                            ],
                            "score": 0.5,
                            "visits": 1,
                        },
                        {
                            "action_ids": [1, 0, 1, 0],
                            "actions": [
                                "turn_left",
                                "move_forward",
                                "turn_left",
                                "move_forward",
                            ],
                            "score": 0.2,
                            "visits": 1,
                        },
                    ],
                },
                "terminal": {
                    "observation": {
                        "text": "terminal observation",
                        "image": "terminal_observation.png",
                        "sha256": "sha256:terminal",
                    },
                    "raw_response": "<think>terminal</think>",
                    "cot": "<think>terminal</think>",
                    "stop_reason": "task_failure",
                    "action_executed": False,
                },
            }
        ],
    }


def test_rollout_audit_accepts_complete_behavior_evidence() -> None:
    validate_rollout_audit(_audit())


def test_rollout_audit_rejects_capability_without_evidence() -> None:
    audit = _audit()
    del audit["turns"][0]["planner"]
    with pytest.raises(ValueError, match="planner capability"):
        validate_rollout_audit(audit)


def test_rollout_audit_rejects_nonfinite_or_truncated_candidates() -> None:
    nonfinite = _audit()
    nonfinite["turns"][0]["planner"]["candidates"][0]["score"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_rollout_audit(nonfinite)

    truncated = copy.deepcopy(_audit())
    truncated["turns"][0]["planner"]["candidates"].pop()
    with pytest.raises(ValueError, match="candidate visits"):
        validate_rollout_audit(truncated)
