"""Lossless browser adapter for existing Nimloth ``RolloutTrajectory`` records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nimloth.environment import get_action_space
from nimloth.rollout.schema import RolloutTrajectory

from .schema import ROLLOUT_AUDIT_SCHEMA


@dataclass(frozen=True)
class RolloutBrowserArtifact:
    """One audit plus source files that must be copied into its archive."""

    audit: dict[str, Any]
    image_sources: dict[str, Any]


def _cot(raw_response: str) -> str:
    boundary = raw_response.find("</think>")
    if not raw_response.startswith("<think>") or boundary < 0:
        raise ValueError("rollout response does not contain actual <think> CoT")
    return raw_response[: boundary + len("</think>")]


def _strict_log_probabilities(values: list[float] | tuple[float, ...]) -> list[float | None]:
    return [None if value == float("-inf") else float(value) for value in values]


def _source_path(path: str, image_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else image_root / candidate


def _token_trace_record(trace: Any) -> dict[str, Any]:
    return {
        "token_ids": list(trace.token_ids),
        "old_log_probabilities": list(trace.old_log_probs),
        "reference_log_probabilities": (
            list(trace.reference_log_probs)
            if trace.reference_log_probs is not None
            else None
        ),
        "loss_mask": list(trace.loss_mask),
        "token_roles": list(trace.token_roles),
        "action_token_ids": list(trace.action_token_ids),
        "reasoning_text": trace.reasoning_text,
        "finish_reason": trace.finish_reason,
        "reasoning_truncated": trace.reasoning_truncated,
    }


def _planner_record(trace: Any, action_names: tuple[str, ...]) -> dict[str, Any]:
    visits = trace.candidate_visit_counts
    candidates = []
    for index, (sequence, score) in enumerate(
        zip(trace.candidate_sequences, trace.candidate_scores, strict=True)
    ):
        candidates.append(
            {
                "action_ids": list(sequence),
                "actions": [action_names[action] for action in sequence],
                "score": float(score),
                "visits": int(visits[index]) if visits is not None else None,
            }
        )
    return {
        "search_mode": trace.search_mode,
        "horizon": trace.horizon,
        "num_simulations": trace.num_simulations,
        "exploration_constant": trace.exploration_constant,
        "root_scores": _strict_log_probabilities(trace.root_action_scores),
        "root_visits": (
            list(trace.root_visit_counts)
            if trace.root_visit_counts is not None
            else None
        ),
        "selection_mode": trace.selection_mode,
        "candidates": candidates,
    }


def rollout_trajectory_artifact(
    trajectory: RolloutTrajectory,
    *,
    policy_family: str,
    image_root: Path,
    data_source: str | None = None,
    seed: int | None = None,
    rollout_sample_id: str | None = None,
    rollout_repeat_index: int = 0,
    provenance: dict[str, Any] | None = None,
) -> RolloutBrowserArtifact:
    """Convert one persisted SFT/greedy/MCTS trajectory without recomputation."""

    if not policy_family.strip():
        raise ValueError("policy_family must be non-empty")
    steps = trajectory.num_steps
    if steps < 1:
        raise ValueError("rollout browser requires at least one action")
    aligned_fields = {
        "action_names": trajectory.action_names,
        "action_log_probs": trajectory.action_log_probs,
        "assistant_responses": trajectory.assistant_responses,
        "rewards": trajectory.rewards,
    }
    mismatches = {
        name: len(values)
        for name, values in aligned_fields.items()
        if len(values) != steps
    }
    if mismatches:
        raise ValueError(f"trajectory step fields are not aligned: {mismatches}")
    if len(trajectory.observation_texts) != steps + 1:
        raise ValueError("trajectory requires one terminal observation text")
    if len(trajectory.image_paths) != steps + 1:
        raise ValueError("trajectory requires one terminal observation image")
    action_space = get_action_space(
        trajectory.action_space_id,
        trajectory.action_space_version,
    )
    action_names = action_space.keys
    if any(len(row) != len(action_names) for row in trajectory.action_log_probs):
        raise ValueError("trajectory action distributions do not match action space")
    token_traces = [trajectory.policy_token_trace(index) for index in range(steps)]
    planner_traces = [trajectory.planner_policy_trace(index) for index in range(steps)]
    has_token_trace = all(trace is not None for trace in token_traces)
    has_planner = all(trace is not None for trace in planner_traces)
    if any(trace is not None for trace in token_traces) != has_token_trace:
        raise ValueError("partial token trace coverage cannot be advertised")
    if any(trace is not None for trace in planner_traces) != has_planner:
        raise ValueError("partial planner trace coverage cannot be advertised")
    if has_planner and any(
        trace is not None and len(trace.root_action_scores) != len(action_names)
        for trace in planner_traces
    ):
        raise ValueError("planner traces do not match trajectory action space")
    has_mcts = has_planner and all(
        trace is not None and trace.search_mode == "mcts"
        for trace in planner_traces
    )
    image_sources: dict[str, Path] = {}
    turns: list[dict[str, Any]] = []
    stop_reason = (
        "success"
        if trajectory.success
        else "environment_failure"
        if trajectory.terminated
        else "truncated"
        if trajectory.truncated
        else "completed"
    )
    for index in range(steps):
        image_name = f"step_{index:02d}_observation.png"
        image_sources[image_name] = _source_path(
            trajectory.image_paths[index], image_root
        )
        action_index = int(trajectory.action_indices[index])
        expected_name = action_space.key_for(action_index)
        recorded_name = trajectory.action_names[index]
        if action_space.index_for(recorded_name) != action_index:
            raise ValueError("trajectory action index/name mismatch")
        final = index == steps - 1
        turn: dict[str, Any] = {
            "turn_index": index,
            "observation": {
                "text": trajectory.observation_texts[index],
                "image": image_name,
                "sha256": "sha256:pending",
            },
            "raw_response": trajectory.assistant_responses[index],
            "cot": _cot(trajectory.assistant_responses[index]),
            "executed_action": {"id": action_index, "name": expected_name},
            "environment": {
                "reward": float(trajectory.rewards[index]),
                "terminated": bool(final and trajectory.terminated),
                "truncated": bool(final and trajectory.truncated),
                "stop_reason": stop_reason if final else "continue",
            },
            "action_distribution": {
                "kind": "behavior_policy",
                "log_probabilities": _strict_log_probabilities(
                    trajectory.action_log_probs[index]
                ),
            },
        }
        token_trace = token_traces[index]
        if token_trace is not None:
            turn["token_trace"] = _token_trace_record(token_trace)
        planner_trace = planner_traces[index]
        if planner_trace is not None:
            turn["planner"] = _planner_record(planner_trace, action_names)
        if final:
            terminal_name = "terminal_observation.png"
            image_sources[terminal_name] = _source_path(
                trajectory.image_paths[-1], image_root
            )
            terminal: dict[str, Any] = {
                "observation": {
                    "text": trajectory.observation_texts[-1],
                    "image": terminal_name,
                    "sha256": "sha256:pending",
                },
                "stop_reason": stop_reason,
                "action_executed": False,
            }
            if trajectory.terminal_assistant_prefix:
                terminal.update(
                    {
                        "raw_response": trajectory.terminal_assistant_prefix,
                        "cot": _cot(trajectory.terminal_assistant_prefix),
                    }
                )
            turn["terminal"] = terminal
        turns.append(turn)
    audit = {
        "schema": ROLLOUT_AUDIT_SCHEMA,
        "identity": {
            "rollout_sample_id": rollout_sample_id,
            "rollout_repeat_index": int(rollout_repeat_index),
            "record_id": trajectory.record_id,
        },
        "policy_family": policy_family,
        "action_space": {
            "id": trajectory.action_space_id,
            "version": trajectory.action_space_version,
            "names": list(action_names),
        },
        "capabilities": {
            "task": True,
            "observations": True,
            "terminal_observation": True,
            "cot": True,
            "token_trace": has_token_trace,
            "action_distribution": True,
            "direct_q": False,
            "state_value": False,
            "planner": has_planner,
            "mcts": has_mcts,
        },
        "task": trajectory.instruction,
        "data_source": data_source,
        "seed": seed,
        "split": trajectory.split,
        "success": bool(trajectory.success),
        "reward": float(trajectory.reward),
        "terminated": bool(trajectory.terminated),
        "truncated": bool(trajectory.truncated),
        "stop_reason": stop_reason,
        "turn_count": steps,
        "provenance": {
            "reward_provenance": trajectory.reward_provenance,
            "sampling_temperature": trajectory.sampling_temperature,
            "sampling_top_p": trajectory.sampling_top_p,
            **(provenance or {}),
        },
        "turns": turns,
    }
    return RolloutBrowserArtifact(audit=audit, image_sources=image_sources)


__all__ = ["RolloutBrowserArtifact", "rollout_trajectory_artifact"]
