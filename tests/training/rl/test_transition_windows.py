"""RL transition-window protocol and ownership tests."""

from __future__ import annotations

import torch
import pytest
from torch import nn

from nimloth.training.rl import trainer
from nimloth.training.rl.rollout import RolloutTrajectory


def _k8_wm_trajectory() -> RolloutTrajectory:
    return RolloutTrajectory(
        record_id="wm-k8",
        image_paths=["0.png", "1.png", "2.png", "3.png"],
        action_indices=[1, 2, 3],
        action_names=["moveback", "moveright", "moveleft"],
        action_log_probs=[None, None, None],
        policy_sources=["wm_value", "wm_value", "wm_value"],
        state_sources=["qwen_gt", "wm_predicted", "qwen_gt"],
        fast_path_steps=[0, 1, 0],
        rollout_policy="wm_value",
        fast_path_horizon=2,
        latent_token_count=8,
        latent_query_mode="inject",
        nav_instruction="Find the couch.",
        reward=10.0,
    )


class FlattenProjector(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.flatten(1)


class AddActionPredictor(nn.Module):
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return state + action.float().unsqueeze(1)


def test_build_transitions_preserves_contiguous_k8_windows(monkeypatch) -> None:
    hiddens = [torch.full((8, 2), float(step)) for step in range(4)]
    monkeypatch.setattr(
        trainer,
        "encode_trajectory_hiddens",
        lambda *args, **kwargs: hiddens,
    )

    transitions = trainer.build_rl_transitions(
        [_k8_wm_trajectory()],
        qwen_model=object(),
        processor=object(),
        token_id_map={},
        device=torch.device("cpu"),
        latent_token_count=8,
        rollout_steps=2,
        rollout_policy="wm_value",
    )

    assert len(transitions) == 3
    assert transitions[0]["action_sequence"].tolist() == [1, 2]
    assert transitions[0]["qwen_hidden_targets"].shape == (2, 8, 2)
    assert transitions[1]["action_sequence"].tolist() == [2, 3]
    assert transitions[2]["action_sequence"].tolist() == [3]
    assert transitions[0]["policy_source"] == "wm_value"
    assert transitions[0]["old_log_prob"] is None
    assert transitions[0]["image_history_paths"] == ["0.png"]
    assert transitions[1]["image_history_paths"] == ["0.png", "1.png"]
    assert transitions[0]["behavior_action_prefix"].tolist() == []
    assert transitions[1]["behavior_action_prefix"].tolist() == [1]
    assert transitions[2]["behavior_action_prefix"].tolist() == []

    behavior_states = trainer.reconstruct_behavior_wm_states(
        transitions,
        state_proj=FlattenProjector(),
        wm_predictor=AddActionPredictor(),
        device=torch.device("cpu"),
    )
    assert behavior_states.shape == (3, 16)
    assert torch.equal(behavior_states[0], torch.zeros(16))
    assert torch.equal(behavior_states[1], torch.ones(16))
    assert torch.equal(behavior_states[2], torch.full((16,), 2.0))


def test_hybrid_transition_batch_trains_qwen_only_on_segment_first_step(monkeypatch) -> None:
    trajectory = RolloutTrajectory(
        record_id="hybrid-k8",
        image_paths=["0.png", "1.png", "2.png"],
        action_indices=[1, 2],
        action_names=["moveback", "moveright"],
        action_log_probs=[[-0.5] * 8, None],
        policy_sources=["qwen", "wm_value"],
        state_sources=["qwen_gt", "wm_predicted"],
        fast_path_steps=[0, 1],
        rollout_policy="qwen_wm",
        fast_path_horizon=2,
        latent_token_count=8,
        latent_query_mode="inject",
        action_log_prob_semantics="sampling_distribution_v1",
        nav_instruction="Find the couch.",
        reward=10.0,
    )
    hiddens = [torch.full((8, 2), float(step)) for step in range(3)]
    monkeypatch.setattr(
        trainer,
        "encode_trajectory_hiddens",
        lambda *args, **kwargs: hiddens,
    )

    transitions = trainer.build_rl_transitions(
        [trajectory],
        qwen_model=object(),
        processor=object(),
        token_id_map={},
        device=torch.device("cpu"),
        latent_token_count=8,
        rollout_steps=2,
        rollout_policy="qwen_wm",
    )

    assert trainer.qwen_actor_batch_indices(transitions) == [0]
    assert transitions[0]["old_log_prob"] == -0.5
    assert transitions[1]["old_log_prob"] is None
    assert transitions[1]["behavior_action_prefix"].tolist() == [1]
    behavior_states = trainer.reconstruct_behavior_wm_states(
        transitions,
        state_proj=FlattenProjector(),
        wm_predictor=AddActionPredictor(),
        device=torch.device("cpu"),
    )
    assert torch.equal(behavior_states[0], torch.zeros(16))
    assert torch.equal(behavior_states[1], torch.ones(16))


def test_legacy_qwen_log_probs_are_not_silently_treated_as_ppo_behavior(monkeypatch) -> None:
    trajectory = RolloutTrajectory(
        record_id="legacy-qwen",
        image_paths=["0.png", "1.png"],
        action_indices=[0],
        action_names=["moveahead"],
        action_log_probs=[[-0.5] * 8],
        policy_sources=["qwen"],
        state_sources=["qwen_gt"],
        fast_path_steps=[0],
        rollout_policy="qwen",
        latent_token_count=1,
        latent_query_mode="inject",
        nav_instruction="Find the couch.",
    )
    monkeypatch.setattr(
        trainer,
        "encode_trajectory_hiddens",
        lambda *args, **kwargs: [torch.zeros(2), torch.ones(2)],
    )

    legacy = trainer.build_rl_transitions(
        [trajectory],
        qwen_model=object(),
        processor=object(),
        token_id_map={},
        device=torch.device("cpu"),
        latent_token_count=1,
        rollout_steps=1,
        rollout_policy="qwen",
    )
    assert legacy[0]["old_log_prob"] is None

    trajectory.action_log_prob_semantics = "sampling_distribution_v1"
    exact = trainer.build_rl_transitions(
        [trajectory],
        qwen_model=object(),
        processor=object(),
        token_id_map={},
        device=torch.device("cpu"),
        latent_token_count=1,
        rollout_steps=1,
        rollout_policy="qwen",
    )
    assert exact[0]["old_log_prob"] == -0.5


def test_build_transitions_rejects_k_mismatch_before_encoding(monkeypatch) -> None:
    monkeypatch.setattr(
        trainer,
        "encode_trajectory_hiddens",
        lambda *args, **kwargs: pytest.fail("must reject before Qwen encoding"),
    )
    with pytest.raises(ValueError, match="protocol mismatch"):
        trainer.build_rl_transitions(
            [_k8_wm_trajectory()],
            qwen_model=object(),
            processor=object(),
            token_id_map={},
            device=torch.device("cpu"),
            latent_token_count=1,
            rollout_steps=2,
            rollout_policy="wm_value",
        )
