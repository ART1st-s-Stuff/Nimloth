"""PPO behavior-policy ownership tests."""

from __future__ import annotations

import torch

from nimloth.training.rl.loss import compute_action_entropy
from nimloth.training.rl.rollout import (
    action_sampling_logits,
    serialize_action_log_probs,
)
from nimloth.training.rl.trainer import qwen_actor_batch_indices


def test_qwen_actor_batch_uses_only_real_qwen_behavior_samples() -> None:
    batch = [
        {"policy_source": "qwen", "old_log_prob": -0.4},
        {"policy_source": "wm_value", "old_log_prob": None},
        {"policy_source": "qwen", "old_log_prob": -1.2},
    ]
    assert qwen_actor_batch_indices(batch) == [0, 2]


def test_qwen_actor_batch_rejects_fake_or_missing_behavior_probabilities() -> None:
    assert qwen_actor_batch_indices([
        {"policy_source": "wm_value", "old_log_prob": -0.1},
        {"policy_source": "qwen", "old_log_prob": None},
    ]) == []


def test_behavior_log_probs_use_temperature_and_top_p_transform() -> None:
    raw = torch.tensor([3.0, 2.0, 1.0, 0.0], requires_grad=True)
    transformed = action_sampling_logits(raw, temperature=0.5, top_p=0.8)
    assert transformed[0].item() == 6.0
    assert torch.isneginf(transformed[1:]).all()
    entropy = compute_action_entropy(transformed.unsqueeze(0))
    assert torch.isfinite(entropy)
    assert entropy.item() == 0.0
    entropy.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    serialized = serialize_action_log_probs(torch.log_softmax(transformed, dim=-1))
    assert serialized[0] == 0.0
    assert serialized[1:] == [None, None, None]
