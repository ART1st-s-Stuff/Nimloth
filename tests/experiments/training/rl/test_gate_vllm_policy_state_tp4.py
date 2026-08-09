"""Validation contracts for the real TP4 policy-state gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "experiments"
    / "training"
    / "rl"
    / "gate_vllm_policy_state_tp4.py"
)
_SPEC = importlib.util.spec_from_file_location("gate_vllm_policy_state_tp4", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GATE)


def _generation(logits: torch.Tensor, latent_offset: float = 0.0):
    behavior = torch.log_softmax(logits.float(), dim=-1)
    return SimpleNamespace(
        qwen_decision=SimpleNamespace(
            action_index=int(logits.argmax().item()),
            action_log_probs=tuple(float(value) for value in behavior),
            response="<think>x</think><|latent_state|><|action_start|>",
        ),
        policy_state=SimpleNamespace(
            latent_hidden=torch.arange(12, dtype=torch.float32).reshape(3, 4)
            + latent_offset,
            action_logits=logits,
        ),
    )


def test_gate_accepts_identity_aligned_distinct_requests() -> None:
    result = _GATE._validate_generations(
        (
            _generation(torch.tensor([1.0, 2.0])),
            _generation(torch.tensor([3.0, -1.0]), latent_offset=0.5),
        ),
        expected_count=2,
        latent_token_count=3,
        action_count=2,
        atol=1e-6,
        rtol=1e-6,
    )

    assert result["status"] == "ALL_OK"
    assert result["pairwise_action_logit_max_abs"] == 3.0
    assert result["pairwise_latent_max_abs"] == 0.5


def test_gate_rejects_state_swapped_away_from_behavior_distribution() -> None:
    first = _generation(torch.tensor([1.0, 2.0]))
    second = _generation(torch.tensor([3.0, -1.0]), latent_offset=0.5)
    first.policy_state.action_logits = second.policy_state.action_logits

    with pytest.raises(RuntimeError, match="do not reproduce request behavior"):
        _GATE._validate_generations(
            (first, second),
            expected_count=2,
            latent_token_count=3,
            action_count=2,
            atol=1e-6,
            rtol=1e-6,
        )
