from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn


class _LanguageModel(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(64, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)


class _InnerModel(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.language_model = _LanguageModel(hidden_dim)


class _Actor(nn.Module):
    def __init__(self, hidden_dim: int = 4) -> None:
        super().__init__()
        self.model = _InnerModel(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, 64, bias=False)

    def forward(self, input_ids, **_kwargs):
        hidden = self.model.language_model.norm(
            self.model.language_model.embed_tokens(input_ids)
        )
        return SimpleNamespace(logits=self.lm_head(hidden))


class _Predictor(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(emb_dim, emb_dim)

    def forward(self, state_emb, action_indices):
        return self.net(state_emb) + action_indices.float().unsqueeze(-1) * 0.01


def test_verl_wm_aux_uses_consecutive_latents_and_stops_target_qwen_grad() -> None:
    from nimloth.training.rl.verl_wm_aux import (
        NimlothWMAuxiliaryModules,
        compute_verl_wm_auxiliary_loss,
    )
    from nimloth.wm.state_proj import StateProjector

    actor = _Actor()
    wm_aux = NimlothWMAuxiliaryModules(
        StateProjector(
            qwen_hidden_dim=4,
            lewm_emb_dim=4,
            projector_hidden_dim=8,
            latent_token_count=2,
        ),
        _Predictor(4),
    )
    input_ids = torch.tensor([[1, 2, 30, 31, 3, 4, 30, 31, 5]])
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(input_ids.shape[1]).repeat(3, 1, 1),
        "wm_latent_positions": torch.tensor([[[2, 3], [6, 7]]]),
        "wm_action_indices": torch.tensor([[3, 1]]),
        "wm_transition_mask": torch.tensor([[1, 0]]),
    }

    loss, metrics = compute_verl_wm_auxiliary_loss(
        actor, wm_aux, batch, latent_token_count=2
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["wm_transitions"] == 1.0
    assert metrics["wm_mse"] == pytest.approx(float(loss.detach().item()))
    embedding_grad = actor.model.language_model.embed_tokens.weight.grad
    assert embedding_grad[30].abs().sum().item() > 0
    # Both current and next blocks use the same query token ids, so inspect the
    # surrounding non-query token: no unrelated token may receive gradients.
    assert embedding_grad[5].abs().sum().item() == 0
    assert any(
        parameter.grad is not None
        for parameter in wm_aux.state_projector.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in wm_aux.predictor.parameters()
    )


def test_pinned_verl_actor_calls_and_checkpoints_wm_auxiliary() -> None:
    from pathlib import Path

    actor = Path(
        "external/VAGEN/verl/verl/workers/actor/dp_actor.py"
    ).read_text(encoding="utf-8")
    assert "compute_verl_wm_auxiliary_loss" in actor
    assert "scaled_wm_loss.backward()" in actor
    assert "policy_no_sync = self.actor_module.no_sync()" in actor
    assert "self.wm_optimizer.step()" in actor
    wm_aux_source = Path(
        "src/nimloth/training/rl/verl_wm_aux.py"
    ).read_text(encoding="utf-8")
    assert "fsdp_backward_anchor = model_output.logits.sum() * 0.0" in wm_aux_source
    worker = Path(
        "external/VAGEN/verl/verl/workers/fsdp_workers.py"
    ).read_text(encoding="utf-8")
    assert "_build_nimloth_wm_auxiliary" in worker
    assert "nimloth_wm_aux.pt" in worker
    assert "enabled Nimloth WM auxiliary checkpoint is missing" in worker
    assert "Nimloth WM auxiliary checkpoint schema mismatch" in worker
    assert "Nimloth WM auxiliary checkpoint query-mode mismatch" in worker
    assert "Nimloth WM auxiliary checkpoint latent-count mismatch" in worker
    assert "Nimloth WM auxiliary checkpoint config mismatch" in worker


def test_verl_wm_aux_rejects_wrong_latent_block_width() -> None:
    from nimloth.training.rl.verl_wm_aux import (
        NimlothWMAuxiliaryModules,
        compute_verl_wm_auxiliary_loss,
    )
    from nimloth.wm.state_proj import StateProjector

    actor = _Actor()
    wm_aux = NimlothWMAuxiliaryModules(
        StateProjector(4, 4, projector_hidden_dim=8, latent_token_count=2),
        _Predictor(4),
    )
    input_ids = torch.tensor([[1, 2, 30, 31]])
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(input_ids.shape[1]).repeat(3, 1, 1),
        "wm_latent_positions": torch.tensor([[[2]]]),
        "wm_action_indices": torch.tensor([[3]]),
        "wm_transition_mask": torch.tensor([[0]]),
    }
    with pytest.raises(ValueError, match=r"shape \[B, T, k\]"):
        compute_verl_wm_auxiliary_loss(
            actor, wm_aux, batch, latent_token_count=2
        )
