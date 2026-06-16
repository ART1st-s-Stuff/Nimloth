from __future__ import annotations

import torch

from nimloth.sft2.loss import StateProjector, compute_combined_loss, compute_wm_alignment_loss, wm_loss_weight_schedule
from nimloth.wm.lewm import LeWMConfig, LeWMWrapper


def test_wm_alignment_loss_backprops_to_state_proj() -> None:
    cfg = LeWMConfig(emb_dim=16, action_emb_dim=8, predictor_hidden_dim=16, predictor_mlp_dim=32)
    lewm = LeWMWrapper.create(cfg)
    lewm.freeze()

    state_proj = StateProjector(qwen_hidden_dim=32, lewm_emb_dim=cfg.emb_dim)
    qwen_hidden = torch.randn(2, 32, requires_grad=True)
    actions = torch.tensor([0, 3])
    next_pixels = torch.randn(2, 3, cfg.img_size, cfg.img_size)

    loss, metrics = compute_wm_alignment_loss(
        qwen_hidden_at_latent=qwen_hidden,
        action_indices=actions,
        next_pixels=next_pixels,
        state_proj=state_proj,
        lewm=lewm,
    )
    loss.backward()

    assert loss.item() > 0
    assert "wm_mse" in metrics
    assert state_proj.net.weight.grad is not None
    assert qwen_hidden.grad is not None


def test_wm_loss_weight_schedule_warms_up() -> None:
    assert wm_loss_weight_schedule(0, 100, start=0.1, end=1.0, warmup_fraction=0.5) == 0.1
    mid = wm_loss_weight_schedule(25, 100, start=0.1, end=1.0, warmup_fraction=0.5)
    assert 0.1 < mid < 1.0
    assert wm_loss_weight_schedule(60, 100, start=0.1, end=1.0, warmup_fraction=0.5) == 1.0


def test_compute_combined_loss() -> None:
    wm = torch.tensor(2.0)
    lm = torch.tensor(3.0)
    total, metrics = compute_combined_loss(wm_loss=wm, lm_loss=lm, lambda_wm=0.5, lambda_ce=1.0)
    assert total.item() == 4.0
    assert metrics["total_loss"] == 4.0
    assert metrics["lm_ce"] == 3.0
