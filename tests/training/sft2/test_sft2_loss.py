from __future__ import annotations

import torch

from nimloth.training.sft2.loss import (
    StateProjector,
    _build_trajectory_sigreg_inputs,
    compute_combined_loss,
    compute_wm_latent_loss,
    wm_loss_weight_schedule,
)
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.lewm import LeWMConfig


def test_wm_latent_loss_no_detach_grad_to_state_proj() -> None:
    """MSE target is NOT detached; state_proj gets gradient from both sides."""
    cfg = LeWMConfig(emb_dim=16, predictor_hidden_dim=16, predictor_mlp_dim=32)
    wm_predictor = LatentWMPredictor.create(cfg)

    state_proj = StateProjector(qwen_hidden_dim=32, lewm_emb_dim=cfg.emb_dim)
    qwen_hidden = torch.randn(2, 32, requires_grad=True)
    # next_hidden has requires_grad=True so we can verify gradient flows
    # through state_proj on the target side too.
    qwen_next_hidden = torch.randn(2, 32, requires_grad=True)
    actions = torch.tensor([0, 3])

    loss, sigreg_loss, metrics = compute_wm_latent_loss(
        qwen_hidden_at_latent=qwen_hidden,
        qwen_hidden_at_next_latent=qwen_next_hidden,
        action_indices=actions,
        state_proj=state_proj,
        wm_predictor=wm_predictor,
    )
    loss.backward()

    assert loss.item() > 0
    assert sigreg_loss is None
    assert "wm_mse" in metrics
    # State projector should receive gradient (it was trainable on both sides).
    assert state_proj.net.net[0].weight.grad is not None
    # Current-side Qwen hidden should receive gradient.
    assert qwen_hidden.grad is not None
    # Target-side Qwen hidden should also receive gradient (no detach on MSE target).
    assert qwen_next_hidden.grad is not None


def test_wm_latent_loss_with_items_trajectory_sigreg() -> None:
    """When items with record_id/step_index are passed, SIGReg runs per-trajectory."""
    cfg = LeWMConfig(emb_dim=4, predictor_hidden_dim=4, predictor_mlp_dim=8)
    wm_predictor = LatentWMPredictor.create(cfg)
    state_proj = StateProjector(qwen_hidden_dim=4, lewm_emb_dim=cfg.emb_dim, projector_hidden_dim=8)

    # Simulate 3 transitions: two from record A (steps 0, 1), one from record B.
    qwen_current = torch.randn(3, 4)
    qwen_next = torch.randn(3, 4)
    actions = torch.tensor([0, 1, 2])
    items = [
        {"record_id": "rec_A", "step_index": 0, "id": "rec_A:0"},
        {"record_id": "rec_A", "step_index": 1, "id": "rec_A:1"},
        {"record_id": "rec_B", "step_index": 5, "id": "rec_B:5"},
    ]

    # Use a simple SIGReg for testing (the real SIGReg from le-wm may not be
    # available in a CPU-only env; we mock its interface).
    class MockSIGReg(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (T, B, D)
            assert x.dim() == 3
            return x.pow(2).mean()

    sigreg = MockSIGReg()
    mse, sigreg_loss, metrics = compute_wm_latent_loss(
        qwen_hidden_at_latent=qwen_current,
        qwen_hidden_at_next_latent=qwen_next,
        action_indices=actions,
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        sigreg_module=sigreg,
        items=items,
    )

    assert sigreg_loss is not None
    assert sigreg_loss.item() > 0
    assert "sigreg_loss" in metrics
    assert "wm_mse" in metrics


def test_build_trajectory_sigreg_inputs() -> None:
    """Unit test for trajectory grouping helper."""
    D = 4
    # Simulate 4 transitions:
    #   rec_A step 0: s0→s1, rec_A step 1: s1→s2
    #   rec_B step 0: s0→s1
    #   rec_C step 3: s3→s4
    state_emb = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # rec_A step 0 state
        [2.0, 0.0, 0.0, 0.0],  # rec_A step 1 state
        [3.0, 0.0, 0.0, 0.0],  # rec_B step 0 state
        [4.0, 0.0, 0.0, 0.0],  # rec_C step 3 state
    ])
    target_emb = torch.tensor([
        [1.1, 0.0, 0.0, 0.0],  # rec_A step 0 target (s1)
        [2.1, 0.0, 0.0, 0.0],  # rec_A step 1 target (s2)
        [3.1, 0.0, 0.0, 0.0],  # rec_B step 0 target (s1)
        [4.1, 0.0, 0.0, 0.0],  # rec_C step 3 target (s4)
    ])
    items = [
        {"record_id": "rec_A", "step_index": 0},
        {"record_id": "rec_A", "step_index": 1},
        {"record_id": "rec_B", "step_index": 0},
        {"record_id": "rec_C", "step_index": 3},
    ]

    result = _build_trajectory_sigreg_inputs(items, state_emb, target_emb)

    # Should get 3 trajectories: rec_A (T=3), rec_B (T=2), rec_C (T=2)
    assert len(result) == 3
    shapes = sorted([tuple(r.shape) for r in result])
    assert shapes == [(2, 1, D), (2, 1, D), (3, 1, D)]

    # Check rec_A trajectory (T=3): s0_state, s1_target, s2_target
    rec_a = [r for r in result if r.shape[0] == 3][0].squeeze(1)
    assert torch.equal(rec_a[0], state_emb[0])   # s0_state (step 0 state)
    assert torch.equal(rec_a[1], target_emb[0])  # s1 from step 0 target
    assert torch.equal(rec_a[2], target_emb[1])  # s2 from step 1 target


def test_build_trajectory_sigreg_inputs_empty() -> None:
    assert _build_trajectory_sigreg_inputs(
        [], torch.empty(0, 4), torch.empty(0, 4)
    ) == []


def test_build_trajectory_sigreg_inputs_old_cache_fallback() -> None:
    """When items have no record_id (old cache), returns empty list → pair fallback."""
    D = 2
    state_emb = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    target_emb = torch.tensor([[1.1, 0.0], [2.1, 0.0], [3.1, 0.0]])
    items = [
        {"id": "?", "step_index": 0},  # no record_id, empty id
        {"id": "?", "step_index": 0},
        {"id": "?", "step_index": 0},
    ]
    result = _build_trajectory_sigreg_inputs(items, state_emb, target_emb)
    assert result == [], f"Expected empty for old cache, got {len(result)} groups"


def test_build_trajectory_sigreg_inputs_fallback_from_id() -> None:
    """When record_id is absent, parses it from the 'id' field."""
    D = 2
    state_emb = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    target_emb = torch.tensor([[1.1, 0.0], [2.1, 0.0]])
    items = [
        {"id": "rec_X:0", "step_index": 0},
        {"id": "rec_X:1", "step_index": 1},
    ]
    result = _build_trajectory_sigreg_inputs(items, state_emb, target_emb)
    assert len(result) == 1
    assert result[0].shape == (3, 1, D)  # T=3 for two consecutive steps


def test_wm_loss_weight_schedule_warms_up() -> None:
    assert wm_loss_weight_schedule(0, 100, start=0.1, end=1.0, warmup_fraction=0.5) == 0.1
    mid = wm_loss_weight_schedule(25, 100, start=0.1, end=1.0, warmup_fraction=0.5)
    assert 0.1 < mid < 1.0
    assert wm_loss_weight_schedule(60, 100, start=0.1, end=1.0, warmup_fraction=0.5) == 1.0


def test_compute_combined_loss() -> None:
    wm = torch.tensor(2.0)
    lm = torch.tensor(3.0)
    total, metrics = compute_combined_loss(wm_loss=wm, value_loss=None, lm_loss=lm, lambda_wm=0.5, lambda_ce=1.0)
    assert total.item() == 4.0
    assert metrics["total_loss"] == 4.0
    assert metrics["lm_ce"] == 3.0
