from __future__ import annotations

import torch

from nimloth.backbone.qwen25vl.transition import QwenTransitionMessages
from nimloth.training.sft2.algorithm import (
    SFT2Losses,
    build_trajectory_sigreg_inputs,
    combine_sft2_losses,
    compute_sft2_dynamics,
    wm_loss_weight_schedule,
)
from nimloth.training.sft2.data.batch import SFT2Transition
from nimloth.wm import LatentWMPredictor, StateProjector
from nimloth.wm.lewm import LeWMConfig


def test_state_projector_accepts_multi_latent_block() -> None:
    state_proj = StateProjector(
        qwen_hidden_dim=8,
        lewm_emb_dim=4,
        projector_hidden_dim=16,
        latent_token_count=3,
    )
    out = state_proj(torch.randn(2, 3, 8))
    assert out.shape == (2, 4)
    assert state_proj.input_dim == 24


def test_sft2_dynamics_keeps_projector_gradient_on_both_sides() -> None:
    cfg = LeWMConfig(emb_dim=16, predictor_hidden_dim=16, predictor_mlp_dim=32)
    wm_predictor = LatentWMPredictor.create(cfg)
    state_proj = StateProjector(qwen_hidden_dim=32, lewm_emb_dim=cfg.emb_dim)
    current_hidden = torch.randn(2, 32, requires_grad=True)
    next_hidden = torch.randn(2, 32, requires_grad=True)

    dynamics, sigreg = compute_sft2_dynamics(
        current_hidden=current_hidden,
        next_hidden=next_hidden,
        action_indices=torch.tensor([0, 3]),
        trajectory_steps=None,
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        sigreg=None,
    )
    dynamics.loss.backward()

    assert dynamics.loss.item() > 0
    assert sigreg is None
    assert state_proj.net.net[0].weight.grad is not None
    assert current_hidden.grad is not None
    assert next_hidden.grad is not None


def test_sft2_dynamics_runs_sigreg_per_trajectory() -> None:
    cfg = LeWMConfig(emb_dim=4, predictor_hidden_dim=4, predictor_mlp_dim=8)
    wm_predictor = LatentWMPredictor.create(cfg)
    state_proj = StateProjector(
        qwen_hidden_dim=4,
        lewm_emb_dim=cfg.emb_dim,
        projector_hidden_dim=8,
    )

    class MockSIGReg(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            assert value.dim() == 3
            return value.pow(2).mean()

    dynamics, sigreg = compute_sft2_dynamics(
        current_hidden=torch.randn(3, 4),
        next_hidden=torch.randn(3, 4),
        action_indices=torch.tensor([0, 1, 2]),
        trajectory_steps=[("rec_A", 0), ("rec_A", 1), ("rec_B", 5)],
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        sigreg=MockSIGReg(),
    )

    assert dynamics.loss.item() > 0
    assert sigreg is not None
    assert sigreg.item() > 0


def test_build_trajectory_sigreg_inputs() -> None:
    dimension = 4
    current = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ]
    )
    target = current + 0.1
    result = build_trajectory_sigreg_inputs(
        [("rec_A", 0), ("rec_A", 1), ("rec_B", 0), ("rec_C", 3)],
        current,
        target,
    )

    assert sorted(tuple(value.shape) for value in result) == [
        (2, 1, dimension),
        (2, 1, dimension),
        (3, 1, dimension),
    ]
    rec_a = next(value for value in result if value.shape[0] == 3).squeeze(1)
    assert torch.equal(rec_a[0], current[0])
    assert torch.equal(rec_a[1], target[0])
    assert torch.equal(rec_a[2], target[1])


def test_build_trajectory_sigreg_inputs_empty_or_legacy() -> None:
    assert build_trajectory_sigreg_inputs(
        [],
        torch.empty(0, 4),
        torch.empty(0, 4),
    ) == []
    assert build_trajectory_sigreg_inputs(
        [("", 0), ("", 1)],
        torch.randn(2, 4),
        torch.randn(2, 4),
    ) == []


def test_sft2_transition_resolves_legacy_record_id() -> None:
    transition = SFT2Transition(
        identifier="rec_X:1",
        record_id="",
        step_index=1,
        action_index=0,
        value_target=1.0,
        success=True,
        qwen=QwenTransitionMessages(current=[], next=None),
    )
    assert transition.trajectory_step == ("rec_X", 1)


def test_wm_loss_weight_schedule_warms_up() -> None:
    assert wm_loss_weight_schedule(
        0,
        100,
        start=0.1,
        end=1.0,
        warmup_fraction=0.5,
    ) == 0.1
    mid = wm_loss_weight_schedule(
        25,
        100,
        start=0.1,
        end=1.0,
        warmup_fraction=0.5,
    )
    assert 0.1 < mid < 1.0
    assert wm_loss_weight_schedule(
        60,
        100,
        start=0.1,
        end=1.0,
        warmup_fraction=0.5,
    ) == 1.0


def test_combine_sft2_losses() -> None:
    weighted = combine_sft2_losses(
        SFT2Losses(
            lm=torch.tensor(3.0),
            dynamics=torch.tensor(2.0),
            sigreg=None,
            value=torch.tensor(0.0),
            metrics={"wm_mse": 2.0, "lm_ce": 3.0},
        ),
        wm_weight=0.5,
        sigreg_weight=0.0,
        value_weight=1.0,
        ce_weight=1.0,
    )
    assert weighted.loss.item() == 4.0
    assert weighted.metrics["total_loss"] == 4.0
    assert weighted.metrics["lm_ce"] == 3.0
