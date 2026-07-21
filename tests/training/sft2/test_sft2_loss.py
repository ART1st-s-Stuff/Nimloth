from __future__ import annotations

import torch

from nimloth.backbone.qwen25vl.transition import (
    QwenTransitionEncoder,
    QwenTransitionMessages,
)
from nimloth.model import NimlothModel
from nimloth.training.sft2.algorithm import (
    SFT2Algorithm,
    SFT2LossWeights,
    SFT2Losses,
    build_trajectory_sigreg_inputs,
    wm_loss_weight_schedule,
)
from nimloth.training.sft2.data.batch import SFT2Transition
from nimloth.wm import LatentWMPredictor, StateProjector, ValueHead, WorldModel
from nimloth.wm.lewm import LeWMConfig


def _algorithm(
    state_proj: torch.nn.Module,
    wm_predictor: torch.nn.Module,
    sigreg: torch.nn.Module | None,
) -> SFT2Algorithm:
    llm = torch.nn.Identity()
    return SFT2Algorithm(
        model=NimlothModel(
            llm=llm,
            wm=WorldModel(
                state_proj=state_proj,
                wm_predictor=wm_predictor,
                value_head=ValueHead(emb_dim=wm_predictor.config.emb_dim),
            ),
        ),
        qwen=QwenTransitionEncoder(
            processor=None,
            token_id_map={},
            device=torch.device("cpu"),
            max_length=16,
            pad_token_id=0,
        ),
        sigreg=sigreg,
    )


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

    dynamics, sigreg = _algorithm(
        state_proj,
        wm_predictor,
        None,
    )._compute_wm_losses(
        current_hidden,
        next_hidden,
        torch.tensor([0, 3]),
        [],
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

    dynamics, sigreg = _algorithm(
        state_proj,
        wm_predictor,
        MockSIGReg(),
    )._compute_wm_losses(
        torch.randn(3, 4),
        torch.randn(3, 4),
        torch.tensor([0, 1, 2]),
        [("rec_A", 0), ("rec_A", 1), ("rec_B", 5)],
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


def test_sft2_losses_apply_runtime_weights() -> None:
    weighted = SFT2Losses(
            lm=torch.tensor(3.0),
            dynamics=torch.tensor(2.0),
            sigreg=None,
            value=torch.tensor(0.0),
            metrics={"wm_mse": 2.0, "lm_ce": 3.0},
        ).weighted(SFT2LossWeights(wm=0.5, sigreg=0.0, value=1.0, ce=1.0))
    assert weighted.loss.item() == 4.0
    assert weighted.metrics["total_loss"] == 4.0
    assert weighted.metrics["lm_ce"] == 3.0
