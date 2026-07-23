from __future__ import annotations

from pathlib import Path

import torch

from nimloth.agent import Agent
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.rollout import TransitionBatch
from nimloth.training.sft2.dino_grid import (
    DINOGridSFT2Algorithm,
    DINOGridSFT2Batch,
)
from nimloth.training.sft2.batch import SFT2Batch
from nimloth.training.sft2.history_cache import OnlineHistoryStateCache
from nimloth.training.sft2.runtime import SFT2ModelRuntime
from nimloth.wm.grid import (
    EMATargetGridEncoder,
    GridStateProjector,
    GridWorldModel,
    LeWMGridDecoder,
    LeWMGridEncoder,
    SharedSlotProjector,
)
from nimloth.wm.value_head import ValueHead


class _TensorGridBackbone(Backbone):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = torch.nn.Identity()
        self.calls = 0

    @property
    def model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        batch: BackboneBatch,
        *,
        include_lm_loss: bool = False,
    ) -> BackboneOutput:
        self.calls += 1
        hidden = self.language_model(batch.tensors["hidden"])
        return BackboneOutput(
            hidden=hidden,
            lm_loss=hidden.mean() if include_lm_loss else None,
        )

    def with_model(self, model: torch.nn.Module) -> "_TensorGridBackbone":
        return self

    def save_pretrained(self, output_dir: Path, **_kwargs) -> None:
        raise NotImplementedError


class _GridPredictor(torch.nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.net = torch.nn.Linear(dimension, dimension, bias=False)

    def forward(
        self,
        states: torch.Tensor,
        _actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(states)


def _runtime() -> tuple[SFT2ModelRuntime, _TensorGridBackbone, GridWorldModel]:
    backbone = _TensorGridBackbone()
    slot_projector = SharedSlotProjector(
        input_dim=6,
        output_dim=8,
        hidden_dim=12,
        grid_tokens=4,
    ).requires_grad_(False)
    online_encoder = LeWMGridEncoder(emb_dim=8, hidden_dim=16)
    state_proj = GridStateProjector(slot_projector, online_encoder)
    wm = GridWorldModel(
        state_proj=state_proj,
        target_encoder=EMATargetGridEncoder(online_encoder, decay=0.99),
        wm_predictor=_GridPredictor(8),
        dino_decoder=LeWMGridDecoder(emb_dim=8, hidden_dim=16),
        value_head=ValueHead(emb_dim=8, hidden_dim=8, num_actions=3),
    )
    cache = OnlineHistoryStateCache()
    cache.start(epoch=1, phase="train")
    return (
        SFT2ModelRuntime(
            agent=Agent(backbone=backbone, wm=wm),
            history_cache=cache,
        ),
        backbone,
        wm,
    )


def _batch() -> DINOGridSFT2Batch:
    trajectory_steps = (
        ("rec_A", 0),
        ("rec_A", 1),
        ("rec_B", 0),
        ("rec_B", 1),
    )
    base = SFT2Batch(
        transitions=TransitionBatch(
            current=BackboneBatch(
                {"hidden": torch.randn(2, 4, 6, requires_grad=True)}
            ),
            next=BackboneBatch(
                {"hidden": torch.randn(4, 4, 6, requires_grad=True)}
            ),
            action_indices=torch.tensor([0, 1, 1, 2]),
            value_targets=torch.tensor([0.0, 1.0, 0.0, 0.5]),
            next_indices=torch.arange(4),
            non_terminal_mask=torch.ones(4, dtype=torch.bool),
            trajectory_steps=trajectory_steps,
        ),
        online_tail=BackboneBatch(
            {"hidden": torch.randn(2, 4, 6, requires_grad=True)}
        ),
        history_size=2,
        sample_weights=torch.ones(2),
        next_image_paths=("a.png", "b.png"),
    )
    return DINOGridSFT2Batch(
        base=base,
        target_grid=torch.randn(2, 4, 8),
    )


def test_dino_grid_primary_step_keeps_one_ce_and_explicit_gradient_boundaries() -> None:
    torch.manual_seed(0)
    runtime, backbone, wm = _runtime()
    batch = _batch()
    older_states = torch.randn(2, 4, 8)
    runtime.history_cache.store(
        (("rec_A", 0), ("rec_B", 0)),
        older_states,
    )
    algorithm = DINOGridSFT2Algorithm(
        history_size=2,
        sigreg=None,
        sigreg_weight=0.0,
        value_weight=1.0,
        ce_weight=1.0,
        value_rank_margin=0.1,
        value_rank_weight=1.0,
        dino_weight=0.5,
    )

    output = algorithm.training_primary_step(
        runtime,
        batch,
        wm_weight=1.0,
    )
    expected_ce = batch.base.current.tensors["hidden"].mean()
    torch.testing.assert_close(output.losses["lm"], expected_ce)
    assert set(output.losses) == {"lm", "wm", "dino", "value"}
    assert output.current_state.shape == (2, 4, 8)
    assert output.metrics["lambda_dino"] == 0.5

    output.loss.backward()

    assert backbone.calls == 2
    assert batch.base.current.tensors["hidden"].grad is not None
    assert batch.base.next.tensors["hidden"].grad is None
    assert older_states.grad is None
    projector = wm.state_proj
    assert all(
        parameter.grad is None
        for parameter in projector.slot_projector.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in projector.online_encoder.parameters()
    )
    assert any(parameter.grad is not None for parameter in wm.wm_predictor.parameters())
    assert any(parameter.grad is not None for parameter in wm.dino_decoder.parameters())
    assert all(
        parameter.grad is None
        for parameter in wm.target_encoder.parameters()
    )
