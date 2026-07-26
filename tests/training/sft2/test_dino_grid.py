from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import json

import torch

from nimloth.agent import Agent
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.rollout import TransitionBatch
import nimloth.training.sft2.dino_grid as dino_grid_module
from nimloth.training.sft2.algorithm import SFT2Algorithm
from nimloth.training.sft2.batch import SFT2Batch
from nimloth.training.sft2.history_cache import OnlineHistoryStateCache
from nimloth.training.sft2.runtime import SFT2ModelRuntime
from nimloth.training.sft2.trainer import _build_world_model
from nimloth.wm.grid import (
    EMATargetGridEncoder,
    GridStateProjector,
    GridWorldModel,
    LeWMGridDecoder,
    LeWMGridEncoder,
    SharedSlotProjector,
)
from nimloth.wm.sigreg import SequenceSIGReg
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


class _RecordingSIGReg(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.inputs.append(value)
        return value.pow(2).mean()


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


def test_grid_world_model_keeps_trainable_grid_modules_in_fp32(tmp_path) -> None:
    slot_projector = SharedSlotProjector(
        input_dim=6,
        output_dim=4,
        hidden_dim=8,
        grid_tokens=2,
    )
    (tmp_path / "grid_state_config.json").write_text(
        json.dumps(
            {
                "grid_tokens": 2,
                "qwen_hidden_dim": 6,
                "state_dim": 4,
                "projector_hidden_dim": 8,
                "shared_slot_projector": True,
                "ordering": "row_major",
            }
        ),
        encoding="utf-8",
    )
    torch.save(slot_projector.state_dict(), tmp_path / "slot_projector.pt")
    model = torch.nn.Linear(1, 1, bias=False).to(dtype=torch.bfloat16)
    model.config = SimpleNamespace(hidden_size=6)
    args = SimpleNamespace(
        objective="dino_grid",
        model=tmp_path,
        emb_dim=4,
        latent_token_count=2,
        grid_encoder_hidden_dim=8,
        grid_ema_decay=0.99,
        history_size=2,
        grid_wm_depth=1,
        grid_wm_heads=1,
        grid_wm_dim_head=4,
        grid_wm_mlp_dim=8,
        grid_wm_dropout=0.0,
        grid_decoder_hidden_dim=8,
        resume=False,
        grid_warmstart=None,
    )

    world_model, world_model_device = _build_world_model(
        args,
        model=model,
        device=torch.device("cpu"),
        pair_parallel=False,
        resume_ckpt_dir=None,
        train_wm_predictor=True,
    )

    assert world_model_device == torch.device("cpu")
    assert next(world_model.state_proj.slot_projector.parameters()).dtype == torch.bfloat16
    for module in (
        world_model.state_proj.online_encoder,
        world_model.target_encoder,
        world_model.wm_predictor,
        world_model.dino_decoder,
        world_model.value_head,
    ):
        assert next(module.parameters()).dtype == torch.float32


def _batch() -> SFT2Batch:
    trajectory_steps = (
        ("rec_A", 0),
        ("rec_A", 1),
        ("rec_B", 0),
        ("rec_B", 1),
    )
    return SFT2Batch(
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
        dino_grid_target=torch.randn(2, 4, 8),
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
    assert not hasattr(dino_grid_module, "DINOGridSFT2Algorithm")
    algorithm = SFT2Algorithm(
        history_size=2,
        sigreg=None,
        sigreg_weight=0.0,
        value_weight=1.0,
        ce_weight=1.0,
        value_rank_margin=0.1,
        value_rank_weight=1.0,
        dino_grid_weight=0.25,
    )

    output = algorithm.training_primary_step(
        runtime,
        batch,
        wm_weight=1.0,
    )
    expected_ce = batch.current.tensors["hidden"].mean()
    torch.testing.assert_close(output.losses["lm"], expected_ce)
    assert set(output.losses) == {"lm", "wm", "dino", "value"}
    assert output.current_state.shape == (2, 4, 8)
    assert output.metrics["lambda_dino"] == 0.25

    output.loss.backward()

    assert backbone.calls == 2
    assert batch.current.tensors["hidden"].grad is not None
    assert batch.next.tensors["hidden"].grad is None
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


def test_grid_sigreg_uses_same_core_stage_with_mean_pooled_slots() -> None:
    runtime, _backbone, _wm = _runtime()
    batch = _batch()
    recording = _RecordingSIGReg()
    algorithm = SFT2Algorithm(
        history_size=2,
        sigreg=SequenceSIGReg(regularizer=recording),
        sigreg_weight=0.1,
        value_weight=1.0,
        ce_weight=1.0,
        value_rank_margin=0.1,
        value_rank_weight=1.0,
        dino_grid_weight=0.5,
    )

    output = algorithm.training_sigreg_step(
        runtime,
        batch,
        detached_current_state=torch.randn(2, 4, 8),
        sigreg_seed=9,
    )

    assert output.raw_loss is not None
    assert len(recording.inputs) == 1
    assert recording.inputs[0].shape == (2, 2, 8)
