from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import json

import torch

from nimloth.agent import Agent
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.rollout import TransitionBatch
import nimloth.training.sft2.dino_grid as dino_grid_module
from nimloth.training.common import world_model_loss
from nimloth.training.sft2.algorithm import SFT2Algorithm
from nimloth.training.sft2.batch import SFT2Batch, SFT2RolloutBatch
from nimloth.training.sft2.dino_grid import DINOGridBatchAssembler
from nimloth.training.sft2.history_cache import OnlineHistoryStateCache
from nimloth.training.sft2.runtime import SFT2ModelRuntime
from nimloth.training.sft2.trainer import _build_world_model
from nimloth.wm.grid import (
    GridWorldModel,
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

    def rollout_from_history(
        self,
        state_history: torch.Tensor,
        previous_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        assert state_history.shape[1] == 1
        assert previous_actions.shape[1] == 0
        state = state_history[:, -1]
        predicted = []
        for _step in range(future_actions.shape[1]):
            state = self.net(state)
            predicted.append(state)
        return torch.stack(predicted, dim=1)


def _runtime() -> tuple[SFT2ModelRuntime, _TensorGridBackbone, GridWorldModel]:
    backbone = _TensorGridBackbone()
    slot_projector = SharedSlotProjector(
        input_dim=6,
        output_dim=8,
        hidden_dim=12,
        grid_tokens=4,
    )
    wm = GridWorldModel(
        state_proj=slot_projector,
        wm_predictor=_GridPredictor(8),
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
        history_size=2,
        grid_wm_depth=1,
        grid_wm_heads=1,
        grid_wm_dim_head=4,
        grid_wm_mlp_dim=8,
        grid_wm_dropout=0.0,
        resume=False,
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
    assert next(world_model.state_proj.parameters()).dtype == torch.float32
    assert all(
        parameter.requires_grad
        for parameter in world_model.state_proj.parameters()
    )
    for module in (
        world_model.wm_predictor,
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


def test_dino_grid_batch_assembler_loads_only_current_next_images() -> None:
    batch = _batch()

    class BaseAssembler:
        processor = None

        @staticmethod
        def prepare(_raw_batch) -> SFT2Batch:  # type: ignore[no-untyped-def]
            return batch

    class Targets:
        grid_size = 4
        identity = SimpleNamespace(hidden_size=1024)

        def __init__(self) -> None:
            self.loaded_paths: list[tuple[str, ...]] = []

        def load(self, paths, *, device):  # type: ignore[no-untyped-def]
            self.loaded_paths.append(tuple(str(path) for path in paths))
            return torch.ones((len(paths), 4, 8), device=device)

    targets = Targets()
    assembler = DINOGridBatchAssembler(
        BaseAssembler(),  # type: ignore[arg-type]
        targets,  # type: ignore[arg-type]
    )

    prepared = assembler.prepare(object())

    assert targets.loaded_paths == [("a.png", "b.png")]
    assert prepared.dino_grid_target is not None
    assert prepared.dino_grid_target.shape == (2, 4, 8)


def test_dino_grid_batch_assembler_loads_all_t4_rollout_targets_in_order() -> None:
    horizon = 4
    paths = tuple(f"next-{index}.png" for index in range(8))
    batch = SFT2RolloutBatch(
        transitions=TransitionBatch(
            current=BackboneBatch({"hidden": torch.randn(2, 4, 6)}),
            next=BackboneBatch({"hidden": torch.randn(8, 4, 6)}),
            action_indices=torch.tensor([2, 0, 1, 2, 1, 2, 0, 1]),
            value_targets=torch.arange(8, dtype=torch.float32),
            next_indices=torch.arange(8),
            non_terminal_mask=torch.ones(8, dtype=torch.bool),
            trajectory_steps=tuple(
                (record_id, step)
                for record_id in ("rec_A", "rec_B")
                for step in range(horizon)
            ),
        ),
        online_tail=BackboneBatch({"hidden": torch.randn(2, 4, 6)}),
        prediction_horizon=horizon,
        sample_weights=torch.ones(2),
        next_image_paths=paths,
    )

    class BaseAssembler:
        processor = None

        @staticmethod
        def prepare(_raw_batch):  # type: ignore[no-untyped-def]
            return batch

    class Targets:
        grid_size = 4
        identity = SimpleNamespace(hidden_size=1024)

        def __init__(self) -> None:
            self.loaded_paths: tuple[str, ...] = ()

        def load(self, loaded_paths, *, device):  # type: ignore[no-untyped-def]
            self.loaded_paths = tuple(loaded_paths)
            return torch.arange(8 * 4 * 8, device=device).reshape(8, 4, 8)

    targets = Targets()
    prepared = DINOGridBatchAssembler(
        BaseAssembler(),  # type: ignore[arg-type]
        targets,  # type: ignore[arg-type]
    ).prepare(object())

    assert targets.loaded_paths == paths
    assert prepared.dino_grid_target is not None
    assert prepared.dino_grid_target.shape == (2, 4, 4, 8)
    torch.testing.assert_close(
        prepared.dino_grid_target.flatten(0, 1),
        torch.arange(8 * 4 * 8).reshape(8, 4, 8),
    )

    runtime, _backbone, _wm = _runtime()
    algorithm = SFT2Algorithm(
        history_size=1,
        prediction_horizon=4,
        sigreg=None,
        sigreg_weight=0.0,
        value_weight=1.0,
        ce_weight=1.0,
        dino_grid_weight=0.5,
    )
    output = algorithm.training_primary_step(runtime, prepared, wm_weight=1.0)

    assert set(output.losses) == {"lm", "wm", "dino", "value"}
    assert output.metrics["prediction_horizon"] == 4.0
    output.loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in runtime.agent.wm.wm_predictor.parameters()
    )


def test_dino_loss_directly_supervises_predicted_state() -> None:
    predicted_state = torch.zeros(1, 4, 8, requires_grad=True)
    target = torch.ones(1, 4, 8)

    objective = world_model_loss(
        predicted_state,
        predicted_state.detach(),
        state_weight=1.0,
        dino_grid_target=target,
        dino_grid_weight=1.0,
    )
    assert objective.dino_grid_mse is not None
    objective.dino_grid_mse.backward()

    torch.testing.assert_close(objective.dino_grid_mse, torch.tensor(1.0))
    assert predicted_state.grad is not None
    assert torch.count_nonzero(predicted_state.grad) == predicted_state.numel()


def test_grid_target_freezes_backbone_and_shared_projector() -> None:
    runtime, _backbone, wm = _runtime()
    next_batch = _batch().next

    expected_next_state = runtime.encode_next_state(next_batch)

    assert not expected_next_state.requires_grad
    assert next_batch.tensors["hidden"].grad is None
    assert all(parameter.grad is None for parameter in wm.state_proj.parameters())


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
    assert any(parameter.grad is not None for parameter in wm.state_proj.parameters())
    projector = wm.state_proj
    assert any(
        parameter.grad is not None
        for parameter in projector.parameters()
    )
    assert any(parameter.grad is not None for parameter in wm.wm_predictor.parameters())


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
