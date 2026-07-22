"""SFT2 Agent forward、objective 与梯度语义测试。"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch

from nimloth.agent import Agent
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.rollout import TransitionBatch
from nimloth.training.sft2.algorithm import SFT2Algorithm
from nimloth.training.sft2.runtime import SFT2ModelRuntime
from nimloth.wm import OneStepSIGReg, StateProjector, ValueHead, WorldModel


class _TensorBackbone(Backbone):
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

    def with_model(self, model: torch.nn.Module) -> "_TensorBackbone":
        return self

    def save_pretrained(self, output_dir: Path, **_kwargs) -> None:
        raise NotImplementedError


class _WrappedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = torch.nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.module(value)


class _ReplaceableBackbone(Backbone):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = model

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def forward(self, batch: BackboneBatch, **_kwargs) -> BackboneOutput:
        return BackboneOutput(self._model(batch.tensors["hidden"]))

    def with_model(self, model: torch.nn.Module) -> "_ReplaceableBackbone":
        return _ReplaceableBackbone(model)

    def save_pretrained(self, output_dir: Path, **_kwargs) -> None:
        raise NotImplementedError


class _RecordingEMA:
    def __init__(self) -> None:
        self.shadow: dict[str, torch.Tensor] = {}
        self.models: list[torch.nn.Module] = []

    def update(self, model: torch.nn.Module) -> None:
        pass

    @contextlib.contextmanager
    def use_ema_weights(self, model: torch.nn.Module):
        self.models.append(model)
        yield

    def save_checkpoint(self, path: Path) -> None:
        pass


class _Predictor(torch.nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.net = torch.nn.Linear(dimension, dimension, bias=False)

    def forward(
        self,
        state: torch.Tensor,
        _action_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(state)


class _RecordingProjector(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4, bias=False)
        self.outputs: list[torch.Tensor] = []

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output = super().forward(hidden)
        output.retain_grad()
        self.outputs.append(output)
        return output


class _RecordingSIGReg(torch.nn.Module):
    """记录算法传给 SIGReg 的实际 ``(T, B, D)`` 输入。"""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.inputs.append(value)
        return value.pow(2).mean()


def _algorithm(
    sigreg: OneStepSIGReg | None = None,
    *,
    sigreg_weight: float = 0.1,
):
    backbone = _TensorBackbone()
    projector = _RecordingProjector()
    agent = Agent(
        backbone=backbone,
        wm=WorldModel(
            state_proj=projector,
            wm_predictor=_Predictor(4),
            value_head=ValueHead(emb_dim=4, num_actions=3),
        ),
    )
    runtime = SFT2ModelRuntime(agent=agent)
    return (
        SFT2Algorithm(
            sigreg=sigreg,
            sigreg_weight=sigreg_weight,
            value_weight=1.0,
            ce_weight=1.0,
            value_rank_margin=0.1,
            value_rank_weight=1.0,
        ),
        runtime,
        backbone,
        projector,
    )


def _batch(
    *,
    trajectory_steps: tuple[tuple[str, int], ...] = (
        ("rec_A", 0),
        ("rec_A", 1),
    ),
    non_terminal_mask: torch.Tensor | None = None,
) -> TransitionBatch:
    row_count = len(trajectory_steps)
    if non_terminal_mask is None:
        non_terminal_mask = torch.ones(row_count, dtype=torch.bool)
    return TransitionBatch(
        current=BackboneBatch(
            {"hidden": torch.randn(row_count, 4, requires_grad=True)}
        ),
        next=BackboneBatch(
            {"hidden": torch.randn(row_count, 4, requires_grad=True)}
        ),
        action_indices=torch.arange(row_count) % 3,
        value_targets=torch.linspace(-0.5, 1.0, row_count),
        next_indices=torch.arange(row_count),
        non_terminal_mask=non_terminal_mask,
        trajectory_steps=trajectory_steps,
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


def test_sft2_algorithm_is_pure_compute_configuration() -> None:
    algorithm, _, _, _ = _algorithm()

    assert not isinstance(algorithm, torch.nn.Module)
    assert not hasattr(algorithm, "agent")
    assert not hasattr(algorithm, "target")
    assert not hasattr(algorithm, "optimizer")


def test_unwrapped_runtime_applies_ema_to_its_own_backbone_model() -> None:
    wrapped_model = _WrappedModel()
    agent = Agent(
        backbone=_ReplaceableBackbone(wrapped_model),
        wm=WorldModel(
            state_proj=torch.nn.Linear(4, 4),
            wm_predictor=_Predictor(4),
            value_head=ValueHead(emb_dim=4, num_actions=3),
        ),
    )
    ema = _RecordingEMA()
    validation_runtime = SFT2ModelRuntime(
        agent=agent,
        backbone_ema=ema,
    ).unwrapped()

    with validation_runtime.evaluation_context():
        pass

    assert validation_runtime.agent.backbone.model is wrapped_model.module
    assert ema.models == [validation_runtime.agent.backbone.model]


def test_sft2_training_step_uses_agent_forward_and_two_sided_projector_gradient() -> None:
    algorithm, runtime, backbone, projector = _algorithm()
    batch = _batch()

    output = algorithm.training_step(runtime, batch, wm_weight=0.5)
    output.losses["wm"].backward()

    assert backbone.calls == 2
    assert output.model_output.lm_loss is not None
    assert projector.outputs[0].grad is not None
    assert projector.outputs[1].grad is not None
    assert batch.current.tensors["hidden"].grad is not None
    assert batch.next.tensors["hidden"].grad is None


def test_sft2_sigreg_receives_one_time_batch_dimension_input() -> None:
    recording = _RecordingSIGReg()
    sigreg = OneStepSIGReg(regularizer=recording)
    algorithm, runtime, _, projector = _algorithm(sigreg)
    batch = _batch(
        trajectory_steps=(
            ("rec_A", 0),
            ("rec_A", 1),
            ("rec_B", 0),
            ("rec_B", 1),
        ),
    )

    output = algorithm.training_step(runtime, batch, wm_weight=1.0)

    assert len(recording.inputs) == 1
    assert recording.inputs[0].shape == (2, 4, 4)
    expected = torch.stack((projector.outputs[0], projector.outputs[1]), dim=0)
    torch.testing.assert_close(recording.inputs[0], expected)
    assert output.losses["sigreg"] is not None
    assert output.metrics["sigreg_loss"] > 0.0


def test_sft2_sigreg_skips_single_transition_batch() -> None:
    recording = _RecordingSIGReg()
    algorithm, runtime, _, _ = _algorithm(OneStepSIGReg(regularizer=recording))

    output = algorithm.training_step(
        runtime,
        _batch(trajectory_steps=(("rec_A", 0),)),
        wm_weight=1.0,
    )

    assert recording.inputs == []
    assert output.losses["sigreg"] is None
    assert output.metrics["sigreg_skipped_small_batch"] == 1.0


def test_sft2_sigreg_excludes_transitions_without_next_state() -> None:
    recording = _RecordingSIGReg()
    algorithm, runtime, _, projector = _algorithm(OneStepSIGReg(regularizer=recording))
    batch = _batch(
        trajectory_steps=(
            ("rec_A", 0),
            ("rec_A", 1),
            ("rec_B", 0),
        ),
        non_terminal_mask=torch.tensor([True, False, True]),
    )

    output = algorithm.training_step(runtime, batch, wm_weight=1.0)

    assert len(recording.inputs) == 1
    expected = torch.stack(
        (projector.outputs[0][[0, 2]], projector.outputs[1][[0, 2]]),
        dim=0,
    )
    torch.testing.assert_close(recording.inputs[0], expected)
    assert output.losses["sigreg"] is not None
    assert "wm_mse" in output.metrics


def test_sft2_evaluation_does_not_use_training_sigreg_layout() -> None:
    recording = _RecordingSIGReg()
    algorithm, runtime, _, _ = _algorithm(OneStepSIGReg(regularizer=recording))

    output = algorithm.evaluation_step(runtime, _batch())

    assert recording.inputs == []
    assert output.losses["sigreg"] is None
    assert output.metrics["lambda_sigreg"] == 0.0


def test_sft2_zero_sigreg_weight_does_not_run_module() -> None:
    recording = _RecordingSIGReg()
    algorithm, runtime, _, _ = _algorithm(
        OneStepSIGReg(regularizer=recording),
        sigreg_weight=0.0,
    )
    batch = _batch(
        trajectory_steps=(("rec_A", 0), ("rec_B", 0)),
    )

    output = algorithm.training_step(runtime, batch, wm_weight=1.0)

    assert recording.inputs == []
    assert output.losses["sigreg"] is None
    assert "sigreg_skipped_small_batch" not in output.metrics


def test_algorithm_wm_weight_warms_up() -> None:
    algorithm, _, _, _ = _algorithm()
    assert algorithm.wm_weight(0, 100) == 0.1
    assert 0.1 < algorithm.wm_weight(15, 100) < 1.0
    assert algorithm.wm_weight(60, 100) == 1.0
