"""SFT2 Agent forward、objective 与梯度语义测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from nimloth.agent import Agent, AgentBatch, AgentTarget
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.training.sft2.algorithm import SFT2Algorithm
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
    return (
        SFT2Algorithm(
            agent=agent,
            target=AgentTarget(agent),
            sigreg=sigreg,
            sigreg_weight=sigreg_weight,
            value_weight=1.0,
            ce_weight=1.0,
            value_rank_margin=0.1,
            value_rank_weight=1.0,
        ),
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
) -> AgentBatch:
    row_count = len(trajectory_steps)
    if non_terminal_mask is None:
        non_terminal_mask = torch.ones(row_count, dtype=torch.bool)
    return AgentBatch(
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


def test_sft2_training_step_uses_agent_forward_and_two_sided_projector_gradient() -> None:
    algorithm, backbone, projector = _algorithm()
    batch = _batch()

    output = algorithm.training_step(batch, wm_weight=0.5)
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
    algorithm, _, projector = _algorithm(sigreg)
    batch = _batch(
        trajectory_steps=(
            ("rec_A", 0),
            ("rec_A", 1),
            ("rec_B", 0),
            ("rec_B", 1),
        ),
    )

    output = algorithm.training_step(batch, wm_weight=1.0)

    assert len(recording.inputs) == 1
    assert recording.inputs[0].shape == (2, 4, 4)
    expected = torch.stack((projector.outputs[0], projector.outputs[1]), dim=0)
    torch.testing.assert_close(recording.inputs[0], expected)
    assert output.losses["sigreg"] is not None
    assert output.metrics["sigreg_loss"] > 0.0


def test_sft2_sigreg_skips_single_transition_batch() -> None:
    recording = _RecordingSIGReg()
    algorithm, _, _ = _algorithm(OneStepSIGReg(regularizer=recording))

    output = algorithm.training_step(
        _batch(trajectory_steps=(("rec_A", 0),)),
        wm_weight=1.0,
    )

    assert recording.inputs == []
    assert output.losses["sigreg"] is None
    assert output.metrics["sigreg_skipped_small_batch"] == 1.0


def test_sft2_sigreg_excludes_transitions_without_next_state() -> None:
    recording = _RecordingSIGReg()
    algorithm, _, projector = _algorithm(OneStepSIGReg(regularizer=recording))
    batch = _batch(
        trajectory_steps=(
            ("rec_A", 0),
            ("rec_A", 1),
            ("rec_B", 0),
        ),
        non_terminal_mask=torch.tensor([True, False, True]),
    )

    output = algorithm.training_step(batch, wm_weight=1.0)

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
    algorithm, _, _ = _algorithm(OneStepSIGReg(regularizer=recording))

    output = algorithm.evaluation_step(_batch())

    assert recording.inputs == []
    assert output.losses["sigreg"] is None
    assert output.metrics["lambda_sigreg"] == 0.0


def test_sft2_zero_sigreg_weight_does_not_run_module() -> None:
    recording = _RecordingSIGReg()
    algorithm, _, _ = _algorithm(
        OneStepSIGReg(regularizer=recording),
        sigreg_weight=0.0,
    )
    batch = _batch(
        trajectory_steps=(("rec_A", 0), ("rec_B", 0)),
    )

    output = algorithm.training_step(batch, wm_weight=1.0)

    assert recording.inputs == []
    assert output.losses["sigreg"] is None
    assert "sigreg_skipped_small_batch" not in output.metrics


def test_algorithm_wm_weight_warms_up() -> None:
    algorithm, _, _ = _algorithm()
    assert algorithm.wm_weight(0, 100) == 0.1
    assert 0.1 < algorithm.wm_weight(15, 100) < 1.0
    assert algorithm.wm_weight(60, 100) == 1.0
