"""SFT2 Agent forward、objective 与梯度语义测试。"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch

from nimloth.agent import Agent
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.rollout import TransitionBatch
from nimloth.training.sft2.algorithm import SFT2Algorithm
from nimloth.training.sft2.batch import SFT2Batch
from nimloth.training.sft2.runtime import SFT2ModelRuntime
from nimloth.wm import SequenceSIGReg, StateProjector, ValueHead, WorldModel


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

    def forward_chunked(
        self,
        batch: BackboneBatch,
        *,
        max_rows: int,
        include_lm_loss: bool = False,
        gradient_rows=None,
        lm_loss_rows=None,
        **_kwargs,
    ) -> BackboneOutput:
        assert max_rows == 1
        self.calls += 1
        source = batch.tensors["hidden"]
        gradient_rows = set(range(source.shape[0])) if gradient_rows is None else set(gradient_rows)
        hidden = torch.stack(
            [row if index in gradient_rows else row.detach() for index, row in enumerate(source)]
        )
        loss_rows = list(range(hidden.shape[0])) if lm_loss_rows is None else list(lm_loss_rows)
        return BackboneOutput(
            hidden=hidden,
            lm_loss=hidden[loss_rows].mean() if include_lm_loss else None,
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
    sigreg: SequenceSIGReg | None = None,
    *,
    history_size: int = 2,
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
            history_size=history_size,
            sigreg=sigreg,
            sigreg_weight=sigreg_weight,
            value_weight=1.0,
            ce_weight=1.0,
            value_rank_margin=0.1,
            value_rank_weight=1.0,
            backbone_rows_per_forward=1,
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
        ("rec_B", 0),
        ("rec_B", 1),
    ),
    history_size: int = 2,
) -> SFT2Batch:
    row_count = len(trajectory_steps)
    batch_size = row_count // history_size
    return SFT2Batch(
        transitions=TransitionBatch(
            current=BackboneBatch(
                {"hidden": torch.randn(row_count, 4, requires_grad=True)}
            ),
            next=BackboneBatch(
                {"hidden": torch.randn(row_count, 4, requires_grad=True)}
            ),
            action_indices=torch.arange(row_count) % 3,
            value_targets=torch.linspace(-0.5, 1.0, row_count),
            next_indices=torch.arange(row_count),
            non_terminal_mask=torch.ones(row_count, dtype=torch.bool),
            trajectory_steps=trajectory_steps,
        ),
        online_tail=BackboneBatch(
            {"hidden": torch.randn(batch_size, 4, requires_grad=True)}
        ),
        history_size=history_size,
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


def test_sft2_training_step_computes_one_current_step_loss_and_detaches_history() -> None:
    algorithm, runtime, backbone, projector = _algorithm()
    batch = _batch()

    output = algorithm.training_step(runtime, batch, wm_weight=0.5)
    output.losses["wm"].backward()

    assert backbone.calls == 2
    assert output.model_output.lm_loss is not None
    assert projector.outputs[0].grad is not None
    assert projector.outputs[1].grad is not None
    current_grad = batch.current.tensors["hidden"].grad
    assert current_grad is not None
    assert torch.count_nonzero(current_grad.reshape(2, 2, 4)[:, :-1]) == 0
    assert torch.count_nonzero(current_grad.reshape(2, 2, 4)[:, -1]) > 0
    assert batch.next.tensors["hidden"].grad is None


def test_sft2_ce_supervises_each_contexts_current_step_only() -> None:
    algorithm, runtime, _, _ = _algorithm()
    batch = _batch()

    output = algorithm.training_step(runtime, batch, wm_weight=1.0)

    expected = batch.current.tensors["hidden"].reshape(2, 2, 4)[:, -1].mean()
    torch.testing.assert_close(output.losses["lm"], expected)


def test_sft2_predictor_receives_full_configured_history_axis() -> None:
    algorithm, runtime, _, _ = _algorithm(history_size=2)
    predictor = runtime.agent.wm.wm_predictor
    seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def record_shape(state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        seen.append((tuple(state.shape), tuple(actions.shape)))
        return state

    predictor.forward = record_shape  # type: ignore[method-assign]
    output = algorithm.training_step(runtime, _batch(), wm_weight=1.0)

    assert seen == [((2, 2, 4), (2, 2))]
    assert output.model_output.predicted_next_state.shape == (2, 4)
    assert output.model_output.action_values.shape == (2, 3)


def test_sft2_sigreg_receives_each_current_next_pair_once() -> None:
    recording = _RecordingSIGReg()
    sigreg = SequenceSIGReg(regularizer=recording)
    algorithm, runtime, _, projector = _algorithm(sigreg)
    batch = _batch()

    output = algorithm.training_step(runtime, batch, wm_weight=1.0)

    assert len(recording.inputs) == 1
    assert recording.inputs[0].shape == (2, 2, 4)
    expected = torch.stack(
        (
            projector.outputs[0].reshape(2, 2, 4)[:, -1],
            projector.outputs[2],
        ),
        dim=0,
    )
    torch.testing.assert_close(recording.inputs[0], expected)
    assert output.losses["sigreg"] is not None
    assert output.metrics["sigreg_loss"] > 0.0
    output.losses["sigreg"].backward()
    assert batch.online_tail.tensors["hidden"].grad is not None
    assert batch.next.tensors["hidden"].grad is None


def test_sft2_sigreg_skips_single_window_batch() -> None:
    recording = _RecordingSIGReg()
    algorithm, runtime, _, _ = _algorithm(
        SequenceSIGReg(regularizer=recording)
    )

    output = algorithm.training_step(
        runtime,
        _batch(
            trajectory_steps=(("rec_A", 0), ("rec_A", 1)),
        ),
        wm_weight=1.0,
    )

    assert recording.inputs == []
    assert output.losses["sigreg"] is None
    assert output.metrics["sigreg_skipped_small_batch"] == 1.0


def test_sft2_evaluation_does_not_use_training_sigreg_layout() -> None:
    recording = _RecordingSIGReg()
    algorithm, runtime, _, _ = _algorithm(
        SequenceSIGReg(regularizer=recording)
    )

    output = algorithm.evaluation_step(runtime, _batch())

    assert recording.inputs == []
    assert output.losses["sigreg"] is None
    assert output.metrics["lambda_sigreg"] == 0.0


def test_sft2_zero_sigreg_weight_does_not_run_module() -> None:
    recording = _RecordingSIGReg()
    algorithm, runtime, _, _ = _algorithm(
        SequenceSIGReg(regularizer=recording),
        sigreg_weight=0.0,
    )
    batch = _batch()

    output = algorithm.training_step(runtime, batch, wm_weight=1.0)

    assert recording.inputs == []
    assert output.losses["sigreg"] is None
    assert "sigreg_skipped_small_batch" not in output.metrics


def test_algorithm_wm_weight_warms_up() -> None:
    algorithm, _, _, _ = _algorithm()
    assert algorithm.wm_weight(0, 100) == 0.1
    assert 0.1 < algorithm.wm_weight(15, 100) < 1.0
    assert algorithm.wm_weight(60, 100) == 1.0
