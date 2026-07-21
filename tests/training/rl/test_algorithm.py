"""RL 核心算法的梯度 ownership 保护测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from nimloth.agent import Agent
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.rollout import EncodedTrajectory, EncodedTransition
from nimloth.training.rl.algorithm import (
    RLAlgorithm,
    RLBatch,
    count_sequence_windows,
    select_sequence_batch,
)
from nimloth.wm.model import WorldModel
from nimloth.wm.sigreg import SequenceSIGReg
from nimloth.wm.value_head import ValueHead


class _UnusedBackbone(Backbone):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = torch.nn.Identity()

    @property
    def model(self) -> torch.nn.Module:
        return self.language_model

    def forward(self, batch: BackboneBatch, **_kwargs) -> BackboneOutput:
        return BackboneOutput(batch.tensors["hidden"])

    def with_model(self, model: torch.nn.Module) -> "_UnusedBackbone":
        return self

    def save_pretrained(self, output_dir: Path, **_kwargs) -> None:
        raise NotImplementedError


class _Predictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)

    def forward(
        self,
        state: torch.Tensor,
        _action_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(state)


class _RecordingSIGReg(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        self.inputs.append(states)
        return states.pow(2).mean()


def _algorithm(
    *,
    sigreg: SequenceSIGReg | None = None,
) -> tuple[RLAlgorithm, torch.nn.Linear, _Predictor, ValueHead]:
    state_proj = torch.nn.Linear(3, 2, bias=False)
    predictor = _Predictor()
    value_head = ValueHead(emb_dim=2, num_actions=3, hidden_dim=2)
    agent = Agent(
        backbone=_UnusedBackbone(),
        wm=WorldModel(
            state_proj=state_proj,
            wm_predictor=predictor,
            value_head=value_head,
        ),
    )
    optimizer = torch.optim.SGD(agent.wm.parameters(), lr=0.01)
    return (
        RLAlgorithm(
            agent=agent,
            optimizer=optimizer,
            device=torch.device("cpu"),
            vision_ema=None,
            policy_replay=None,
            history_size=2,
            sigreg=sigreg,
            sigreg_weight=0.1 if sigreg is not None else 0.0,
            value_rank_margin=0.1,
            value_rank_weight=1.0,
            ppo_clip_ratio=0.2,
            entropy_weight=0.0,
        ),
        state_proj,
        predictor,
        value_head,
    )


def _transition(
    record_id: str,
    step_index: int,
    current_hidden: torch.Tensor,
    next_hidden: torch.Tensor,
) -> EncodedTransition:
    return EncodedTransition(
        record_id=record_id,
        step_index=step_index,
        current_hidden=current_hidden,
        next_hidden=next_hidden,
        action_index=step_index % 3,
        value_target=float(step_index),
        old_log_prob=-0.5,
        policy_messages=[],
        policy_image_paths=[],
        sampling_temperature=1.0,
        sampling_top_p=1.0,
        latent_token_count=1,
    )


def _trajectory(record_id: str, length: int, *, offset: float = 0.0) -> EncodedTrajectory:
    hidden = [torch.full((3,), offset + step) for step in range(length + 1)]
    return EncodedTrajectory(
        record_id=record_id,
        transitions=tuple(
            _transition(record_id, step, hidden[step], hidden[step + 1])
            for step in range(length)
        ),
    )


def _batch() -> RLBatch:
    hidden_states = torch.randn(2, 3, 3, requires_grad=True)
    windows = tuple(
        tuple(
            _transition(
                f"record_{batch_index}",
                step,
                hidden_states.detach()[batch_index, step],
                hidden_states.detach()[batch_index, step + 1],
            )
            for step in range(2)
        )
        for batch_index in range(2)
    )
    return RLBatch(
        windows=windows,
        hidden_states=hidden_states,
        action_indices=torch.tensor([[0, 2], [1, 0]]),
        return_targets=torch.tensor([[1.0, -0.5], [0.5, 0.25]]),
        old_log_probs=torch.zeros(2, 2),
    )


def test_sequence_batch_preserves_trajectory_boundaries_and_alignment() -> None:
    trajectories = (
        _trajectory("short", 1),
        _trajectory("long", 3, offset=10.0),
    )

    assert count_sequence_windows(trajectories, history_size=2) == 2
    batch = select_sequence_batch(
        trajectories,
        history_size=2,
        batch_size=2,
        seed=7,
        device=torch.device("cpu"),
    )

    assert batch.hidden_states.shape == (2, 3, 3)
    assert batch.action_indices.shape == (2, 2)
    assert all(
        {transition.record_id for transition in window} == {"long"}
        for window in batch.windows
    )
    for batch_index, window in enumerate(batch.windows):
        expected = torch.stack(
            [window[0].current_hidden]
            + [transition.next_hidden for transition in window]
        )
        torch.testing.assert_close(batch.hidden_states[batch_index], expected)


def test_rl_wm_stops_gradient_on_next_projector_target() -> None:
    algorithm, state_proj, predictor, _ = _algorithm()
    batch = _batch()
    output = algorithm.training_step(batch)

    output.losses["wm"].backward()

    assert batch.hidden_states.grad is not None
    assert bool(batch.hidden_states.grad[:, :-1].abs().sum() > 0)
    assert torch.count_nonzero(batch.hidden_states.grad[:, -1]) == 0
    assert state_proj.weight.grad is not None
    assert predictor.linear.weight.grad is not None


def test_rl_value_updates_head_but_not_state_projector() -> None:
    algorithm, state_proj, _, value_head = _algorithm()
    batch = _batch()
    output = algorithm.training_step(batch)

    output.losses["value"].backward()

    assert batch.hidden_states.grad is None
    assert state_proj.weight.grad is None
    assert value_head.net[0].weight.grad is not None


def test_rl_sigreg_receives_full_history_plus_target_sequence() -> None:
    recording = _RecordingSIGReg()
    algorithm, _, _, _ = _algorithm(
        sigreg=SequenceSIGReg(regularizer=recording),
    )
    batch = _batch()

    output = algorithm.training_step(batch)

    assert len(recording.inputs) == 1
    assert recording.inputs[0].shape == (3, 2, 2)
    assert output.losses["sigreg"] is not None
