"""Native trajectory objectives retain window timing and direct gradient paths."""
from __future__ import annotations

import contextlib
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nimloth.agent import Agent
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.training.sft.stage3.algorithm import SFT2Algorithm
from nimloth.training.sft.stage3.runtime import SFT2ModelRuntime
from nimloth.wm import WorldModel


class TensorBackbone(Backbone):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Linear(4, 4, bias=False)
        self.calls = []

    @property
    def model(self):
        return self.language_model

    def forward(self, batch, *, include_lm_loss=False):
        self.calls.append((torch.is_grad_enabled(), self.training, include_lm_loss))
        hidden = self.model(batch.tensors['hidden'])
        losses = hidden[batch.tensors['starts']].square().mean(-1) if include_lm_loss else None
        return BackboneOutput(hidden, lm_losses=losses)

    def with_model(self, model):
        result = TensorBackbone()
        result.language_model = model
        return result

    def save_pretrained(self, *args, **kwargs):
        raise NotImplementedError


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 4, bias=False)
        self.config = SimpleNamespace(history_size=1)
        self.calls = 0

    def forward(self, states, actions):
        self.calls += 1
        assert states.shape[1] == actions.shape[1] == 1
        return self.net(states) + actions[..., None]


def runtime_and_batch(weights=(1., 1., 1.), success=(1., 1., 0.)):
    torch.manual_seed(17)
    agent = Agent(backbone=TensorBackbone(), wm=WorldModel(
        state_proj=nn.Linear(4, 4, bias=False), wm_predictor=Predictor(),
        value_head=nn.Linear(4, 3, bias=False)))
    # Two complete trajectories, states[0:4] and states[4:7], T=2 -> W=2+1.
    starts = torch.tensor([0, 1, 4])
    batch = SimpleNamespace(
        inputs=BackboneBatch({'hidden': torch.randn(7, 4), 'starts': starts}),
        current_indices=starts, next_indices=torch.tensor([[1, 2], [2, 3], [5, 6]]),
        action_sequences=torch.tensor([[0, 1], [1, 2], [2, 0]]),
        value_targets=torch.tensor([[.3, .2], [.2, .1], [1., .5]]),
        sample_weights=torch.tensor(weights), lm_weights=torch.tensor(success) * torch.tensor(weights),
        outcome_targets=torch.zeros(3, 2), outcome_mask=torch.zeros(3, 2, dtype=torch.bool),
        dino_grid_target=torch.randn(3, 2, 4), current_dino_target=None,
        observed_dino_target=torch.randn(7, 4),
        observed_state_weights=torch.tensor([weights[0]] * 4 + [weights[2]] * 3),
        prediction_horizon=2, state_offsets=(0, 4, 7), window_offsets=(0, 2, 3),
        trajectory_ids=('first', 'second'), batch_size=3,
    )
    return SFT2ModelRuntime(agent), batch


def algorithm(**kwargs):
    config = dict(history_size=1, prediction_horizon=2, sigreg=None,
                  sigreg_weight=0., value_weight=.7, ce_weight=.3, dino_grid_weight=.5)
    return SFT2Algorithm(**(config | kwargs))


def test_native_targets_first_then_one_online_and_one_batched_rollout():
    runtime, batch = runtime_and_batch()
    events = []
    class EMA:
        @contextlib.contextmanager
        def use_ema_weights(self, model):
            events.append('enter')
            yield
            events.append('exit')
    runtime = SFT2ModelRuntime(runtime.agent, EMA())
    output = algorithm().training_primary_step(runtime, batch, wm_weight=.4)
    assert runtime.agent.backbone.calls == [(False, False, False), (True, True, True)]
    assert events == ['enter', 'exit']
    assert runtime.agent.wm.wm_predictor.calls == batch.prediction_horizon
    assert output.sample_count == 3
    assert output.online_states.shape == (7, 4)
    assert not output.diagnostics['target_states'].requires_grad
    output.loss.backward()
    for name, parameter in runtime.agent.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize('padding', [False, True])
def test_native_window_losses_and_parameter_gradients_match_individual_windows(padding):
    runtime, batch = runtime_and_batch(weights=(1., 1., 0.) if padding else (1., 1., 1.))
    reference = copy.deepcopy(runtime.agent)
    output = algorithm().training_primary_step(runtime, batch, wm_weight=.4)
    output.loss.backward()
    all_count = batch.sample_weights.sum().clamp_min(1)
    lm_count = batch.lm_weights.sum().clamp_min(1)
    manual = 0.
    for row in range(3):
        with torch.no_grad():
            hidden = reference.backbone.model(batch.inputs.tensors['hidden'])
            target = reference.wm.project_state(hidden)[batch.next_indices[row]]
        hidden = reference.backbone.model(batch.inputs.tensors['hidden'][batch.current_indices[row]:batch.current_indices[row]+1])
        state = reference.wm.project_state(hidden)
        actions = batch.action_sequences[row:row+1]
        predictions = reference.wm.simulate_action_sequences(state[:, None], actions[:, :0], actions)
        decisions = torch.cat([state[:, None], predictions[:, :-1]], 1)
        values = reference.wm.predict_action_values(decisions).gather(-1, actions[..., None]).squeeze(-1)
        loss = (.4 * (predictions[0]-target).square().mean()
                + .7 * (values[0]-batch.value_targets[row]).square().mean())
        manual = manual + loss * batch.sample_weights[row] / all_count
        manual = manual + .3 * hidden.square().mean() * batch.lm_weights[row] / lm_count
    observed = reference.wm.project_state(reference.backbone.model(batch.inputs.tensors['hidden']))
    manual = manual + .5 * (((observed - batch.observed_dino_target).square().mean(-1)
                            * batch.observed_state_weights).sum()
                           / batch.observed_state_weights.sum().clamp_min(1))
    torch.testing.assert_close(output.loss, manual)
    manual.backward()
    for (name, actual), (_, expected) in zip(runtime.agent.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, msg=name, atol=2e-6, rtol=2e-5)


def test_all_padding_is_connected_zero_for_every_trained_component():
    runtime, batch = runtime_and_batch(weights=(0., 0., 0.))
    output = algorithm().training_primary_step(runtime, batch, wm_weight=.4)
    assert output.sample_count == 0 and output.loss.item() == 0
    output.loss.backward()
    for name, parameter in runtime.agent.named_parameters():
        assert parameter.grad is not None and parameter.grad.count_nonzero() == 0, name


def test_outgoing_values_use_predicted_decisions_and_only_executed_actions():
    runtime, batch = runtime_and_batch()
    values = []
    hook = runtime.agent.wm.value_head.register_forward_hook(lambda module, args, output: values.append((args[0], output)))
    output = algorithm().training_primary_step(runtime, batch, wm_weight=.4)
    hook.remove()
    decisions, action_values = values[0]
    assert decisions.shape == (3, 2, 4)
    torch.testing.assert_close(decisions[:, 0], output.current_state)
    torch.testing.assert_close(decisions[:, 1], output.diagnostics['predicted_states'][:, 0])
    grad, = torch.autograd.grad(output.losses['value'], action_values)
    selected = torch.nn.functional.one_hot(batch.action_sequences, 3).bool()
    assert grad[~selected].count_nonzero() == 0


def test_ema_target_uses_its_own_values_and_restores_online_parameters():
    runtime, batch = runtime_and_batch()
    original = runtime.agent.backbone.model.weight.detach().clone()
    class EMA:
        @contextlib.contextmanager
        def use_ema_weights(self, model):
            with torch.no_grad():
                model.weight.fill_(.5)
            yield
            with torch.no_grad():
                model.weight.copy_(original)
    runtime = SFT2ModelRuntime(runtime.agent, EMA())
    output = algorithm().training_primary_step(runtime, batch, wm_weight=.4)
    target_hidden = torch.nn.functional.linear(batch.inputs.tensors['hidden'], torch.full_like(original, .5))
    expected = runtime.agent.wm.project_state(target_hidden)[batch.next_indices]
    torch.testing.assert_close(output.diagnostics['target_states'], expected)
    torch.testing.assert_close(runtime.agent.backbone.model.weight, original)
    output.loss.backward()  # EMA restoration preceded graph construction; no version error.


def test_native_rejects_historical_context_and_warms_weight():
    with pytest.raises(ValueError, match='history_size=1'):
        algorithm(history_size=2)
    instance = algorithm()
    assert instance.wm_weight(0, 100) == pytest.approx(.1)
    assert instance.wm_weight(30, 100) == pytest.approx(1.)


def test_observed_dino_only_reaches_encoder_projector_and_all_real_states():
    runtime, batch = runtime_and_batch(weights=(1., 1., 0.))
    batch.observed_dino_target.requires_grad_()
    output = algorithm().training_primary_step(runtime, batch, wm_weight=.4)
    output.online_states.retain_grad()
    output.losses['dino'].backward()
    assert runtime.agent.backbone.model.weight.grad.abs().sum() > 0
    assert runtime.agent.wm.state_proj.weight.grad.abs().sum() > 0
    assert runtime.agent.wm.wm_predictor.net.weight.grad is None
    assert runtime.agent.wm.value_head.weight.grad is None
    assert batch.observed_dino_target.grad is None
    assert output.online_states.grad[:4].abs().sum(-1).gt(0).all()  # Includes terminal.
    assert output.online_states.grad[4:].count_nonzero() == 0


def test_wm_explicitly_detaches_future_encoder_target():
    runtime, batch = runtime_and_batch()
    teacher = torch.randn(7, 4, requires_grad=True)
    runtime = SimpleNamespace(agent=runtime.agent, encode_next_state=lambda inputs: teacher)
    output = algorithm().training_primary_step(runtime, batch, wm_weight=1.)
    output.losses['wm'].backward()
    assert teacher.grad is None
    assert runtime.agent.wm.wm_predictor.net.weight.grad.abs().sum() > 0


def test_window_overlap_does_not_reweight_observed_dino():
    runtime, batch = runtime_and_batch()
    original = algorithm().evaluation_step(runtime, batch)
    for name in ('current_indices', 'next_indices', 'action_sequences', 'value_targets',
                 'sample_weights', 'lm_weights', 'outcome_targets', 'outcome_mask', 'dino_grid_target'):
        value = getattr(batch, name)
        setattr(batch, name, torch.cat((value, value[:1]), dim=0))
    batch.inputs.tensors['starts'] = batch.current_indices
    duplicated = algorithm().evaluation_step(runtime, batch)
    torch.testing.assert_close(original.losses['dino'], duplicated.losses['dino'])
