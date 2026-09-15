"""Exact input guards and encoder derivative scheduling, independent of GPU gates."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nimloth.backbone import BackboneBatch
from nimloth.backbone.qwen25vl.batch import collate_qwen_encodings
from nimloth.training.sft.stage3.trajectory import (
    SharedTrajectoryExecution, encoded_rows, make_prefix_plan, require_deterministic_encoder,
    require_prefix,
)
from nimloth.util.profiling import StepTimer


class Processor:
    image_token = "image"
    image_processor = SimpleNamespace(merge_size=1)
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: 9, pad_token_id=0)


def row(turns):
    ids = torch.tensor([value for _ in range(turns) for value in (9, 2, 3, 4)])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                image_grid_thw=torch.ones(turns, 3, dtype=torch.long),
                pixel_values=torch.arange(turns, dtype=torch.float32).reshape(-1, 1))


class Rope(nn.Module):
    def get_rope_index(self, *, input_ids, attention_mask, **_):
        positions = (attention_mask.cumsum(-1) - 1).unsqueeze(0).expand(3, -1, -1)
        return positions, None


class Builder:
    processor = Processor()

    def collate_encoded(self, rows, *, include_labels):
        return BackboneBatch(collate_qwen_encodings(rows, 0))


def test_shared_plan_matches_images_positions_and_retains_state_order():
    first, second, other = row(1), row(3), row(2)
    backbone = SimpleNamespace(model=Rope(), token_id_map={"<|latent_state|>": 2, "<|latent_state_1|>": 3}, latent_token_count=2)
    plan = make_prefix_plan({("a", 0): first, ("a", 2): second, ("b", 1): other}, Builder(), backbone)
    assert plan.batch.tensors["input_ids"].shape == (2, 12)
    assert plan.batch.tensors["state_positions"].tolist() == [
        [[0, 1], [0, 2]], [[0, 9], [0, 10]], [[1, 5], [1, 6]]]
    packed = Builder().collate_encoded([first, other], include_labels=False)
    recovered = encoded_rows(packed, Processor())
    for expected, actual in zip((first, other), recovered, strict=True):
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key])


@pytest.mark.parametrize("field", ["input_ids", "pixel_values", "image_grid_thw"])
def test_shared_prefix_rejects_changed_input(field):
    short, long = row(1), row(2)
    long[field][0] += 1
    with pytest.raises(ValueError, match="differs"):
        require_prefix(short, long)


def test_shared_prefix_rejects_changed_rope_even_with_equal_tokens():
    class BadRope(Rope):
        def get_rope_index(self, **kwargs):
            value, _ = super().get_rope_index(**kwargs)
            return value + kwargs["input_ids"].shape[-1], None
    backbone = SimpleNamespace(model=BadRope(), token_id_map={"<|latent_state|>": 2, "<|latent_state_1|>": 3}, latent_token_count=2)
    with pytest.raises(ValueError, match="mRoPE"):
        make_prefix_plan({("a", 0): row(1), ("a", 1): row(2)}, Builder(), backbone)


def test_shared_derivative_schedule_equals_direct_accumulated_gradient():
    torch.manual_seed(4)
    encoder = nn.Linear(3, 5)
    projector = nn.Linear(5, 2)
    head = nn.Linear(5, 1)
    inputs = torch.randn(5, 3)
    hidden = encoder(inputs)
    states, lm = projector(hidden), head(hidden).flatten().square()
    def objective(states, lm, index):
        # Adjacent windows overlap. SIGReg-like term stops only current-side grad.
        primary = (states[index:index + 2] ** 2).mean() / 3
        regularizer = (states[index].detach() - states[index + 1]).square().mean() / 3
        return primary + regularizer + lm[index] * (index != 1) / 2
    sum(objective(states, lm, i) for i in range(3)).backward()
    parameters = [*encoder.parameters(), *projector.parameters(), *head.parameters()]
    expected = [p.grad.clone() for p in parameters]
    for p in parameters:
        p.grad = None
    hidden = encoder(inputs)
    shared = SharedTrajectoryExecution.__new__(SharedTrajectoryExecution)
    shared.states, shared.lm_losses = projector(hidden), head(hidden).flatten().square()
    shared.state_leaf = shared.states.detach().requires_grad_()
    shared.lm_leaf = shared.lm_losses.detach().requires_grad_()
    shared.timer = StepTimer(enabled=False)
    for index in range(3):
        objective(shared.state_leaf, shared.lm_leaf, index).backward()
    assert all(p.grad is None for p in parameters)
    shared.finish()
    for parameter, gradient in zip(parameters, expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


def test_shared_encoder_rejects_stochastic_dropout():
    agent = SimpleNamespace(backbone=nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.1)),
                            wm=SimpleNamespace(state_proj=nn.Identity()))
    with pytest.raises(ValueError, match="dropout"):
        require_deterministic_encoder(agent)


def test_preencoded_rollout_matches_original_losses_and_all_gradients():
    import copy
    from nimloth.agent.model import AgentStateOutput
    import importlib.util
    from pathlib import Path
    fixture_path = Path(__file__).with_name("test_sft2_loss.py")
    spec = importlib.util.spec_from_file_location("shared_trajectory_loss_fixtures", fixture_path)
    assert spec is not None and spec.loader is not None
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    _algorithm, _rollout_batch, _RecordingSIGReg = (
        fixtures._algorithm, fixtures._rollout_batch, fixtures._RecordingSIGReg)
    torch.manual_seed(8)
    algorithm, runtime, _, _ = _algorithm(history_size=1, prediction_horizon=4,
                                         sigreg=_RecordingSIGReg())
    batch = _rollout_batch()
    clone_runtime = copy.deepcopy(runtime)
    output = algorithm.training_primary_step(runtime, batch, wm_weight=0.4)
    output.loss.backward()
    regularizer = algorithm.training_sigreg_step(runtime, batch,
        detached_current_state=output.current_state.detach(), sigreg_seed=77)
    regularizer.loss.backward()
    expected = {name: param.grad.clone() for name, param in runtime.agent.named_parameters()}
    targets = clone_runtime.encode_next_state(batch.next)
    encoded = clone_runtime.agent.encode_state(batch.current, include_lm_loss=True)
    tail = clone_runtime.agent.encode_state(batch.online_tail).state
    state_leaf = encoded.state.detach().requires_grad_()
    tail_leaf = tail.detach().requires_grad_()
    lm_leaf = encoded.lm_loss.detach().requires_grad_()
    actual = algorithm.training_primary_step(clone_runtime, batch, wm_weight=0.4,
        encoded_current=AgentStateOutput(encoded.hidden.detach(), state_leaf, lm_leaf),
        target_states=targets)
    actual.loss.backward()
    reg = algorithm.training_sigreg_step(clone_runtime, batch,
        detached_current_state=actual.current_state.detach(), sigreg_seed=77,
        online_next_state=tail_leaf)
    reg.loss.backward()
    torch.autograd.backward((encoded.state, tail, encoded.lm_loss),
                           (state_leaf.grad, tail_leaf.grad, lm_leaf.grad))
    torch.testing.assert_close(output.loss, actual.loss)
    torch.testing.assert_close(regularizer.loss, reg.loss)
    for name, parameter in clone_runtime.agent.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name])


@pytest.mark.parametrize("state_trainable,lm_trainable", [(False, True), (True, False), (False, False)])
def test_shared_vjp_allows_frozen_encoder_outputs(state_trainable, lm_trainable):
    group = SharedTrajectoryExecution.__new__(SharedTrajectoryExecution)
    group.states = torch.ones(2, requires_grad=state_trainable)
    group.lm_losses = torch.ones(2, requires_grad=lm_trainable)
    group.state_leaf = group.states.detach().requires_grad_()
    group.lm_leaf = group.lm_losses.detach().requires_grad_()
    group.timer = StepTimer(enabled=False)
    (group.state_leaf.sum() + group.lm_leaf.sum()).backward()
    group.finish()
