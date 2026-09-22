"""Causal sharing preserves prefix states, virtual losses and accumulated gradients."""
from types import SimpleNamespace
import copy

import pytest
import torch
from torch import nn

from nimloth.backbone.qwen25vl.latent import extract_qwen_trajectory_latents
from nimloth.backbone.selected_token_rows import _install_leaf


class CausalModel(nn.Module):
    """Small causal test encoder; this is not a real Qwen/GPU validation."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(31, 8)
        self.model = nn.Module()
        self.model.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 31, bias=False)
        _install_leaf(self.head, (20, 21), (22, 23))

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, **kwargs):
        hidden = self.model.norm(self.embedding(input_ids).cumsum(1))
        return SimpleNamespace(logits=self.head(hidden), loss=None)


@pytest.mark.parametrize("weights", [[1., 1.], [1., 0.], [0., 0.]])
def test_shared_prefix_states_losses_and_gradients(weights):
    torch.manual_seed(4)
    model = CausalModel()
    reference = copy.deepcopy(model)
    ids = torch.tensor([[2, 3, 20, 21, 4, 5, 20, 21, 6]])
    labels = torch.tensor([[-100, 3, -100, -100, 4, -100, -100, -100, -100],
                           [-100, -100, -100, -100, -100, 5, -100, -100, 6]])
    positions = torch.tensor([[[0, 2], [0, 3]], [[0, 6], [0, 7]]])
    mapping = {"<|latent_state|>": 20, "<|latent_state_1|>": 21}
    weights = torch.tensor(weights)
    states, losses = extract_qwen_trajectory_latents(
        model, {"input_ids": ids}, mapping, torch.device("cpu"),
        state_positions=positions, latent_token_count=2,
        lm_labels=labels, lm_source_rows=torch.zeros(2, dtype=torch.long),
        lm_row_weights=weights,
    )
    shared_loss = states.square().sum() + losses.sum()
    shared_loss.backward()
    reference_states, reference_losses = [], []
    for row, length in enumerate([5, 9]):
        state, loss = extract_qwen_trajectory_latents(
            reference, {"input_ids": ids[:, :length]}, mapping, torch.device("cpu"),
            state_positions=positions[row:row+1], latent_token_count=2,
            lm_labels=labels[row:row+1, :length], lm_source_rows=torch.zeros(1, dtype=torch.long),
            lm_row_weights=weights[row:row+1],
        )
        reference_states.append(state)
        reference_losses.append(loss)
        (state.square().sum() + loss.sum()).backward()
    torch.testing.assert_close(states, torch.cat(reference_states))
    torch.testing.assert_close(losses, torch.cat(reference_losses))
    for (name, parameter), (_, other) in zip(model.named_parameters(), reference.named_parameters()):
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-5, rtol=2e-5, msg=name)
    changed = ids.clone()
    changed[:, 4:] = 10
    early, _ = extract_qwen_trajectory_latents(
        model, {"input_ids": changed}, mapping, torch.device("cpu"),
        state_positions=positions[:1], latent_token_count=2,
    )
    torch.testing.assert_close(early, states[:1])


@pytest.mark.parametrize("positions", [
    [[[0, 2], [0, 4]]], [[[0, 1], [0, 2]]], [[[1, 2], [1, 3]]],
])
def test_invalid_explicit_slots_rejected(positions):
    with pytest.raises(ValueError):
        extract_qwen_trajectory_latents(
            CausalModel(), {"input_ids": torch.tensor([[2, 3, 20, 21]])},
            {"<|latent_state|>": 20, "<|latent_state_1|>": 21}, torch.device("cpu"),
            state_positions=torch.tensor(positions), latent_token_count=2,
        )
