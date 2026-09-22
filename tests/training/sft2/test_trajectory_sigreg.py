"""Unique real transition ownership and direct SIGReg gradient contract."""
from types import SimpleNamespace

import torch

from nimloth.training.sft.stage3.algorithm import SFT2Algorithm
from nimloth.training.sft.stage3.batch import Stage3BatchAssembler


class PairLoss(torch.nn.Module):
    def forward(self, pairs):
        self.pairs = pairs.detach().clone()
        if len(pairs) < 2:
            return None
        return pairs.square().mean()


def run(trajectory_factory, weights):
    rows = [trajectory_factory(record=str(i), length=length, weight=weight)
            for i, (length, weight) in enumerate(weights)]
    batch = Stage3BatchAssembler(input_builder=rows[0][0], device=torch.device('cpu'),
                                prediction_horizon=4).prepare([row[2] for row in rows])
    states = torch.arange(len(batch.state_keys) * 2, dtype=torch.float32).reshape(-1, 2).requires_grad_()
    regularizer = PairLoss()
    algorithm = SFT2Algorithm(history_size=1, sigreg=regularizer, sigreg_weight=.2,
                             value_weight=1., ce_weight=1., prediction_horizon=4)
    runtime = SimpleNamespace(agent=SimpleNamespace(wm=SimpleNamespace(sigreg_state=lambda value: value)))
    result = algorithm.training_sigreg_step(runtime, batch, online_states=states, sigreg_seed=77)
    result.loss.backward()
    return batch, states, regularizer, result


def test_all_real_pairs_once_not_overlapping_windows(trajectory_factory):
    batch, states, regularizer, result = run(trajectory_factory, [(6, 1.), (4, 1.)])
    # Four WM windows, but all ten real transitions belong to SIGReg exactly once.
    assert batch.batch_size == 4
    indices = torch.tensor([0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
    expected = torch.stack((states.detach()[indices], states.detach()[indices + 1]), 1)
    torch.testing.assert_close(regularizer.pairs, expected)
    assert result.metrics['sigreg_global_batch_size'] == 10
    gradient = torch.zeros_like(states)
    gradient[indices + 1] = .2 * 2 * states.detach()[indices + 1] / expected.numel()
    torch.testing.assert_close(states.grad, gradient)


def test_padding_trajectory_contributes_no_pair_or_gradient(trajectory_factory):
    _, states, regularizer, result = run(trajectory_factory, [(6, 1.), (4, 0.)])
    assert len(regularizer.pairs) == 6
    assert result.metrics['sigreg_global_batch_size'] == 6
    assert torch.count_nonzero(states.grad[7:]) == 0


def test_all_padding_keeps_zero_backward_graph(trajectory_factory):
    _, states, _, result = run(trajectory_factory, [(4, 0.)])
    assert result.raw_loss is None
    assert result.metrics['sigreg_global_batch_size'] == 0
    assert torch.count_nonzero(states.grad) == 0
