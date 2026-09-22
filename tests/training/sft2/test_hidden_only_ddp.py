"""Real CPU DDP regression for alternating LM and hidden-only backwards."""
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from nimloth.backbone.qwen25vl.latent import connect_unused_logits


class AlternatingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(3, 3)
        self.head = nn.Linear(3, 1)

    def forward(self, inputs, lm):
        hidden = self.body(inputs)
        logits = self.head(hidden)
        connected = connect_unused_logits(hidden, logits)
        assert torch.equal(connected, hidden)
        return connected.square().mean() + (logits.square().mean() if lm else 0)


def _worker(rank, path):
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank, world_size=2)
    torch.manual_seed(42)
    model = nn.parallel.DistributedDataParallel(AlternatingModel(), static_graph=True)
    for _ in range(3):
        model.zero_grad()
        model(torch.full((2, 3), float(rank + 1)), True).backward()
        model(torch.full((2, 3), float(rank + 2)), False).backward()
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            other = parameter.grad.clone()
            dist.broadcast(other, 0)
            torch.testing.assert_close(other, parameter.grad)
    dist.destroy_process_group()


def test_hidden_only_logits_zero_edge_preserves_static_ddp(tmp_path: Path):
    mp.spawn(_worker, args=(str(tmp_path / 'gloo'),), nprocs=2, join=True)


def test_zero_edge_numerical_and_gradient_boundary():
    hidden = torch.randn(2, 4, requires_grad=True)
    logits = torch.randn(2, 7, requires_grad=True)
    connected = connect_unused_logits(hidden, logits)
    assert torch.equal(connected, hidden)
    connected.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
