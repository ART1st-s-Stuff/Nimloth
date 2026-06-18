"""WM step helpers used under DDP."""

from __future__ import annotations

import torch

from nimloth.training.sft2 import step
from nimloth.training.sft2.step import compute_step_wm_loss, wm_eligible_indices


def test_wm_eligible_indices_skips_terminal_steps() -> None:
    items = [
        {"next_messages": None},
        {"next_messages": [{"role": "user", "content": "ok"}]},
    ]
    assert wm_eligible_indices(items) == [1]


def test_wm_eligible_indices_all_terminal() -> None:
    items = [{"next_messages": None}, {"next_messages": None}]
    assert wm_eligible_indices(items) == []


def test_terminal_only_wm_loss_runs_dummy_aux_forwards(monkeypatch) -> None:
    """Terminal-only ranks must still call aux forwards for DDP synchronization."""

    class CountingStateProj(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Linear(3, 2, bias=False)
            self.calls = 0

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return self.net(hidden)

    class CountingWMPredictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Linear(2, 2, bias=False)
            self.calls = 0

        def forward(self, state_emb: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            assert action_indices.dtype == torch.long
            return self.net(state_emb)

    def fake_build_qwen_batch(items, processor, max_length):
        assert items == [{"messages": [{"role": "user", "content": "terminal"}]}]
        return {"input_ids": torch.tensor([[1]])}

    def fake_extract_qwen_latents(model, enc, token_id_map, device):
        return torch.ones(1, 3, device=device), None

    monkeypatch.setattr(step, "build_qwen_batch", fake_build_qwen_batch)
    monkeypatch.setattr(step, "extract_qwen_latents", fake_extract_qwen_latents)

    state_proj = CountingStateProj()
    wm_predictor = CountingWMPredictor()
    current_latent = torch.randn(2, 3, requires_grad=True)
    loss, metrics = compute_step_wm_loss(
        model=torch.nn.Identity(),
        items=[
            {
                "messages": [{"role": "user", "content": "terminal"}],
                "next_messages": None,
                "action_index": 0,
            }
        ],
        current_latent=current_latent,
        processor=None,
        token_id_map={},
        device=torch.device("cpu"),
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        max_length=16,
    )

    assert metrics == {}
    assert float(loss.detach()) == 0.0
    assert state_proj.calls == 2  # current grad path + no-grad target path, matching real WM path
    assert wm_predictor.calls == 1
    loss.backward()
    assert current_latent.grad is not None
