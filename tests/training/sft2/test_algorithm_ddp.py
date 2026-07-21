"""SFT2 algorithm 在 terminal-only DDP rank 上的模块调用契约。"""

from __future__ import annotations

import torch

from nimloth.backbone.qwen25vl.transition import (
    QwenTransitionEncoder,
    QwenTransitionMessages,
)
from nimloth.model import NimlothModel
from nimloth.training.sft2.algorithm import SFT2Algorithm, SFT2Mode
from nimloth.training.sft2.data.batch import SFT2Batch, SFT2Transition
from nimloth.wm.model import WorldModel


def test_terminal_only_batch_runs_ddp_aligned_module_forwards(monkeypatch) -> None:
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

        def forward(
            self,
            state: torch.Tensor,
            action_indices: torch.Tensor,
        ) -> torch.Tensor:
            self.calls += 1
            assert action_indices.dtype == torch.long
            return self.net(state)

    current_hidden = torch.randn(1, 3, requires_grad=True)
    extract_calls = 0

    def fake_extract(_model, _encoding, _token_id_map, _device, **_kwargs):
        nonlocal extract_calls
        extract_calls += 1
        if extract_calls == 1:
            return current_hidden, torch.tensor(0.0, requires_grad=True)
        return torch.ones(1, 3), None

    def fake_build(items, _processor, _max_length, **_kwargs):
        assert items == [
            {"messages": [{"role": "user", "content": "terminal"}]}
        ]
        return {"input_ids": torch.tensor([[1]])}

    monkeypatch.setattr(
        "nimloth.backbone.qwen25vl.transition.extract_qwen_latents",
        fake_extract,
    )
    monkeypatch.setattr(
        "nimloth.backbone.qwen25vl.transition.build_qwen_batch",
        fake_build,
    )

    state_proj = CountingStateProj()
    wm_predictor = CountingWMPredictor()
    llm = torch.nn.Identity()
    nimloth_model = NimlothModel(
        llm=llm,
        wm=WorldModel(
            state_proj=state_proj,
            wm_predictor=wm_predictor,
            value_head=torch.nn.Linear(2, 8),
        ),
    )
    algorithm = SFT2Algorithm(
        model=nimloth_model,
        qwen=QwenTransitionEncoder(
            processor=None,
            token_id_map={},
            device=torch.device("cpu"),
            max_length=16,
            pad_token_id=0,
        ),
        sigreg=None,
    )
    batch = SFT2Batch(
        transitions=(
            SFT2Transition(
                identifier="terminal:0",
                record_id="terminal",
                step_index=0,
                action_index=0,
                value_target=0.0,
                success=False,
                qwen=QwenTransitionMessages(
                    current=[{"role": "user", "content": "terminal"}],
                    next=None,
                ),
            ),
        ),
        current_encoding={"input_ids": torch.tensor([[1]])},
        cached_next=None,
    )

    losses = algorithm.compute(batch, mode=SFT2Mode.TRAIN)

    assert losses.sigreg is None
    assert losses.metrics.get("wm_mse") is None
    assert float(losses.dynamics.detach()) == 0.0
    # dynamics current/target 各一次，随后 value 使用当前 state 一次。
    assert state_proj.calls == 3
    assert wm_predictor.calls == 1
    losses.dynamics.backward()
    assert current_hidden.grad is not None
