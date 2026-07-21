"""terminal-only batch 的统一 Agent 调用结构测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from nimloth.agent import Agent, AgentBatch, AgentTarget
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.training.sft2.algorithm import SFT2Algorithm
from nimloth.training.sft2.objective import SFT2Objective
from nimloth.wm.model import WorldModel


def test_terminal_only_batch_runs_backbone_and_wm_with_masked_zero_loss() -> None:
    class CountingBackbone(Backbone):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = torch.nn.Identity()
            self.calls = 0

        @property
        def model(self) -> torch.nn.Module:
            return self.language_model

        def forward(self, batch: BackboneBatch, **_kwargs) -> BackboneOutput:
            self.calls += 1
            return BackboneOutput(batch.tensors["hidden"])

        def with_model(self, model: torch.nn.Module) -> "CountingBackbone":
            return self

        def save_pretrained(self, output_dir: Path, **_kwargs) -> None:
            raise NotImplementedError

    backbone = CountingBackbone()
    class Predictor(torch.nn.Linear):
        def forward(
            self,
            state: torch.Tensor,
            _action_indices: torch.Tensor,
        ) -> torch.Tensor:
            return super().forward(state)

    predictor = Predictor(2, 2, bias=False)
    agent = Agent(
        backbone=backbone,
        wm=WorldModel(
            state_proj=torch.nn.Linear(3, 2, bias=False),
            wm_predictor=predictor,
            value_head=torch.nn.Linear(2, 3),
        ),
    )
    algorithm = SFT2Algorithm(
        agent=agent,
        target=AgentTarget(agent),
        objective=SFT2Objective(
            sigreg=None,
            sigreg_weight=0.0,
            value_weight=1.0,
            ce_weight=1.0,
            value_rank_margin=0.1,
            value_rank_weight=1.0,
        ),
    )
    current = torch.randn(1, 3, requires_grad=True)
    batch = AgentBatch(
        current=BackboneBatch({"hidden": current}),
        next=BackboneBatch({"hidden": torch.randn(1, 3)}),
        action_indices=torch.tensor([0]),
        value_targets=torch.tensor([0.0]),
        next_indices=torch.tensor([0]),
        non_terminal_mask=torch.tensor([False]),
        trajectory_steps=(("terminal", 0),),
    )

    output = algorithm.training_step(batch, wm_weight=1.0)

    assert backbone.calls == 2
    assert output.metrics.get("wm_mse") is None
    assert output.losses["sigreg"] is None
    assert float(output.losses["wm"].detach()) == 0.0
    output.losses["wm"].backward()
    assert current.grad is not None
    assert predictor.weight.grad is not None
