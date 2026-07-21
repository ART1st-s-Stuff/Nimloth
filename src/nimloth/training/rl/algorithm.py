"""RL 单个 transition batch 的模型前向编排。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nimloth.agent import ActionLogProbReplay, Agent
from nimloth.training.rl.batch import RLBatch
from nimloth.training.rl.objective import RLObjective, RLStepOutput


@dataclass(frozen=True)
class RLAlgorithm:
    """组织 WM/value 前向、可选 policy replay 与 RL objective。"""

    agent: Agent
    objective: RLObjective
    policy_replay: ActionLogProbReplay | None

    def training_step(self, batch: RLBatch) -> RLStepOutput:
        # RL 的 WM current 分支更新 StateProjector 与 WMPredictor。
        current_state = self.agent.wm.project_state(batch.current_hidden)
        predicted_next_state = self.agent.wm.predict_next_state(
            current_state,
            batch.action_indices,
        )

        # 下一状态只作为固定监督 target，不更新 StateProjector。
        with torch.no_grad():
            target_next_state = self.agent.wm.project_state(batch.next_hidden)

        # 延续既有梯度契约：value loss 只更新 ValueHead。
        action_values = self.agent.wm.predict_action_values(current_state.detach())

        new_log_probs: torch.Tensor | None = None
        action_log_probs: torch.Tensor | None = None
        if self.policy_replay is not None:
            new_log_probs, action_log_probs = self.policy_replay(batch.transitions)

        return self.objective(
            predicted_next_state=predicted_next_state,
            target_next_state=target_next_state,
            action_values=action_values,
            action_indices=batch.action_indices,
            return_targets=batch.return_targets,
            old_log_probs=batch.old_log_probs,
            new_log_probs=new_log_probs,
            action_log_probs=action_log_probs,
        )


__all__ = ["RLAlgorithm"]
