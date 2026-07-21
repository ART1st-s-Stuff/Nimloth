"""RL 单批算法所需的模型执行契约。"""

from __future__ import annotations

from dataclasses import dataclass

from nimloth.agent import ActionLogProbReplay, Agent


@dataclass(frozen=True)
class RLModelRuntime:
    """显式提供在线 Agent 与可选 actor 概率重放能力。"""

    agent: Agent
    policy_replay: ActionLogProbReplay | None


__all__ = ["RLModelRuntime"]
