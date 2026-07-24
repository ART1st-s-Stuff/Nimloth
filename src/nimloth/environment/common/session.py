"""Agent episode runner 所需的最小环境 session 协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from nimloth.environment.common.action_space import DiscreteActionSpace


@dataclass(frozen=True)
class EnvironmentObservation:
    text: str
    image: Any
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentStep:
    observation: EnvironmentObservation
    reward: float
    done: bool
    success: bool
    info: dict[str, Any] = field(default_factory=dict)


class EnvironmentSession(Protocol):
    """单个 episode 的环境接口；具体后端负责动作提交格式。"""

    @property
    def action_space(self) -> DiscreteActionSpace:
        ...

    @property
    def system_prompt(self) -> str:
        ...

    def reset(self, *, seed: int) -> EnvironmentObservation:
        ...

    def step(self, *, action_index: int, response: str) -> EnvironmentStep:
        ...

    def close(self) -> None:
        ...
