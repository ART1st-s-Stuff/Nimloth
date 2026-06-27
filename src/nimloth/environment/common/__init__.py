"""Common environment interfaces and types.

Provides the abstract :class:`BaseEnvManager` and shared data structures
(:class:`StepResult`, :class:`EnvConfig`, :class:`TrajectoryRecording`).
"""

from nimloth.environment.common.base import BaseEnvManager
from nimloth.environment.common.types import EnvConfig, StepResult, TrajectoryRecording

__all__ = [
    "BaseEnvManager",
    "EnvConfig",
    "StepResult",
    "TrajectoryRecording",
]
