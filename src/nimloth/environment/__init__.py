"""Nimloth environment abstraction layer.

Environment managers provide batch lifecycle management (create, reset, step,
close) over a fleet of parallel task environments.  The common module defines
the abstract interface and shared types; concrete backends live in subpackages
(navigation, alfworld, etc.).
"""

from nimloth.environment.common import BaseEnvManager, EnvConfig, StepResult, TrajectoryRecording
from nimloth.environment.navigation import NavigationEnvManager

__all__ = [
    "BaseEnvManager",
    "EnvConfig",
    "NavigationEnvManager",
    "StepResult",
    "TrajectoryRecording",
]
