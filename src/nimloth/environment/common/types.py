"""Common types shared across environment managers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    """Result of a single environment step."""

    env_id: int
    obs_str: str
    image: Any  # PIL.Image
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvConfig:
    """Minimal task configuration for environment reset."""

    env_name: str
    env_config: dict[str, Any]
    seed: int


@dataclass
class TrajectoryRecording:
    """Recorded steps for a single environment episode."""

    env_id: int
    env_name: str = ""
    eval_set: str = ""
    seed: int = 0
    instruction: str = ""
    image_paths: list[str] = field(default_factory=list)
    """image_paths[t] = observation seen *before* taking action_indices[t]."""
    action_indices: list[int] = field(default_factory=list)
    """Action indices (0-7, Nimloth format)."""
    action_names: list[str] = field(default_factory=list)
    """Action names (e.g. ``move_forward``) for debugging."""
    messages: list[dict[str, Any]] = field(default_factory=list)
    """Full conversation: system, user, assistant per turn."""
    reward: float = 0.0
    success: bool = False
    done: bool = False
    num_steps: int = 0

    @property
    def total_steps(self) -> int:
        return len(self.action_indices)
