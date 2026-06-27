"""Abstract base class for environment managers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nimloth.environment.common.types import EnvConfig, StepResult, TrajectoryRecording


class BaseEnvManager(ABC):
    """Interface for managing a collection of parallel environments.

    Each manager handles the full lifecycle: create → reset → step … → close.
    Subclasses implement the concrete environment backend (e.g. AI2-THOR
    navigation, ALFWorld, etc.).
    """

    @abstractmethod
    def reset(self, env_configs: list[EnvConfig]) -> list[dict[str, Any]]:
        """(Re-)initialise environments for a new batch of tasks.

        Args:
            env_configs: One config per environment slot.  Slot indices are
                used as ``env_id`` throughout the episode.

        Returns:
            Initial observation dict per environment (same length and order as
            ``env_configs``).
        """
        ...

    @abstractmethod
    def step(self, actions: list[str]) -> list[StepResult]:
        """Execute one action in each active environment.

        Args:
            actions: Action string per active environment (same length and
                order as the *active* environments as reported by
                :meth:`active_env_ids`).

        Returns:
            :class:`StepResult` per active environment.
        """
        ...

    @abstractmethod
    def active_env_ids(self) -> list[int]:
        """Return the ids of environments that are not yet done."""
        ...

    @abstractmethod
    def is_done(self, env_id: int) -> bool:
        """Check whether a specific environment has finished."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all managed environments and free resources."""
        ...

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @abstractmethod
    def start_recording(self, env_id: int, system_prompt: str = "") -> None:
        """Begin recording a new episode for *env_id*."""
        ...

    @abstractmethod
    def record_step(self, env_id: int, result: StepResult) -> None:
        """Append one step to the recording for *env_id*."""
        ...

    @abstractmethod
    def get_trajectories(self) -> list[TrajectoryRecording]:
        """Return all completed trajectory recordings.

        Only trajectories whose environments have finished (``done=True``)
        SHOULD be included; in-progress episodes MAY be skipped.
        """
        ...

    @abstractmethod
    def clear_recordings(self) -> None:
        """Discard all in-memory recordings (e.g. between iterations)."""
        ...
