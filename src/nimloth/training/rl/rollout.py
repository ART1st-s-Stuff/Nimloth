"""Online rollout collection: Qwen policy interacting with VAGEN environments.

The rollout collector runs the Qwen policy in the VAGEN navigation environment,
collecting trajectories that include per-frame images, taken actions, and sparse rewards.
Each trajectory is later encoded into WM latent states by the trainer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RolloutTrajectory:
    """One episode collected by the Qwen policy in the environment."""

    record_id: str
    image_paths: list[str] = field(default_factory=list)
    """image_paths[t] = observation *before* taking action t."""
    action_indices: list[int] = field(default_factory=list)
    """action_indices[t] = action taken at step t (0..7)."""
    success: bool = False
    reward: float = 0.0
    split: str = "train"
    messages: list[dict[str, Any]] = field(default_factory=list)
    """Full conversation history (system, user, assistant turns)."""

    @property
    def num_steps(self) -> int:
        return len(self.action_indices)

    def to_record(self) -> dict[str, Any]:
        """Serialize to the Nimloth JSONL record format."""
        return {
            "id": self.record_id,
            "split": self.split,
            "success": self.success,
            "reward": self.reward,
            "messages": self.messages,
            "image_paths": self.image_paths,
            "action_indices": self.action_indices,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RolloutTrajectory":
        return cls(
            record_id=str(record.get("id", "")),
            image_paths=list(record.get("image_paths", [])),
            action_indices=list(record.get("action_indices", [])),
            success=bool(record.get("success", False)),
            reward=float(record.get("reward", 0.0)),
            split=str(record.get("split", "train")),
            messages=list(record.get("messages", [])),
        )


# ---------------------------------------------------------------------------
# Collector interface
# ---------------------------------------------------------------------------


class RolloutCollector(Protocol):
    """Interface for collecting trajectories from an environment."""

    def collect(
        self,
        *,
        num_episodes: int,
        max_steps_per_episode: int = 20,
        output_dir: Path | None = None,
    ) -> list[RolloutTrajectory]:
        """Run ``num_episodes`` episodes and return collected trajectories."""
        ...


# ---------------------------------------------------------------------------
# VAGEN-backed collector (delegates to VAGEN's trainer.val_only rollout)
# ---------------------------------------------------------------------------


class VAGENRolloutCollector:
    """Collect trajectories by running VAGEN in validation-only mode.

    This is a thin wrapper that delegates to VAGEN's existing rollout
    infrastructure.  It expects a VAGEN config and a running environment
    server (AI2-THOR).  The Qwen model loaded by VAGEN is used as the
    behaviour policy — its action prior (``<|action_start|>`` logits argmax)
    selects actions.

    The output JSONL written by VAGEN is parsed into ``RolloutTrajectory``
    objects for the trainer to encode and train on.
    """

    def __init__(
        self,
        vagen_config_path: Path,
        vagen_checkpoint_dir: Path,
        output_root: Path,
    ) -> None:
        self._vagen_config_path = vagen_config_path
        self._vagen_checkpoint_dir = vagen_checkpoint_dir
        self._output_root = output_root

    def collect(
        self,
        *,
        num_episodes: int,
        max_steps_per_episode: int = 20,
        output_dir: Path | None = None,
    ) -> list[RolloutTrajectory]:
        """Run VAGEN val_only rollout and parse the resulting JSONL.

        .. note::
            This method shells out to VAGEN's training CLI.  In production the
            VAGEN rollout is typically launched via Slurm; for interactive /
            single-node RL training the collector is called directly from the
            trainer process.
        """
        out_dir = output_dir or self._output_root
        jsonl_path = out_dir / "0.jsonl"  # val_only writes 0.jsonl

        # Delegate to VAGEN trainer (val_only=True, single validation step).
        # The VAGEN trainer handles Qwen loading, env connection, and rollout.
        _run_vagen_rollout(
            config_path=self._vagen_config_path,
            checkpoint_dir=self._vagen_checkpoint_dir,
            output_dir=out_dir,
            num_episodes=num_episodes,
            max_steps=max_steps_per_episode,
        )

        # Parse resulting JSONL into RolloutTrajectory objects.
        trajectories: list[RolloutTrajectory] = []
        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    trajectories.append(RolloutTrajectory.from_record(json.loads(line)))
        return trajectories


def _run_vagen_rollout(
    config_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    num_episodes: int,
    max_steps: int,
) -> None:
    """Invoke VAGEN's ``main_ppo`` in val_only mode as a subprocess.

    This function is a placeholder — the actual integration depends on the
    VAGEN codebase and the runtime environment (Slurm, Ray, etc.).  In the
    first iteration we reuse the existing ``experiments/training/baseline/``
    Slurm scripts; this function is filled in once we move to inline
    single-process RL training.
    """
    raise NotImplementedError(
        "VAGEN subprocess rollout is not yet implemented. "
        "Use the existing Slurm-based rollout scripts in experiments/training/baseline/ "
        "for the first iteration, or call VAGEN's trainer Python API directly."
    )


# ---------------------------------------------------------------------------
# JSONL-backed collector (reads pre-collected trajectories from disk)
# ---------------------------------------------------------------------------


class JSONLRolloutCollector:
    """Read trajectories from pre-existing JSONL files.

    Used when VAGEN rollout is run externally (e.g. via Slurm) and the RL
    trainer consumes the resulting JSONL files.  Each call to ``collect``
    reads the given output directory's JSONL and returns parsed trajectories.
    """

    def __init__(self) -> None:
        pass

    def collect(
        self,
        *,
        num_episodes: int,
        max_steps_per_episode: int = 20,
        output_dir: Path | None = None,
    ) -> list[RolloutTrajectory]:
        if output_dir is None:
            return []
        jsonl_path = output_dir / "trajectories.jsonl"
        if not jsonl_path.exists():
            return []
        return load_trajectories(jsonl_path)[:num_episodes]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_trajectories(trajectories: list[RolloutTrajectory], output_dir: Path) -> Path:
    """Write trajectories to a Nimloth JSONL file, one record per line."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "trajectories.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_record(), ensure_ascii=False) + "\n")
    return jsonl_path


def load_trajectories(jsonl_path: Path) -> list[RolloutTrajectory]:
    """Read trajectories from a Nimloth JSONL file."""
    trajectories: list[RolloutTrajectory] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajectories.append(RolloutTrajectory.from_record(json.loads(line)))
    return trajectories
