"""Non-launching filesystem lifecycle owner for Query-State pilot/formal runs."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

_TERMINAL_FILES = {
    "completed": "COMPLETED.json",
    "failed": "FAILED.json",
    "preempted": "PREEMPTED.json",
    "validator_failed": "VALIDATOR_FAILED.json",
}
_HEX = frozenset("0123456789abcdef")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    if path.exists():
        raise FileExistsError(f"immutable controller artifact exists: {path}")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


class QueryStateTrainingController:
    """Own run claim and terminal publication, never Slurm/W&B initialization."""

    def __init__(
        self,
        *,
        run_root: Path,
        controller_root: Path,
        run_identity: str,
        mode: str,
    ) -> None:
        if (
            not isinstance(run_identity, str)
            or len(run_identity) != 64
            or any(char not in _HEX for char in run_identity)
            or mode not in {"pilot", "formal"}
        ):
            raise ValueError("Query-State controller run/mode identity is invalid")
        self.run_root = Path(run_root).resolve()
        self.controller_root = Path(controller_root).resolve()
        self.run_identity = run_identity
        self.mode = mode

    def claim(
        self,
        *,
        resolved_config: Mapping[str, Any],
        command_manifest: Mapping[str, Any],
    ) -> None:
        if self.run_root.exists():
            raise FileExistsError(f"Query-State run output already exists or is claimed: {self.run_root}")
        self.run_root.mkdir(parents=True)
        try:
            self.controller_root.mkdir(parents=True, exist_ok=False)
        except Exception:
            self.run_root.rmdir()
            raise
        _atomic_json(self.run_root / "resolved_config.json", dict(resolved_config))
        _atomic_json(self.run_root / "command_manifest.json", dict(command_manifest))
        readme = (
            "# Query-State training run\n\n"
            f"- mode: `{self.mode}`\n"
            f"- run identity: `{self.run_identity}`\n"
            "- server step/segment/checkpoint evidence is authoritative.\n"
            "- completion never authorizes formal extension, SFT2, rollout, or export.\n"
        )
        (self.run_root / "README.md").write_text(readme, encoding="utf-8")
        _atomic_json(self.controller_root / "CLAIMED.json", {
            "run_identity": self.run_identity,
            "mode": self.mode,
            "run_root": str(self.run_root),
        })

    def verify_existing_claim(self) -> None:
        """Verify that restart/replay reuses the exact live run owner."""

        if not self.run_root.is_dir() or not self.controller_root.is_dir():
            raise FileNotFoundError(
                "Query-State restart/replay requires an existing run claim"
            )
        try:
            claimed = json.loads(
                (self.controller_root / "CLAIMED.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Query-State existing claim evidence is invalid") from error
        expected = {
            "run_identity": self.run_identity,
            "mode": self.mode,
            "run_root": str(self.run_root),
        }
        if claimed != expected:
            raise ValueError("Query-State existing claim identity mismatch")
        if not (self.run_root / "resolved_config.json").is_file() or not (
            self.run_root / "command_manifest.json"
        ).is_file():
            raise ValueError("Query-State existing claim artifacts are incomplete")
        existing = [
            name for name in _TERMINAL_FILES.values()
            if (self.run_root / name).exists() or (self.controller_root / name).exists()
        ]
        if existing:
            raise RuntimeError("Query-State terminal run cannot restart or replay")

    def record_terminal(
        self,
        *,
        status: str,
        details: Mapping[str, Any],
    ) -> Path:
        if status not in _TERMINAL_FILES:
            raise ValueError("unknown Query-State terminal status")
        if not self.run_root.is_dir() or not self.controller_root.is_dir():
            raise RuntimeError("Query-State run must be claimed before terminal publication")
        existing = [name for name in _TERMINAL_FILES.values() if (self.run_root / name).exists()]
        if existing:
            raise RuntimeError("Query-State terminal status is already immutable")
        next_stage = details.get("next_stage")
        if next_stage is not None:
            if str(next_stage).lower().startswith("sft2"):
                raise ValueError("automatic SFT2 next stage is forbidden")
            raise ValueError("automatic formal next stage is forbidden; a new gate is required")
        payload = {
            "run_identity": self.run_identity,
            "mode": self.mode,
            "status": status,
            "details": dict(details),
            "automatic_formal_extension": False,
            "automatic_sft2_authorization": False,
            "automatic_export": False,
        }
        path = _atomic_json(self.run_root / _TERMINAL_FILES[status], payload)
        _atomic_json(self.controller_root / _TERMINAL_FILES[status], payload)
        return path


__all__ = ["QueryStateTrainingController"]
