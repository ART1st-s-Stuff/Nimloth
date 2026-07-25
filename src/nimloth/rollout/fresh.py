"""一次性 fresh-policy rollout artifact 契约。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch.distributed as dist

from nimloth.rollout.source import JSONLRolloutCollector


def file_artifact_fingerprint(path: Path) -> str:
    """Hash one immutable rollout artifact by bytes."""

    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"rollout artifact does not exist: {artifact}")
    digest = hashlib.sha256()
    with artifact.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_artifact_fingerprint(model_dir: Path) -> str:
    """对影响 policy 行为的 HF artifact 计算内容指纹。"""

    root = Path(model_dir).resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"policy artifact has no config.json: {root}")
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name.endswith((".json", ".safetensors", ".bin"))
            or path.name in {"chat_template.jinja", "merges.txt", "vocab.json"}
        )
    )
    digest = hashlib.sha256()
    for path in candidates:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def auxiliary_artifact_fingerprint(path: Path) -> str:
    """Hash one planner module file/directory including its relative layout."""

    root = Path(path).resolve()
    if root.is_file():
        files = (root,)
        base = root.parent
    elif root.is_dir():
        files = tuple(sorted(candidate for candidate in root.rglob("*") if candidate.is_file()))
        base = root
    else:
        raise FileNotFoundError(f"planner artifact does not exist: {root}")
    digest = hashlib.sha256()
    for candidate in files:
        digest.update(str(candidate.relative_to(base)).encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FreshRolloutManifest:
    """vLLM rollout 与随后唯一一次 PPO update 的连接证据。"""

    format_version: int
    policy_fingerprint: str
    policy_path: str
    trajectory_path: str
    trajectory_fingerprint: str
    num_trajectories: int
    created_at: str
    planner_fingerprints: dict[str, str] = field(default_factory=dict)
    planner_paths: dict[str, str] = field(default_factory=dict)
    reference_policy_fingerprint: str | None = None
    reference_policy_path: str | None = None
    behavior_trajectory_path: str = ""
    behavior_trajectory_fingerprint: str = ""

    @classmethod
    def create(
        cls,
        *,
        policy_path: Path,
        trajectory_path: Path,
        num_trajectories: int,
        planner_artifacts: dict[str, Path] | None = None,
    ) -> "FreshRolloutManifest":
        artifacts = planner_artifacts or {}
        resolved_trajectory = Path(trajectory_path).resolve()
        trajectory_fingerprint = file_artifact_fingerprint(resolved_trajectory)
        return cls(
            format_version=4,
            policy_fingerprint=policy_artifact_fingerprint(policy_path),
            policy_path=str(Path(policy_path).resolve()),
            trajectory_path=str(resolved_trajectory),
            trajectory_fingerprint=trajectory_fingerprint,
            num_trajectories=int(num_trajectories),
            created_at=datetime.now(timezone.utc).isoformat(),
            planner_fingerprints={
                name: auxiliary_artifact_fingerprint(path)
                for name, path in sorted(artifacts.items())
            },
            planner_paths={
                name: str(Path(path).resolve())
                for name, path in sorted(artifacts.items())
            },
            behavior_trajectory_path=str(resolved_trajectory),
            behavior_trajectory_fingerprint=trajectory_fingerprint,
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(json.dumps(self.__dict__, indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary_path.replace(path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def with_reference(
        self,
        *,
        reference_policy_path: Path,
        trajectory_path: Path,
    ) -> "FreshRolloutManifest":
        """Bind frozen reference provenance and the enriched trajectory artifact."""

        if self.reference_policy_fingerprint is not None:
            raise ValueError("fresh rollout manifest already has a reference policy")
        resolved_trajectory = Path(trajectory_path).resolve()
        return replace(
            self,
            trajectory_path=str(resolved_trajectory),
            trajectory_fingerprint=file_artifact_fingerprint(resolved_trajectory),
            reference_policy_fingerprint=policy_artifact_fingerprint(
                reference_policy_path
            ),
            reference_policy_path=str(Path(reference_policy_path).resolve()),
        )

    def validate_trajectory_artifacts(self) -> None:
        """Reject changed behavior or reference-enriched trajectory bytes."""

        actual = file_artifact_fingerprint(Path(self.trajectory_path))
        if actual != self.trajectory_fingerprint:
            raise ValueError(
                "fresh rollout trajectory fingerprint mismatch: "
                f"manifest={self.trajectory_fingerprint}, current={actual}"
            )
        behavior_actual = file_artifact_fingerprint(
            Path(self.behavior_trajectory_path)
        )
        if behavior_actual != self.behavior_trajectory_fingerprint:
            raise ValueError(
                "fresh behavior trajectory fingerprint mismatch: "
                f"manifest={self.behavior_trajectory_fingerprint}, "
                f"current={behavior_actual}"
            )

    @classmethod
    def read(cls, path: Path) -> "FreshRolloutManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = cls(**payload)
        if manifest.format_version != 4:
            raise ValueError(
                f"unsupported fresh rollout manifest version {manifest.format_version}"
            )
        return manifest


class FreshJSONLRolloutCollector(JSONLRolloutCollector):
    """仅允许与当前 policy artifact 匹配的 JSONL 被 PPO 消费一次。"""

    def __init__(
        self,
        manifest_path: Path,
        *,
        model_path: Path,
        planner_artifacts: dict[str, Path] | None = None,
        reference_model_path: Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = FreshRolloutManifest.read(self.manifest_path)
        self.model_path = Path(model_path).resolve()
        self.planner_artifacts = planner_artifacts or {}
        self.reference_model_path = (
            Path(reference_model_path).resolve()
            if reference_model_path is not None
            else None
        )
        super().__init__([Path(self.manifest.trajectory_path)], loop=False)

    def validate_policy(self) -> None:
        self.manifest.validate_trajectory_artifacts()
        actual = policy_artifact_fingerprint(self.model_path)
        if actual != self.manifest.policy_fingerprint:
            raise ValueError(
                "fresh rollout policy fingerprint mismatch: "
                f"manifest={self.manifest.policy_fingerprint}, current={actual}"
            )
        if set(self.planner_artifacts) != set(self.manifest.planner_fingerprints):
            raise ValueError(
                "fresh rollout planner artifact set does not match the manifest"
            )
        for name, path in self.planner_artifacts.items():
            actual = auxiliary_artifact_fingerprint(path)
            expected = self.manifest.planner_fingerprints[name]
            if actual != expected:
                raise ValueError(
                    f"fresh rollout planner fingerprint mismatch for {name}: "
                    f"manifest={expected}, current={actual}"
                )
        if self.manifest.reference_policy_fingerprint is None:
            if self.reference_model_path is not None:
                raise ValueError(
                    "fresh rollout manifest has no frozen reference policy"
                )
        else:
            if self.reference_model_path is None:
                raise ValueError(
                    "reference-enriched fresh rollout requires reference model path"
                )
            actual_reference = policy_artifact_fingerprint(
                self.reference_model_path
            )
            if actual_reference != self.manifest.reference_policy_fingerprint:
                raise ValueError(
                    "fresh rollout reference fingerprint mismatch: "
                    f"manifest={self.manifest.reference_policy_fingerprint}, "
                    f"current={actual_reference}"
                )

    @property
    def consumption_path(self) -> Path:
        return self.manifest_path.with_suffix(
            self.manifest_path.suffix + ".consumption.json"
        )

    def _distributed_rank_zero(self, operation) -> None:  # type: ignore[no-untyped-def]
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        error_message: str | None = None
        if rank == 0:
            try:
                operation()
            except Exception as error:
                error_message = str(error)
        if distributed:
            messages = [error_message]
            dist.broadcast_object_list(messages, src=0)
            error_message = messages[0]
        if error_message is not None:
            raise RuntimeError(error_message)

    def begin_consumption(self, *, output_dir: Path, global_step: int) -> str:
        """Atomically mark a validated batch as in-progress before optimizer work."""

        manifest_fingerprint = file_artifact_fingerprint(self.manifest_path)
        consumption_id = hashlib.sha256(
            (
                manifest_fingerprint
                + str(Path(output_dir).resolve())
                + str(global_step)
            ).encode("utf-8")
        ).hexdigest()

        def begin() -> None:
            path = self.consumption_path
            temporary_path: Path | None = None
            try:
                with NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    json.dump(
                        {
                            "state": "in_progress",
                            "consumption_id": consumption_id,
                            "manifest_fingerprint": manifest_fingerprint,
                            "trajectory_fingerprint": (
                                self.manifest.trajectory_fingerprint
                            ),
                            "policy_fingerprint": self.manifest.policy_fingerprint,
                            "output_dir": str(Path(output_dir).resolve()),
                            "starting_global_step": int(global_step),
                        },
                        stream,
                        indent=2,
                    )
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary_path, path)
            except FileExistsError as error:
                payload = json.loads(path.read_text(encoding="utf-8"))
                raise RuntimeError(
                    "fresh rollout consumption already exists: "
                    f"state={payload.get('state')}, path={path}"
                ) from error
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        self._distributed_rank_zero(begin)
        return consumption_id

    def abort_consumption(self, consumption_id: str) -> None:
        """Release a claim only when no optimizer step could have completed."""

        def abort() -> None:
            path = self.consumption_path
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("state") != "in_progress"
                or payload.get("consumption_id") != consumption_id
            ):
                raise RuntimeError("fresh rollout consumption abort state mismatch")
            path.unlink()

        self._distributed_rank_zero(abort)

    def commit_consumption(
        self,
        consumption_id: str,
        *,
        checkpoint_path: Path,
        global_step: int,
    ) -> None:
        """Commit consumption only after a durable post-update checkpoint exists."""

        def commit() -> None:
            path = self.consumption_path
            payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("state") != "in_progress"
                or payload.get("consumption_id") != consumption_id
            ):
                raise RuntimeError("fresh rollout consumption commit state mismatch")
            checkpoint = Path(checkpoint_path).resolve()
            if not checkpoint.is_dir() or not (checkpoint / "rl_state.pt").is_file():
                raise RuntimeError(
                    f"fresh rollout commit requires complete checkpoint: {checkpoint}"
                )
            committed = {
                **payload,
                "state": "committed",
                "checkpoint_path": str(checkpoint),
                "committed_global_step": int(global_step),
            }
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
                    stream.write(json.dumps(committed, indent=2) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

        self._distributed_rank_zero(commit)

    def collect(self, **kwargs):  # type: ignore[no-untyped-def]
        if self._call_count > 0:
            raise RuntimeError("fresh rollout collector can only be consumed once")
        self.manifest.validate_trajectory_artifacts()
        requested = int(kwargs.get("num_episodes", 0))
        if requested != self.manifest.num_trajectories:
            raise ValueError(
                "fresh rollout must consume the complete manifest batch: "
                f"requested={requested}, manifest={self.manifest.num_trajectories}"
            )
        trajectories = super().collect(**kwargs)
        if len(trajectories) != self.manifest.num_trajectories:
            raise ValueError(
                "fresh rollout trajectory count does not match manifest: "
                f"loaded={len(trajectories)}, manifest={self.manifest.num_trajectories}"
            )
        return trajectories


__all__ = [
    "FreshJSONLRolloutCollector",
    "FreshRolloutManifest",
    "auxiliary_artifact_fingerprint",
    "file_artifact_fingerprint",
    "policy_artifact_fingerprint",
]
