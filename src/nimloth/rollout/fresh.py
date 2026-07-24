"""一次性 fresh-policy rollout artifact 契约。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch.distributed as dist

from nimloth.rollout.source import JSONLRolloutCollector


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


@dataclass(frozen=True)
class FreshRolloutManifest:
    """vLLM rollout 与随后唯一一次 PPO update 的连接证据。"""

    format_version: int
    policy_fingerprint: str
    policy_path: str
    trajectory_path: str
    num_trajectories: int
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        policy_path: Path,
        trajectory_path: Path,
        num_trajectories: int,
    ) -> "FreshRolloutManifest":
        return cls(
            format_version=1,
            policy_fingerprint=policy_artifact_fingerprint(policy_path),
            policy_path=str(Path(policy_path).resolve()),
            trajectory_path=str(Path(trajectory_path).resolve()),
            num_trajectories=int(num_trajectories),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "FreshRolloutManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = cls(**payload)
        if manifest.format_version != 1:
            raise ValueError(
                f"unsupported fresh rollout manifest version {manifest.format_version}"
            )
        return manifest


class FreshJSONLRolloutCollector(JSONLRolloutCollector):
    """仅允许与当前 policy artifact 匹配的 JSONL 被 PPO 消费一次。"""

    def __init__(self, manifest_path: Path, *, model_path: Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = FreshRolloutManifest.read(self.manifest_path)
        self.model_path = Path(model_path).resolve()
        super().__init__([Path(self.manifest.trajectory_path)], loop=False)

    def validate_policy(self) -> None:
        actual = policy_artifact_fingerprint(self.model_path)
        if actual != self.manifest.policy_fingerprint:
            raise ValueError(
                "fresh rollout policy fingerprint mismatch: "
                f"manifest={self.manifest.policy_fingerprint}, current={actual}"
            )

    def _claim_once(self) -> None:
        claim_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".consumed")
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        error_message: str | None = None
        if rank == 0:
            try:
                descriptor = claim_path.open("x", encoding="utf-8")
            except FileExistsError:
                error_message = f"fresh rollout manifest was already consumed: {claim_path}"
            else:
                with descriptor:
                    descriptor.write(self.manifest.policy_fingerprint + "\n")
        if distributed:
            messages = [error_message]
            dist.broadcast_object_list(messages, src=0)
            error_message = messages[0]
        if error_message is not None:
            raise RuntimeError(error_message)

    def collect(self, **kwargs):  # type: ignore[no-untyped-def]
        if self._call_count > 0:
            raise RuntimeError("fresh rollout collector can only be consumed once")
        self._claim_once()
        return super().collect(**kwargs)


__all__ = [
    "FreshJSONLRolloutCollector",
    "FreshRolloutManifest",
    "policy_artifact_fingerprint",
]
