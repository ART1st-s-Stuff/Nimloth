"""Strict identity manifest for the state-interface-v2 canary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from nimloth.training.sft1.config import STATE_INTERFACE_OBJECTIVE_VERSION
from nimloth.training.verl.source import (
    PINNED_VAGEN_COMMIT,
    PINNED_VERL_COMMIT,
    verify_pinned_vagen_verl_source,
)


SFT1_V2_MANIFEST_SCHEMA = "nimloth_sft1_state_v2_manifest_v1"
SFT1_V2_SUPERVISION_SCHEMA = "nimloth_sft1_state_v2_supervision_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SFT1V2Manifest:
    schema: str
    objective_version: str
    supervision_schema: str
    vagen_commit: str
    verl_commit: str
    actor_checkpoint_sha256: str
    processor_sha256: str
    prompt_template_sha256: str
    token_table_sha256: str
    dino_identity_sha256: str
    trajectory_sha256: str
    teacher_cache_sha256: str
    latent_query_mode: str
    query_count: int
    action_count: int
    action_token_ids: tuple[int, ...]
    train_split: str
    external_validation_split: str

    @property
    def identity(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _strict_fields(raw: Mapping[str, Any], expected: set[str]) -> None:
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown:
        raise ValueError(f"unknown SFT1-v2 manifest field: {unknown[0]}")
    if missing:
        raise ValueError(f"missing SFT1-v2 manifest field: {missing[0]}")


def parse_sft1_v2_manifest(raw: Mapping[str, Any]) -> SFT1V2Manifest:
    if not isinstance(raw, Mapping):
        raise ValueError("SFT1-v2 manifest must be a mapping")
    fields = set(SFT1V2Manifest.__dataclass_fields__)
    _strict_fields(raw, fields)
    hashes = {
        "actor_checkpoint_sha256",
        "processor_sha256",
        "prompt_template_sha256",
        "token_table_sha256",
        "dino_identity_sha256",
        "trajectory_sha256",
        "teacher_cache_sha256",
    }
    for field in hashes:
        if not isinstance(raw[field], str) or _SHA256_RE.fullmatch(raw[field]) is None:
            raise ValueError(f"manifest {field} must be a lowercase SHA256 digest")
    if raw["schema"] != SFT1_V2_MANIFEST_SCHEMA:
        raise ValueError("unsupported SFT1-v2 manifest schema")
    if raw["objective_version"] != STATE_INTERFACE_OBJECTIVE_VERSION:
        raise ValueError("manifest contains an old state objective")
    if raw["supervision_schema"] != SFT1_V2_SUPERVISION_SCHEMA:
        raise ValueError("unsupported SFT1-v2 supervision schema")
    if raw["vagen_commit"] != PINNED_VAGEN_COMMIT:
        raise ValueError("manifest VAGEN commit differs from the pinned source")
    if raw["verl_commit"] != PINNED_VERL_COMMIT:
        raise ValueError("manifest VERL commit differs from the pinned source")
    if (
        raw["latent_query_mode"] != "inject"
        or isinstance(raw["query_count"], bool)
        or not isinstance(raw["query_count"], int)
        or raw["query_count"] != 16
    ):
        raise ValueError("manifest requires the injected K16 query contract")
    if (
        isinstance(raw["action_count"], bool)
        or not isinstance(raw["action_count"], int)
        or raw["action_count"] != 8
    ):
        raise ValueError("manifest requires the eight-action contract")
    action_token_ids = raw["action_token_ids"]
    if (
        not isinstance(action_token_ids, (list, tuple))
        or len(action_token_ids) != 8
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in action_token_ids
        )
        or len(set(action_token_ids)) != 8
    ):
        raise ValueError("manifest action_token_ids must contain eight distinct integers")
    train_split = raw["train_split"]
    validation_split = raw["external_validation_split"]
    if (
        not isinstance(train_split, str)
        or not train_split
        or not isinstance(validation_split, str)
        or not validation_split
        or train_split == validation_split
    ):
        raise ValueError("manifest train/external-validation split identities are invalid")
    return SFT1V2Manifest(
        **{
            **{field: raw[field] for field in fields - {"action_token_ids"}},
            "action_token_ids": tuple(action_token_ids),
        }
    )


def load_sft1_v2_manifest(path: Path) -> SFT1V2Manifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_sft1_v2_manifest(raw)


__all__ = [
    "PINNED_VAGEN_COMMIT",
    "PINNED_VERL_COMMIT",
    "SFT1V2Manifest",
    "SFT1_V2_MANIFEST_SCHEMA",
    "SFT1_V2_SUPERVISION_SCHEMA",
    "load_sft1_v2_manifest",
    "parse_sft1_v2_manifest",
    "verify_pinned_vagen_verl_source",
]
