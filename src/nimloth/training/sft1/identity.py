"""CPU-verifiable ID176 processor/token/query identity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

from nimloth.latent import LatentActionTokens, latent_state_tokens
from nimloth.training.sft1.data import sha256_file


@dataclass(frozen=True)
class SFT1V2ProcessorIdentity:
    processor_sha256: str
    tokenizer_sha256: str
    prompt_template_sha256: str
    token_table_sha256: str
    action_token_ids: tuple[int, ...]


def _bundle_digest(root: Path, names: Sequence[str]) -> str:
    hashes: dict[str, str] = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"ID176 identity file is missing: {path}")
        hashes[name] = sha256_file(path)
    return hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit_id176_processor_identity(actor_checkpoint: Path) -> SFT1V2ProcessorIdentity:
    root = Path(actor_checkpoint)
    processor_sha = _bundle_digest(root, (
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ))
    tokenizer_sha = _bundle_digest(root, (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
    ))
    prompt_sha = sha256_file(root / "chat_template.jinja")
    added = json.loads((root / "added_tokens.json").read_text(encoding="utf-8"))
    if not isinstance(added, dict):
        raise ValueError("ID176 added_tokens.json must be a token-to-id mapping")
    token_ids = {str(token): int(value) for token, value in added.items()}
    tokens = LatentActionTokens()
    required = (
        *latent_state_tokens(16, tokens),
        tokens.action_start,
        tokens.action_end,
        *tokens.action_tokens,
    )
    missing = [token for token in required if token not in token_ids]
    if missing:
        raise ValueError(f"ID176 token table is missing {missing[0]}")
    table = {token: token_ids[token] for token in required}
    table_sha = hashlib.sha256(
        json.dumps(table, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    action_ids = tuple(table[token] for token in tokens.action_tokens)
    if len(set(action_ids)) != 8:
        raise ValueError("ID176 action token IDs are not distinct")
    return SFT1V2ProcessorIdentity(
        processor_sha256=processor_sha,
        tokenizer_sha256=tokenizer_sha,
        prompt_template_sha256=prompt_sha,
        token_table_sha256=table_sha,
        action_token_ids=action_ids,
    )


__all__ = ["SFT1V2ProcessorIdentity", "audit_id176_processor_identity"]
