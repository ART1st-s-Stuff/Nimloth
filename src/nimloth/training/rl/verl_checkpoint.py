"""Fail-closed checkpoint views for pinned VERL Transformers 4.49.

Nimloth SFT currently exports Qwen2.5-VL with Transformers 4.55.4.  Its
``text_config`` nesting is not loadable by the pinned VAGEN/VERL 4.49 model
class.  This module creates a separate, immutable compatibility view; it never
mutates the source checkpoint or its weight shards.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping


_PROTOCOL_VERSION = "nimloth-verl-transformers449-view-v1"
_SOURCE_TRANSFORMERS_VERSION = "4.55.4"
_TARGET_TRANSFORMERS_VERSION = "4.49.0"
_SHARED_TEXT_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "hidden_act",
    "rms_norm_eps",
    "rope_theta",
)
_HF_AUXILIARY_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def translate_qwen25vl_config_for_transformers449(
    source_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Flatten a 4.55 Qwen2.5-VL config for the pinned 4.49 class."""

    source = json.loads(json.dumps(dict(source_config)))
    if source.get("model_type") != "qwen2_5_vl":
        raise ValueError(
            "VERL compatibility view requires model_type=qwen2_5_vl"
        )
    version = source.get("transformers_version")
    if version != _SOURCE_TRANSFORMERS_VERSION:
        raise ValueError(
            "VERL compatibility source transformers_version mismatch: "
            f"expected {_SOURCE_TRANSFORMERS_VERSION}, got {version!r}"
        )
    text_config = source.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("Transformers 4.55 Qwen config must contain text_config")
    if text_config.get("model_type") != "qwen2_5_vl_text":
        raise ValueError(
            "Transformers 4.55 Qwen text_config.model_type must be "
            "qwen2_5_vl_text"
        )
    for field in _SHARED_TEXT_FIELDS:
        if field in text_config and source.get(field) != text_config[field]:
            raise ValueError(
                f"top-level and text_config {field} disagree: "
                f"{source.get(field)!r} != {text_config[field]!r}"
            )
    if text_config.get("tie_word_embeddings") is not True:
        raise ValueError(
            "Transformers 4.49 view requires text_config.tie_word_embeddings=true"
        )

    source.pop("text_config")
    # In 4.55 the language model owns this setting.  In 4.49 the Qwen VL
    # config is flat, so leaving the top-level false silently initializes an
    # absent lm_head instead of tying it to model.embed_tokens.
    source["tie_word_embeddings"] = True
    source["transformers_version"] = _TARGET_TRANSFORMERS_VERSION
    return source


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def prepare_transformers449_checkpoint_view(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    weight_mode: str = "hardlink",
) -> dict[str, Any]:
    """Create an atomic 4.49-compatible view over complete HF weight shards."""

    source = Path(source_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source checkpoint directory not found: {source}")
    if output.exists():
        raise FileExistsError(f"output directory must not exist: {output}")
    if weight_mode not in {"hardlink", "copy"}:
        raise ValueError("weight_mode must be hardlink or copy")

    source_config = _load_json(source / "config.json")
    translated_config = translate_qwen25vl_config_for_transformers449(source_config)
    index = _load_json(source / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json requires nonempty weight_map")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise ValueError("checkpoint weight_map must map strings to shard names")
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        shard_path = source / shard
        if shard_path.parent != source or not shard_path.is_file():
            raise FileNotFoundError(f"checkpoint shard missing or unsafe: {shard}")

    manifest: dict[str, Any] = {
        "protocol_version": _PROTOCOL_VERSION,
        "source_dir": str(source),
        "source_config_sha256": _canonical_json_sha256(source_config),
        "translated_config_sha256": _canonical_json_sha256(translated_config),
        "source_transformers_version": source_config["transformers_version"],
        "target_transformers_version": _TARGET_TRANSFORMERS_VERSION,
        "weight_mode": weight_mode,
        "weight_shards": shards,
        "weight_tensor_count": len(weight_map),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.mkdir()
        (temporary / "config.json").write_text(
            json.dumps(translated_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(
            source / "model.safetensors.index.json",
            temporary / "model.safetensors.index.json",
        )
        for filename in _HF_AUXILIARY_FILES:
            path = source / filename
            if path.is_file():
                shutil.copy2(path, temporary / filename)
        for shard in shards:
            if weight_mode == "hardlink":
                os.link(source / shard, temporary / shard)
            else:
                shutil.copy2(source / shard, temporary / shard)
        (temporary / "nimloth_verl_view.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest
