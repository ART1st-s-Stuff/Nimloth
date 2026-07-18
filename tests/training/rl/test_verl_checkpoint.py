from __future__ import annotations

import json
from pathlib import Path

import pytest


def _config() -> dict:
    return {
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "hidden_size": 16,
        "intermediate_size": 32,
        "model_type": "qwen2_5_vl",
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "tie_word_embeddings": False,
        "transformers_version": "4.55.4",
        "vocab_size": 128,
        "vision_config": {"depth": 2, "hidden_size": 8},
        "text_config": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "model_type": "qwen2_5_vl_text",
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "tie_word_embeddings": True,
            "vocab_size": 128,
        },
    }


def test_translate_qwen455_config_to_transformers449_is_strict() -> None:
    from nimloth.training.rl.verl_checkpoint import (
        translate_qwen25vl_config_for_transformers449,
    )

    source = _config()
    translated = translate_qwen25vl_config_for_transformers449(source)
    assert "text_config" not in translated
    assert translated["transformers_version"] == "4.49.0"
    # 4.55 stores the language-model tying contract in text_config.  The 4.49
    # flat config must inherit it or lm_head is silently initialized.
    assert translated["tie_word_embeddings"] is True
    assert translated["vocab_size"] == 128
    assert source["text_config"]["tie_word_embeddings"] is True

    bad = _config()
    bad["text_config"]["hidden_size"] = 17
    with pytest.raises(ValueError, match="hidden_size"):
        translate_qwen25vl_config_for_transformers449(bad)


def test_prepare_transformers449_view_hardlinks_complete_checkpoint(tmp_path: Path) -> None:
    from nimloth.training.rl.verl_checkpoint import (
        prepare_transformers449_checkpoint_view,
    )

    source = tmp_path / "source"
    output = tmp_path / "view"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (source / "model-00001-of-00002.safetensors").write_bytes(b"first-shard")
    (source / "model-00002-of-00002.safetensors").write_bytes(b"second-shard")
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 23},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    (source / "README.md").write_text("not part of the HF view", encoding="utf-8")

    manifest = prepare_transformers449_checkpoint_view(source, output)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["transformers_version"] == "4.49.0"
    assert "text_config" not in config
    assert (output / "tokenizer.json").read_text(encoding="utf-8") == "{}"
    assert not (output / "README.md").exists()
    assert (output / "model-00001-of-00002.safetensors").stat().st_ino == (
        source / "model-00001-of-00002.safetensors"
    ).stat().st_ino
    assert manifest["protocol_version"] == "nimloth-verl-transformers449-view-v1"
    assert manifest["source_transformers_version"] == "4.55.4"
    assert manifest["target_transformers_version"] == "4.49.0"
    assert manifest["weight_shards"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert json.loads((output / "nimloth_verl_view.json").read_text(encoding="utf-8")) == manifest


def test_prepare_transformers449_view_rejects_missing_or_existing_artifacts(
    tmp_path: Path,
) -> None:
    from nimloth.training.rl.verl_checkpoint import (
        prepare_transformers449_checkpoint_view,
    )

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "missing.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        prepare_transformers449_checkpoint_view(source, tmp_path / "missing-view")

    (source / "missing.safetensors").write_bytes(b"weights")
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "sentinel").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must not exist"):
        prepare_transformers449_checkpoint_view(source, existing)
