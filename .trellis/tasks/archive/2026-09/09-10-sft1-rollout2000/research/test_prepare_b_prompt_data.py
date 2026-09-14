import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_b", Path(__file__).with_name("prepare_b_prompt_data.py")
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def prompt():
    return "\n".join(
        [
            "moveahead: Move forward by some distance",
            "Format correct only when the answer is exactly one valid action name.",
            "Invalid action names receive negative feedback.",
            "The answer must be exactly one of these lowercase action names: "
            + ", ".join(m.NAMES)
            + ".",
            "The action must be exactly one of these lowercase action names: "
            + ", ".join(m.NAMES)
            + ".",
            m.OLD_LEGEND,
            "<|latent_state|><|action_start|><|action_(0)|><|action_end|>",
        ]
    )


def record():
    return {
        "id": "one",
        "system_prompt": prompt(),
        "messages": [
            {"role": "system", "content": prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt()},
                    [{"type": "input_text", "text": prompt()}],
                    {
                        "type": "image",
                        "image": "Invalid action names receive negative feedback.",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "<think>"
                + prompt()
                + "</think><|action_start|><|action_(0)|><|action_end|>",
            },
        ],
        "image_paths": ["img.png"],
        "actions": ["moveahead"],
        "source_audit": {"system_prompt": prompt()},
        "conversion_provenance": {"unchanged": True},
    }


def test_real_clauses_and_meanings():
    revised = m.rewrite_text(prompt())
    assert "lowercase action names" not in revised
    assert "valid action name" not in revised
    assert "moveahead: Move forward by some distance" in revised
    assert "<|latent_state|>" in revised
    for token, name, meaning in zip(m.TOKENS, m.NAMES, m.MEANINGS):
        assert f"{token} = {name} ({meaning})" in revised
    assert m.rewrite_text(revised) == revised


def test_only_allowed_fields_change():
    old = record()
    untouched = copy.deepcopy(old)
    new = m.transform(old)
    assert old == untouched
    assert new["messages"][2] == old["messages"][2]
    assert new["source_audit"] == old["source_audit"]
    assert new["image_paths"] == old["image_paths"]
    assert new["messages"][1]["content"][2] == old["messages"][1]["content"][2]
    assert new["messages"][1]["content"][1][0]["text"] == m.rewrite_text(prompt())
    m.validate_preservation(old, new)
    new["messages"][2]["content"] += "bad"
    with pytest.raises(ValueError, match="Non-prompt"):
        m.validate_preservation(old, new)


def test_unknown_clause_fails_closed():
    with pytest.raises(ValueError, match="Unconverted"):
        m.rewrite_text("The action must use lowercase action names: surprise.")


def test_directory_roundtrip_and_source_immutable(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    name = "sft1_train_all.jsonl"
    path = source / name
    path.write_text(json.dumps(record()) + "\n")
    before = path.read_bytes()
    out = tmp_path / "derived"
    result = m.prepare(source, out, {name: 1})
    assert result["counts"][name] == 1
    assert path.read_bytes() == before
    assert (out / "COMPLETED.json").exists()
    assert (out / "prompt_examples.diff").read_text()
    assert json.loads((out / name).read_text()) == m.transform(record())
    with pytest.raises(FileExistsError):
        m.prepare(source, out, {name: 1})


def test_invalid_count_creates_nothing(tmp_path):
    name = "sft1_train_all.jsonl"
    (tmp_path / name).write_text(json.dumps(record()) + "\n")
    with pytest.raises(ValueError, match="expected"):
        m.prepare(tmp_path, tmp_path / "derived", {name: 2})
    assert not (tmp_path / "derived").exists()


def test_overlap_rejected(tmp_path):
    for name in ("train", "val"):
        (tmp_path / name).write_text(json.dumps(record()) + "\n")
    with pytest.raises(ValueError, match="overlapping"):
        m.prepare(tmp_path, tmp_path / "derived", {"train": 1, "val": 1})
