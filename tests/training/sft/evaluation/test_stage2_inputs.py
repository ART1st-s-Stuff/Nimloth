from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.navigation_baseline import convert_sft1_rollouts_to_nimloth
from experiments.training.sft.evaluation import stage2_inputs


def _source_row(answer: str) -> dict:
    return {
        "input": "<|im_start|>system\ns<|im_end|><|im_start|>user\n<image> u<|im_end|>",
        "output": answer,
    }


def _converted(record_id: str, source: Path, line_index: int, cot: str) -> dict:
    return {
        "id": record_id,
        "source_jsonl": str(source),
        "source_line_index": line_index,
        "messages": [
            {"role": "user", "content": "<image> u"},
            {
                "role": "assistant",
                "content": f"<think>{cot}</think><|latent_state|><|action_start|><|action_0|><|action_end|>",
            },
        ],
        "image_paths": [f"/observations/{record_id}.png"],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_future_converter_distinguishes_missing_and_empty_cot():
    check = convert_sft1_rollouts_to_nimloth.cot_validation_issue
    assert check("<action>move_forward</action>") == "missing_think_tag"
    assert check("<think> \n </think><action>x</action>") == "empty_think_body"
    assert check("<think>observed</think><action>x</action>") is None


def test_materialize_excludes_whole_trajectory_and_preserves_reason(tmp_path):
    original = tmp_path / "raw.jsonl"
    _write_jsonl(
        original,
        [
            _source_row("<action>move_forward</action>"),
            _source_row("<think>   </think><action>move_forward</action>"),
            _source_row("<think>real reasoning</think><action>move_forward</action>"),
        ],
    )
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_jsonl(
        train,
        [
            _converted("missing", original, 0, ""),
            _converted("valid", original, 2, "real reasoning"),
        ],
    )
    _write_jsonl(
        val,
        [
            _converted("empty", original, 1, ""),
            _converted("valid-val", original, 2, "real reasoning"),
        ],
    )
    output = tmp_path / "stage2-v1"

    args = SimpleNamespace(
        train_source=str(train), val_source=str(val), output_root=str(output)
    )
    assert stage2_inputs.materialize(args) == 0

    manifest = json.loads((output / "manifest.json").read_text())
    exclusions = [
        json.loads(line)
        for line in (output / "exclusions.jsonl").read_text().splitlines()
    ]
    retained = [
        json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in retained] == ["valid"]
    assert manifest["splits"]["train"]["before"]["trajectories"] == 2
    assert manifest["splits"]["train"]["after"]["trajectories"] == 1
    assert manifest["splits"]["val"]["after"]["trajectories"] == 1
    assert [row["record_id"] for row in exclusions] == ["missing", "empty"]
    assert exclusions[0]["invalid_turns"][0]["reason"] == "missing_think_tag"
    assert exclusions[1]["invalid_turns"][0]["reason"] == "empty_think_body"

    with pytest.raises(FileExistsError, match="refusing to replace"):
        stage2_inputs.materialize(args)


def test_validate_rejects_changed_source_before_dino_loading(tmp_path):
    original = tmp_path / "raw.jsonl"
    _write_jsonl(
        original, [_source_row("<think>x</think><action>move_forward</action>")]
    )
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    row = _converted("valid", original, 0, "x")
    _write_jsonl(train, [row])
    _write_jsonl(val, [dict(row, id="valid-val")])
    output = tmp_path / "stage2-v1"
    stage2_inputs.materialize(
        SimpleNamespace(
            train_source=str(train), val_source=str(val), output_root=str(output)
        )
    )
    train.write_text(train.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        stage2_inputs.validate(
            SimpleNamespace(
                input_root=str(output),
                dino_cache_root="unused",
                expected_train_source=str(train),
                expected_val_source=str(val),
            )
        )


def test_materialize_rejects_empty_retained_split(tmp_path):
    original = tmp_path / "raw.jsonl"
    _write_jsonl(original, [_source_row("<think></think><action>x</action>")])
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_jsonl(train, [_converted("bad", original, 0, "")])
    _write_jsonl(val, [_converted("bad-val", original, 0, "")])

    with pytest.raises(ValueError, match="train Stage2 split has no eligible"):
        stage2_inputs.materialize(
            SimpleNamespace(
                train_source=str(train),
                val_source=str(val),
                output_root=str(tmp_path / "stage2-v1"),
            )
        )


def test_audit_rejects_converted_cot_that_differs_from_original(tmp_path):
    original = tmp_path / "raw.jsonl"
    _write_jsonl(original, [_source_row("<think>real</think><action>x</action>")])
    converted = _converted("changed", original, 0, "invented")

    exclusion = stage2_inputs.audit_record(
        converted, split="train", line_number=1, source_path=tmp_path / "train.jsonl"
    )[0]
    assert (
        exclusion["invalid_turns"][0]["reason"] == "converted_cot_differs_from_source"
    )


def test_validate_recomputes_whole_trajectory_output(tmp_path):
    original = tmp_path / "raw.jsonl"
    _write_jsonl(original, [_source_row("<think>x</think><action>x</action>")])
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train_row = _converted("train", original, 0, "x")
    val_row = _converted("val", original, 0, "x")
    _write_jsonl(train, [train_row])
    _write_jsonl(val, [val_row])
    output = tmp_path / "stage2-v1"
    stage2_inputs.materialize(
        SimpleNamespace(
            train_source=str(train), val_source=str(val), output_root=str(output)
        )
    )

    changed = dict(train_row, id="substituted")
    _write_jsonl(output / "train.jsonl", [changed])
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["train"]["output_sha256"] = stage2_inputs.sha256(
        output / "train.jsonl"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="whole-trajectory audit"):
        stage2_inputs.validate(
            SimpleNamespace(
                input_root=str(output),
                dino_cache_root="unused",
                expected_train_source=str(train),
                expected_val_source=str(val),
            )
        )
