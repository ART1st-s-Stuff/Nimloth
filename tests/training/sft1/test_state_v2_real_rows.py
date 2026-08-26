from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from nimloth.latent import LatentActionTokens, latent_state_block, latent_state_tokens
from nimloth.training.sft1.data import sha256_file
from nimloth.training.sft1.experiment_config import load_sft1_v2_config
from nimloth.training.sft1.real_rows import (
    EARLY4_ROW_SCHEMA,
    SFT1V2Early4Row,
    audit_rendered_token_upper_bound,
    index_early4_rows,
    render_early4_row,
)
from tests.training.sft1._state_v2_fixtures import pre_rl_trajectory_record


ROOT = Path(__file__).resolve().parents[3]


class _Tokenizer:
    unk_token_id = -1

    def __init__(self) -> None:
        tokens = LatentActionTokens()
        all_tokens = (
            *latent_state_tokens(16), tokens.action_start, tokens.action_end,
            *tokens.action_tokens,
        )
        self.ids = {token: 10 + index for index, token in enumerate(all_tokens)}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.ids.get(token, self.unk_token_id)

    def encode(self, token: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        value = self.convert_tokens_to_ids(token)
        return [] if value == self.unk_token_id else [value]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[Any]]:
        assert not add_special_tokens
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        specials = sorted(self.ids, key=len, reverse=True)
        index = 0
        while index < len(text):
            special = next((token for token in specials if text.startswith(token, index)), None)
            if special is None:
                ids.append(1000 + ord(text[index]))
                offsets.append((index, index + 1))
                index += 1
            else:
                ids.append(self.ids[special])
                offsets.append((index, index + len(special)))
                index += len(special)
        result: dict[str, list[Any]] = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.image_processor = SimpleNamespace(patch_size=14, merge_size=2)

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert not tokenize and not add_generation_prompt
        parts: list[str] = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                payload = content
            else:
                payload_parts = []
                for item in content:
                    if item["type"] == "text":
                        payload_parts.append(item["text"])
                    elif item["type"] == "image":
                        payload_parts.append("<image>")
                payload = "".join(payload_parts)
            parts.append(f"<{message['role']}>{payload}</{message['role']}>")
        return "".join(parts)

    def __call__(self, *, text, images, padding, truncation, return_tensors):
        assert padding is False and truncation is False and return_tensors == "pt"
        encoded = self.tokenizer(
            text[0], add_special_tokens=False, return_offsets_mapping=True
        )
        return {
            "input_ids": torch.tensor([encoded["input_ids"]], dtype=torch.long),
            "attention_mask": torch.ones(1, len(encoded["input_ids"]), dtype=torch.long),
        }


def _write_source(path: Path, record: dict[str, object]) -> str:
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def _small_config(
    tmp_path: Path,
    train_record: dict[str, object],
    validation_record: dict[str, object],
):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_sha = _write_source(train_path, train_record)
    validation_sha = _write_source(validation_path, validation_record)
    config = load_sft1_v2_config(
        ROOT / "configs/training/sft1/state_interface_v2_early4_report_first.yaml"
    )
    return replace(
        config,
        data=replace(
            config.data,
            train_jsonl=str(train_path),
            train_sha256=train_sha,
            validation_jsonl=str(validation_path),
            validation_sha256=validation_sha,
        ),
    )


def test_exact_pre_rl_schema_extracts_instruction_from_real_observation_span(
    tmp_path: Path,
) -> None:
    train, _ = pre_rl_trajectory_record(tmp_path, record_id="train-row")
    validation, _ = pre_rl_trajectory_record(
        tmp_path, record_id="validation-row", split="val"
    )
    rows, audit = index_early4_rows(
        _small_config(tmp_path, train, validation),
        enforce_approved_counts=False,
    )

    assert (audit.train_records, audit.validation_records) == (1, 1)
    assert len(rows) == 2
    row = rows[0]
    assert row.instruction == "Find the target object."
    start, stop = row.instruction_char_span
    assert train["observation_texts"][0][start:stop] == row.instruction
    assert "instruction" not in row.record
    assert "action_log_probs" not in row.record


def test_pre_rl_index_excludes_only_source_empty_cot_rows(tmp_path: Path) -> None:
    train, _ = pre_rl_trajectory_record(tmp_path, record_id="train-row")
    validation, _ = pre_rl_trajectory_record(
        tmp_path, record_id="validation-row", split="val"
    )
    validation["image_paths"][0] = train["image_paths"][0]
    train["think_texts"][0] = ""
    response = train["assistant_responses"][0]
    train["assistant_responses"][0] = response.replace(
        "<think>The observation supports this executed action.</think>",
        "<think></think>",
        1,
    )

    rows, audit = index_early4_rows(
        _small_config(tmp_path, train, validation),
        enforce_approved_counts=False,
    )

    assert [row.split for row in rows] == ["val"]
    assert rows[0].external_eligible is False
    assert audit.external_validation_rows == 0
    assert audit.excluded_train_empty_cot_rows == 1
    assert audit.excluded_validation_empty_cot_rows == 0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda record: record.update(extra_field=True), "exact 28-field"),
        (
            lambda record: record["observation_texts"].__setitem__(
                0,
                record["observation_texts"][0]
                + "\nHuman Instruction: duplicate\nDecide your next action(s).",
            ),
            "exactly one Human Instruction",
        ),
        (lambda record: record["actions"].__setitem__(0, "move_left"), "action name"),
    ],
)
def test_pre_rl_schema_rejects_aliases_ambiguous_instruction_and_action_drift(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    train, _ = pre_rl_trajectory_record(tmp_path, record_id="train-row")
    validation, _ = pre_rl_trajectory_record(
        tmp_path, record_id="validation-row", split="val"
    )
    mutation(train)
    with pytest.raises(ValueError, match=match):
        index_early4_rows(
            _small_config(tmp_path, train, validation),
            enforce_approved_counts=False,
        )


def test_actual_k8_structural_row_renders_k16_without_changing_cot_or_action_boundary(
    tmp_path: Path,
) -> None:
    record, image = pre_rl_trajectory_record(tmp_path, latent_token_count=8)
    for path in record["image_paths"]:
        Image.new("RGB", (2, 2), color=(1, 2, 3)).save(path)
    response = record["assistant_responses"][0]
    assert latent_state_block(8) + LatentActionTokens().action_start in response
    instruction = "Find the target object."
    observation = record["observation_texts"][0]
    char_start = observation.index("Human Instruction: ") + len("Human Instruction: ")
    char_stop = observation.index("\nDecide your next action(s).", char_start)
    instruction_group = hashlib.sha256(instruction.encode()).hexdigest()
    row = SFT1V2Early4Row(
        schema=EARLY4_ROW_SCHEMA, ordinal=0, source_path="train.jsonl",
        source_sha256="a" * 64, split="train", record_id=str(record["id"]),
        step_index=0, original_image_path=str(image),
        original_image_sha256=sha256_file(image), image_content_group=sha256_file(image),
        instruction=instruction, instruction_char_span=(char_start, char_stop),
        instruction_equivalence_group=instruction_group,
        archived_assistant_response=response, executed_action_index=0,
        movement_success=True, external_eligible=True, record=record,
    )

    rendered = render_early4_row(row, processor=_Processor(), max_length=8192)

    assert latent_state_block(16) + LatentActionTokens().action_start in rendered.rendered_text
    assert latent_state_block(8) + LatentActionTokens().action_start not in rendered.rendered_text
    assert "The observation supports this executed action." in rendered.rendered_text
    expected = response[: response.index(LatentActionTokens().action_start)]
    expected = expected.replace(latent_state_block(8), latent_state_block(16))
    assert expected in rendered.rendered_text
    assert rendered.instruction_token_span[1] <= rendered.action_boundary_index
    assert "labels" not in rendered.encoded_tensors
    assert audit_rendered_token_upper_bound(
        (row,),
        processor=_Processor(),
        max_sequence_length=8192,
        max_pixels=100352,
    ) < 8192
