from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from nimloth.latent import LatentActionTokens, latent_state_block, latent_state_tokens
from nimloth.rollout import RolloutTrajectory
from nimloth.training.sft1.data import sha256_file
from nimloth.training.sft1.real_rows import (
    EARLY4_ROW_SCHEMA,
    SFT1V2Early4Row,
    audit_rendered_token_upper_bound,
    render_early4_row,
)
from tests.training.sft1._state_v2_fixtures import trajectory_record


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


def test_actual_k8_structural_row_renders_k16_without_changing_cot_or_action_boundary(
    tmp_path: Path,
) -> None:
    record, image = trajectory_record(tmp_path, latent_token_count=8)
    trajectory = RolloutTrajectory.from_record(record)
    for path in trajectory.image_paths:
        Image.new("RGB", (2, 2), color=(1, 2, 3)).save(path)
    trajectory.observation_texts[0] = (
        "Human Instruction: Find the target object.\nDecide your next action(s).\n<image>"
    )
    record = trajectory.to_record()
    response = trajectory.assistant_responses[0]
    assert latent_state_block(8) + LatentActionTokens().action_start in response
    instruction_group = hashlib.sha256(trajectory.instruction.encode()).hexdigest()
    row = SFT1V2Early4Row(
        schema=EARLY4_ROW_SCHEMA, ordinal=0, source_path="train.jsonl",
        source_sha256="a" * 64, split="train", record_id=trajectory.record_id,
        step_index=0, original_image_path=str(image),
        original_image_sha256=sha256_file(image), image_content_group=sha256_file(image),
        instruction=trajectory.instruction,
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
