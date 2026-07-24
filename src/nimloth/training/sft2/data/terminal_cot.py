"""Generate and persist CoT-conditioned terminal state prefixes for SFT2."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from nimloth.agent import bind_image_placeholders
from nimloth.backbone.qwen25vl.policy import (
    collect_policy_images,
    render_policy_messages,
)
from nimloth.latent import LatentActionTokens, latent_state_tokens
from nimloth.rollout.transitions import TERMINAL_ASSISTANT_PREFIX_FIELD


@dataclass(frozen=True)
class TerminalCoTGeneration:
    prefix: str
    reasoning_token_count: int


class _MaskNimlothProtocolTokens(LogitsProcessor):
    def __init__(self, token_ids: Sequence[int]) -> None:
        self._token_ids = tuple(int(token_id) for token_id in token_ids)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        scores[:, list(self._token_ids)] = float("-inf")
        return scores


class _StopAfterText(StoppingCriteria):
    def __init__(self, tokenizer: Any, *, start_length: int, text: str) -> None:
        self._tokenizer = tokenizer
        self._start_length = int(start_length)
        self._text = text

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        continuation = self._tokenizer.decode(
            input_ids[0, self._start_length :].tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=False,
        )
        return self._text in continuation


def terminal_cot_prompt_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the final-observation prompt ending at the start of real CoT."""

    system_prompt = str(record.get("system_prompt", ""))
    observation_texts = tuple(
        str(text) for text in record.get("observation_texts", [])
    )
    if system_prompt and observation_texts:
        action_indices = tuple(int(index) for index in record.get("action_indices", []))
        assistant_responses = tuple(
            str(response) for response in record.get("assistant_responses", [])
        )
        if len(observation_texts) != len(action_indices) + 1:
            raise ValueError("structured record requires one final observation")
        if len(assistant_responses) != len(action_indices):
            raise ValueError("structured record assistant response/action mismatch")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        for step_index, response in enumerate(assistant_responses):
            messages.extend(
                (
                    {"role": "user", "content": observation_texts[step_index]},
                    {"role": "assistant", "content": response},
                )
            )
        messages.append({"role": "user", "content": observation_texts[-1]})
    else:
        messages = [dict(message) for message in record.get("messages", [])]
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("legacy record must end with the final user observation")
    messages.append({"role": "assistant", "content": "<think>"})
    return messages


@torch.no_grad()
def generate_terminal_cot_prefix(
    *,
    record: dict[str, Any],
    model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
    latent_token_count: int,
    max_reasoning_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    n: int,
) -> TerminalCoTGeneration:
    """Generate terminal CoT and append only the injected state query prefix."""

    messages = terminal_cot_prompt_messages(record)
    image_paths = tuple(str(path) for path in record.get("image_paths", []))
    bound_messages = bind_image_placeholders(messages, image_paths)
    text = render_policy_messages(
        bound_messages,
        processor,
        latent_token_count=latent_token_count,
    )
    images = collect_policy_images(bound_messages)
    inputs = processor(
        text=[text],
        images=[images] if images else None,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {key: value.to(device) for key, value in inputs.items()}
    close_ids = tuple(
        int(token_id)
        for token_id in processor.tokenizer.encode(
            "</think>",
            add_special_tokens=False,
        )
    )
    if not close_ids:
        raise ValueError("tokenizer produced no IDs for '</think>'")
    protocol_ids = tuple(int(token_id) for token_id in token_id_map.values())
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_reasoning_tokens + len(close_ids),
        "do_sample": do_sample,
        "num_return_sequences": n,
        "logits_processor": LogitsProcessorList(
            [_MaskNimlothProtocolTokens(protocol_ids)]
        ),
        "stopping_criteria": StoppingCriteriaList(
            [
                _StopAfterText(
                    processor.tokenizer,
                    start_length=model_inputs["input_ids"].shape[1],
                    text="</think>",
                )
            ]
        ),
        "pad_token_id": pad_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    output_ids = model.generate(**model_inputs, **generation_kwargs)
    continuation_ids = tuple(
        int(token_id)
        for token_id in output_ids[0, model_inputs["input_ids"].shape[1] :].tolist()
    )
    decoded_continuation = processor.tokenizer.decode(
        list(continuation_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
        spaces_between_special_tokens=False,
    )
    close_start = decoded_continuation.find("</think>")
    thought = decoded_continuation[:close_start] if close_start >= 0 else ""
    reasoning_token_count = len(
        processor.tokenizer.encode(thought, add_special_tokens=False)
    )
    if close_start < 0 or reasoning_token_count > max_reasoning_tokens:
        record_id = str(record.get("id", ""))
        raise RuntimeError(
            f"record {record_id!r}: terminal CoT did not emit '</think>' within "
            f"{max_reasoning_tokens} reasoning tokens"
        )
    tokens = LatentActionTokens()
    latent_block = "".join(latent_state_tokens(latent_token_count, tokens))
    return TerminalCoTGeneration(
        prefix=(
            f"<think>{thought}</think>{latent_block}{tokens.action_start}"
        ),
        reasoning_token_count=reasoning_token_count,
    )


def write_augmented_records(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> int:
    """Atomically write records after terminal CoT generation succeeds."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite terminal CoT data: {path}")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    count = 0
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return count


__all__ = [
    "TerminalCoTGeneration",
    "generate_terminal_cot_prefix",
    "terminal_cot_prompt_messages",
    "write_augmented_records",
]
