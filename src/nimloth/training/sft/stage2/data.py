"""Align every answer in a full trajectory with its observation and DINO grid."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nimloth.latent import (
    latent_state_block,
    latent_state_tokens,
    normalize_latent_state_blocks,
)
from nimloth.training.sft.stage1.data import (
    NimlothVLSFTDataset,
    assistant_token_spans,
    collate_fn,
)


def validate_query_alignment_jsonl(
    path: Path,
    *,
    split: str,
    query_count: int,
    max_records: int = -1,
    max_images_per_record: int = -1,
) -> int:
    """Validate the Stage2 answer-view contract without loading a model or CUDA."""
    try:
        dataset = NimlothVLSFTDataset(
            path,
            processor=None,
            max_records=max_records,
            max_images_per_record=max_images_per_record,
        )
    except json.JSONDecodeError as exc:
        source_line = None
        record_index = 0
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                if line.strip() == exc.doc.strip():
                    source_line = line_number
                    break
                record_index += 1
        location = (
            f"record {record_index}, source line {source_line}"
            if source_line is not None
            else "an unknown source line"
        )
        raise ValueError(
            f"Stage2 query alignment {split} dataset contains invalid JSONL at "
            f"{path} ({location}): {exc.msg} (column {exc.colno})"
        ) from exc
    if not dataset.records:
        raise ValueError(f"Stage2 query alignment {split} dataset is empty: {path}")

    for index, record in enumerate(dataset.records):
        location = f"{split} record {index} in {path}"
        if not isinstance(record, dict):
            raise TypeError(
                f"Stage2 query alignment {location} must be a JSON object"
            )
        if "messages" not in record:
            record_format = record.get("record_format", "unknown")
            raise ValueError(
                f"Stage2 query alignment {location} must be answer-view JSONL with "
                f"top-level 'messages'; received record_format={record_format!r}"
            )
        if not isinstance(record.get("id"), str) or not record["id"]:
            raise ValueError(
                f"Stage2 query alignment {location} requires a nonempty string 'id'"
            )
        if type(record.get("success")) is not bool:
            raise ValueError(
                f"Stage2 query alignment {location} requires boolean 'success'"
            )
        messages = record["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"Stage2 query alignment {location} requires a nonempty 'messages' list"
            )
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or not isinstance(
                message.get("role"), str
            ):
                raise TypeError(
                    f"Stage2 query alignment {location} message {message_index} "
                    "requires a string 'role'"
                )
            if not isinstance(message.get("content"), (str, list)):
                raise TypeError(
                    f"Stage2 query alignment {location} message {message_index} "
                    "requires string or multimodal-list 'content'"
                )
            if isinstance(message["content"], list):
                for part_index, part in enumerate(message["content"]):
                    if not isinstance(part, dict):
                        raise TypeError(
                            f"Stage2 query alignment {location} message "
                            f"{message_index} content part {part_index} must be an object"
                        )
                    if part.get("type") == "image" and (
                        not isinstance(part.get("image"), str) or not part["image"]
                    ):
                        raise ValueError(
                            f"Stage2 query alignment {location} message "
                            f"{message_index} content part {part_index} requires a "
                            "nonempty string image path"
                        )
        if "image_paths" in record:
            image_paths = record["image_paths"]
            if not isinstance(image_paths, list) or any(
                not isinstance(image_path, str) or not image_path
                for image_path in image_paths
            ):
                raise ValueError(
                    f"Stage2 query alignment {location} requires 'image_paths' to be "
                    "a list of nonempty strings"
                )
        try:
            messages = dataset.get_messages(index)
            answer_observation_paths([{"messages": messages}])
            _answer_messages_with_query_boundary(messages, query_count)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Stage2 query alignment {location} violates the answer-view contract: "
                f"{exc}"
            ) from exc
    return len(dataset.records)


def _answer_messages_with_query_boundary(
    messages: list[dict[str, Any]], query_count: int
) -> list[dict[str, Any]]:
    answer_messages = [message for message in messages if message["role"] == "assistant"]
    boundary = (
        r"</think>\s*"
        + re.escape(latent_state_block(query_count))
        + r"\s*<\|action_start\|>"
    )
    if any(
        not isinstance(message["content"], str)
        or re.search(
            boundary,
            normalize_latent_state_blocks(message["content"], query_count),
        )
        is None
        for message in answer_messages
    ):
        raise ValueError("query state must follow the recorded CoT and precede the action")
    return answer_messages


def answer_observation_paths(batch: list[dict[str, Any]]) -> list[str]:
    """Return one unambiguous current observation for every recorded answer."""
    paths: list[str] = []
    for record in batch:
        observation = None
        answers = 0
        for message in record["messages"]:
            if message["role"] == "user":
                content = message["content"]
                images = (
                    [part["image"] for part in content if part.get("type") == "image"]
                    if isinstance(content, list)
                    else []
                )
                observation = images[0] if len(images) == 1 else None
            if message["role"] != "assistant":
                continue
            if observation is None:
                raise ValueError(
                    "each query-alignment answer needs exactly one image in its user turn"
                )
            content = message["content"]
            if (
                not isinstance(content, str)
                or re.search(r"<think>\s*\S.*?</think>", content, re.DOTALL) is None
            ):
                raise ValueError(
                    "query alignment requires the recorded nonempty CoT for this observation"
                )
            paths.append(str(observation))
            observation = None
            answers += 1
        if answers == 0:
            raise ValueError("query-alignment trajectory has no recorded answers")
    return paths


@dataclass
class QueryAlignmentCollator:
    processor: Any
    max_length: int
    query_count: int
    targets: Any
    mask_latent_query_labels: bool = True

    def __call__(self, batch):
        if any(type(record.get("success")) is not bool for record in batch):
            raise ValueError("query alignment requires explicit trajectory success boolean")
        paths = answer_observation_paths(batch)
        encoded = collate_fn(
            batch,
            self.processor,
            self.max_length,
            latent_token_count=self.query_count,
            mask_latent_query_labels=self.mask_latent_query_labels,
            require_complete=True,
        )
        query_ids = [
            self.processor.tokenizer.convert_tokens_to_ids(token)
            for token in latent_state_tokens(self.query_count)
        ]
        if len(set(query_ids)) != self.query_count or any(
            token_id is None
            or token_id == self.processor.tokenizer.unk_token_id
            for token_id in query_ids
        ):
            raise ValueError("query tokens must have distinct registered token IDs")

        answer_indices = torch.full_like(encoded["labels"], -1)
        query_batch_indices: list[int] = []
        query_positions: list[list[int]] = []
        answer_index = 0
        for row, record in enumerate(batch):
            messages = record["messages"]
            answer_messages = _answer_messages_with_query_boundary(
                messages, self.query_count
            )
            spans = assistant_token_spans(
                messages,
                self.processor,
                self.max_length,
                latent_token_count=self.query_count,
            )
            if len(spans) != len(answer_messages):
                raise ValueError("truncation removed a recorded answer from the trajectory")
            ids = encoded["input_ids"][row].tolist()
            for start, end in spans:
                indices = [
                    position
                    for position in range(start, min(end, len(ids)))
                    if ids[position] in query_ids
                ]
                if (
                    len(indices) != self.query_count
                    or [ids[position] for position in indices] != query_ids
                    or indices
                    != list(range(indices[0], indices[0] + self.query_count))
                ):
                    raise ValueError(
                        "answer query slots must be complete, contiguous and in row-major order"
                    )
                supervised = encoded["labels"][row, start:end] != -100
                if not torch.any(supervised):
                    raise ValueError("recorded answer has no supervised target tokens")
                answer_indices[row, start:end][supervised] = answer_index
                query_batch_indices.append(row)
                query_positions.append(indices)
                answer_index += 1
        if answer_index != len(paths):
            raise ValueError("answer/query/observation counts do not align")
        encoded["lm_answer_mask"] = torch.tensor(
            [batch[row]["success"] for row in query_batch_indices], dtype=torch.bool
        )
        encoded["answer_indices"] = answer_indices
        encoded["query_batch_indices"] = torch.tensor(
            query_batch_indices, dtype=torch.long
        )
        encoded["query_positions"] = torch.tensor(query_positions, dtype=torch.long)
        with torch.no_grad():
            encoded["dino_target"] = self.targets.load(
                paths, device=torch.device("cpu")
            ).detach()
        return encoded
