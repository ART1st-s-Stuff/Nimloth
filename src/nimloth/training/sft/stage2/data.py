"""Align every answer in a full trajectory with its observation and DINO grid."""

import re
from dataclasses import dataclass
from typing import Any

import torch

from nimloth.latent import (
    latent_state_block,
    latent_state_tokens,
    normalize_latent_state_blocks,
)
from nimloth.training.sft.stage1.data import assistant_token_spans, collate_fn


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
        boundary = (
            r"</think>\s*"
            + re.escape(latent_state_block(self.query_count))
            + r"\s*<\|action_start\|>"
        )
        for row, record in enumerate(batch):
            messages = record["messages"]
            answer_messages = [m for m in messages if m["role"] == "assistant"]
            if any(
                not isinstance(message["content"], str)
                or re.search(
                    boundary,
                    normalize_latent_state_blocks(
                        message["content"], self.query_count
                    ),
                )
                is None
                for message in answer_messages
            ):
                raise ValueError(
                    "query state must follow the recorded CoT and precede the action"
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
