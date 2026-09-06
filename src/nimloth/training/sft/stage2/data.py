"""Pair each recorded answer with its own observation and row-major DINO grid."""

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


def answer_examples(batch: list[dict[str, Any]]) -> tuple[list[dict], list[str]]:
    """Expand dialogues into prefixes; never borrow a later observation or thought."""
    examples, paths = [], []
    for record in batch:
        messages = record["messages"]
        observation = None
        for index, message in enumerate(messages):
            if message["role"] == "user":
                content = message["content"]
                images = (
                    [part["image"] for part in content if part.get("type") == "image"]
                    if isinstance(content, list)
                    else []
                )
                # Do not silently choose a view when the record is ambiguous.
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
            examples.append({"messages": messages[: index + 1]})
            paths.append(str(observation))
            observation = None
    if not examples:
        raise ValueError("query alignment batch has no recorded answers")
    return examples, paths


@dataclass
class QueryAlignmentCollator:
    processor: Any
    max_length: int
    query_count: int
    targets: Any
    mask_latent_query_labels: bool = True

    def __call__(self, batch):
        examples, paths = answer_examples(batch)
        for example in examples:
            answer = normalize_latent_state_blocks(
                example["messages"][-1]["content"], self.query_count
            )
            boundary = (
                r"</think>\s*"
                + re.escape(latent_state_block(self.query_count))
                + r"\s*<\|action_start\|>"
            )
            if re.search(boundary, answer) is None:
                raise ValueError(
                    "query state must follow the recorded CoT and precede the action"
                )
        encoded = collate_fn(
            examples,
            self.processor,
            self.max_length,
            latent_token_count=self.query_count,
            mask_latent_query_labels=self.mask_latent_query_labels,
        )
        query_ids = [
            self.processor.tokenizer.convert_tokens_to_ids(t)
            for t in latent_state_tokens(self.query_count)
        ]
        if len(set(query_ids)) != self.query_count or any(
            t is None or t == self.processor.tokenizer.unk_token_id for t in query_ids
        ):
            raise ValueError("query tokens must have distinct registered token IDs")
        positions = []
        for row, example in enumerate(examples):
            spans = assistant_token_spans(
                example["messages"],
                self.processor,
                self.max_length,
                latent_token_count=self.query_count,
            )
            if len(spans) != sum(m["role"] == "assistant" for m in example["messages"]):
                raise ValueError(
                    "truncation removed a recorded answer from the query prefix"
                )
            start, end = spans[-1]
            # History provides context but is not supervised again for every prefix.
            encoded["labels"][row, :start] = -100
            encoded["labels"][row, end:] = -100
            ids = encoded["input_ids"][row].tolist()
            indices = [
                i for i in range(start, min(end, len(ids))) if ids[i] in query_ids
            ]
            if [ids[i] for i in indices] != query_ids or indices != list(
                range(indices[0], indices[0] + self.query_count)
            ):
                raise ValueError(
                    "answer query slots must be complete, contiguous and in row-major order"
                )
            if not torch.any(encoded["labels"][row, 1:] != -100):
                raise ValueError("truncation removed the target answer")
            positions.append(indices)
        encoded["query_positions"] = torch.tensor(positions, dtype=torch.long)
        with torch.no_grad():
            encoded["dino_target"] = self.targets.load(
                paths, device=torch.device("cpu")
            ).detach()
        return encoded
