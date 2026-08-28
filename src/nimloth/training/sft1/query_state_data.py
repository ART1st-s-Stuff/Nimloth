"""Real archived-response rendering and detached targets for Query-State SFT1.

This module is deliberately separate from the historical state-interface-v2
prepared-row contract.  It reuses the audited archive parser, but renders the
complete current assistant response and carries no student hidden/state cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from nimloth.agent.template import bind_image_placeholders
from nimloth.backbone.dino_grid import (
    DINOV2_LARGE_IDENTITY,
    FrozenDINOGridTargets,
)
from nimloth.backbone.qwen25vl.batch import collect_message_images, render_messages
from nimloth.backbone.qwen25vl.state_training import (
    find_current_state_extraction_positions,
    require_archived_assistant_response,
)
from nimloth.latent import (
    LatentActionTokens,
    find_all_latent_state_blocks,
    latent_state_block,
    special_token_ids,
)
from nimloth.training.sft1.data import sha256_file
from nimloth.training.sft1.real_rows import (
    SFT1V2Early4Row,
    _context_structural_example_count,
    _parse_pre_rl_record,
)


QUERY_STATE_RENDERED_ROW_SCHEMA = "nimloth_sft1_query_state_rendered_row_v1"
QUERY_STATE_PREPARED_ROW_SCHEMA = "nimloth_sft1_query_state_prepared_row_v1"
_FORBIDDEN_STUDENT_KEYS = frozenset(
    {"hidden", "query_hidden", "student_hidden", "state", "projected_state"}
)
_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


@dataclass(frozen=True)
class QueryStateRenderedRow:
    schema: str
    row: SFT1V2Early4Row
    rendered_text: str
    input_ids: torch.Tensor
    action_boundary_index: int
    encoded_tensors: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class QueryStatePreparedRow:
    schema: str
    source_manifest_identity: str
    record_id: str
    step_index: int
    split: str
    original_image_path: str
    original_image_sha256: str
    archived_assistant_response: str
    executed_action_index: int
    encoded_tensors: Mapping[str, torch.Tensor]
    dino_regions: torch.Tensor

    @property
    def token_count(self) -> int:
        return int(self.encoded_tensors["input_ids"].numel())


def _full_current_messages(row: SFT1V2Early4Row) -> tuple[list[dict[str, Any]], int]:
    trajectory = _parse_pre_rl_record(row.record)
    if trajectory.record_id != row.record_id or not 0 <= row.step_index < trajectory.num_steps:
        raise ValueError("Query-State row record identity mismatch")
    response = require_archived_assistant_response(
        trajectory.assistant_responses[row.step_index], source="archived"
    )
    if response != row.archived_assistant_response:
        raise ValueError("Query-State archived response changed after indexing")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": trajectory.system_prompt}
    ]
    for prior in range(row.step_index):
        messages.append({"role": "user", "content": trajectory.observation_texts[prior]})
        messages.append({"role": "assistant", "content": trajectory.assistant_responses[prior]})
    messages.append(
        {"role": "user", "content": trajectory.observation_texts[row.step_index]}
    )
    # The complete real archived current response owns CoT, K8->K16 structure,
    # actual action token, and action_end.  No prefix/fixed response is built.
    messages.append({"role": "assistant", "content": response})
    bound = bind_image_placeholders(messages, trajectory.image_paths[: row.step_index + 1])
    expected_pairs = (
        _context_structural_example_count(trajectory, row.step_index)
        + row.step_index
        + 1
    )
    return bound, expected_pairs


def render_query_state_row(
    row: SFT1V2Early4Row,
    *,
    processor: Any,
    max_length: int,
) -> QueryStateRenderedRow:
    """Render one full current archived response with exact final-span labels."""

    if max_length < 1:
        raise ValueError("Query-State max_length must be positive")
    messages, expected_pairs = _full_current_messages(row)
    response = row.archived_assistant_response
    tokens = LatentActionTokens()
    k8 = latent_state_block(8)
    k16 = latent_state_block(16)
    expected_action = (
        tokens.action_start
        + tokens.action_tokens[row.executed_action_index]
        + tokens.action_end
    )
    if response.count(k8 + tokens.action_start) != 1:
        raise ValueError("Query-State historical response requires one K8 structural block")
    if response.count(expected_action) != 1:
        raise ValueError("Query-State response action disagrees with the executed action")

    rendered = render_messages(
        messages,
        processor,
        add_generation_prompt=False,
        latent_token_count=16,
    )
    normalized_response = response.replace(k8, k16)
    if normalized_response not in rendered:
        raise ValueError("Query-State rendering changed the archived current response")
    if k8 + tokens.action_start in rendered:
        raise ValueError("Query-State renderer did not normalize K8 to K16")

    images = collect_message_images(messages)
    encoded = processor(
        text=[rendered],
        images=[images] if images else None,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Query-State processor returned invalid input_ids")
    if input_ids.shape[1] > max_length:
        raise ValueError("Query-State row would truncate; truncation is forbidden")
    one_row = input_ids[0].detach().cpu()
    token_map = special_token_ids(processor.tokenizer, latent_token_count=16)
    positions = find_current_state_extraction_positions(
        one_row,
        token_map,
        expected_pair_count=expected_pairs,
        latent_token_count=16,
    )
    if positions.action_start_index is None or positions.latent_state_indices is None:
        raise ValueError("Query-State current K16/action boundary is absent")
    if positions.action_start_index + 2 >= one_row.numel():
        raise ValueError("Query-State current action/action_end response is incomplete")
    expected_action_ids = torch.tensor(
        [
            token_map[tokens.action_start],
            token_map[tokens.action_tokens[row.executed_action_index]],
            token_map[tokens.action_end],
        ],
        dtype=torch.long,
    )
    actual_action_ids = one_row[
        positions.action_start_index : positions.action_start_index + 3
    ]
    if not torch.equal(actual_action_ids, expected_action_ids):
        raise ValueError("Query-State encoded actual action boundary is invalid")

    # Determine the final assistant boundary from a second *processor encoding*,
    # not tokenizer-only offsets. Qwen's multimodal processor expands image-pad
    # tokens, so character/tokenizer offsets after an image are not indices into
    # the actual processed input_ids. The generation-prefix encoding contains the
    # exact same images and must be an exact token prefix of the complete row.
    prefix_rendered = render_messages(
        messages[:-1],
        processor,
        add_generation_prompt=True,
        latent_token_count=16,
    )
    if not rendered.startswith(prefix_rendered):
        raise ValueError("Query-State final assistant generation prefix is not exact")
    prefix_messages = messages[:-1]
    prefix_images = collect_message_images(prefix_messages)
    prefix_encoded = processor(
        text=[prefix_rendered],
        images=[prefix_images] if prefix_images else None,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )
    prefix_ids = prefix_encoded.get("input_ids")
    if (
        not isinstance(prefix_ids, torch.Tensor)
        or prefix_ids.ndim != 2
        or prefix_ids.shape[0] != 1
        or prefix_ids.shape[1] >= input_ids.shape[1]
        or not torch.equal(input_ids[0, : prefix_ids.shape[1]], prefix_ids[0])
    ):
        raise ValueError("Query-State processed final assistant BPE boundary is not exact")
    labels = torch.full_like(input_ids, -100)
    labels[:, prefix_ids.shape[1] :] = input_ids[:, prefix_ids.shape[1] :]
    # Mask exact validated structural positions, not every occurrence of a token
    # ID: ordinary text could legally collide with a special-token ID in tests or
    # a malformed tokenizer mapping must not broaden the mask silently.
    all_blocks = find_all_latent_state_blocks(
        one_row,
        token_map,
        tokens,
        latent_token_count=16,
    )
    for block in all_blocks:
        labels[0, torch.tensor(block, dtype=torch.long)] = -100
    if labels.shape != input_ids.shape or labels.dtype != torch.long:
        raise ValueError("Query-State final-assistant labels are invalid")
    valid = labels != -100
    if not torch.any(valid):
        raise ValueError("Query-State final assistant span has no valid LM labels")
    if torch.any(valid & labels.ne(input_ids)):
        raise ValueError("Query-State labels changed rendered BPE token IDs")
    if torch.any(labels[:, 0] != -100):
        raise ValueError("Query-State labels may not supervise sequence position zero")
    for block in all_blocks:
        if torch.any(labels[0, torch.tensor(block, dtype=torch.long)] != -100):
            raise RuntimeError("Query-State failed to mask a validated Query position")
    # CoT and the complete action envelope must be supervised in the final span.
    action_target_positions = torch.tensor(
        [
            positions.action_start_index,
            positions.action_start_index + 1,
            positions.action_start_index + 2,
        ],
        dtype=torch.long,
    )
    if torch.any(labels[0].index_select(0, action_target_positions) == -100):
        raise ValueError("Query-State labels omit the real action envelope")

    tensors: dict[str, torch.Tensor] = {}
    for name, value in encoded.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError("Query-State processor outputs must be tensors")
        squeezed = value.squeeze(0)
        if name == "image_grid_thw" and squeezed.ndim == 1:
            squeezed = squeezed.unsqueeze(0)
        tensors[name] = squeezed.contiguous().detach()
    tensors["labels"] = labels.squeeze(0).contiguous().detach()
    return QueryStateRenderedRow(
        schema=QUERY_STATE_RENDERED_ROW_SCHEMA,
        row=row,
        rendered_text=rendered,
        input_ids=one_row,
        action_boundary_index=int(positions.action_start_index),
        encoded_tensors=tensors,
    )


class FreshQueryStateDINOTeacher:
    """Read only original archived observations through the frozen DINO path."""

    def __init__(self, dino: FrozenDINOGridTargets) -> None:
        if not isinstance(dino, FrozenDINOGridTargets):
            raise TypeError(
                "Query-State fresh teacher rejects cached/protocol-only DINO targets"
            )
        if (
            dino.grid_size != 4
            or dino.identity != DINOV2_LARGE_IDENTITY
            or dino.identity.hidden_size != 1024
        ):
            raise ValueError("Query-State DINO teacher must be pinned DINOv2-large 4x4x1024")
        model = getattr(dino, "model", None)
        if isinstance(model, torch.nn.Module):
            if model.training or any(parameter.requires_grad for parameter in model.parameters()):
                raise ValueError("Query-State DINO model must be frozen and in eval mode")
        self.dino = dino

    @torch.no_grad()
    def build_many(
        self,
        rendered: tuple[QueryStateRenderedRow, ...],
    ) -> tuple[torch.Tensor, ...]:
        if not rendered:
            return ()
        paths = [item.row.original_image_path for item in rendered]
        targets = self.dino.load(paths, device=torch.device("cpu")).detach().float().cpu()
        if targets.shape != (len(rendered), 16, 1024) or not torch.isfinite(targets).all():
            raise RuntimeError("Query-State DINO teacher returned invalid targets")
        return tuple(targets[index].clone() for index in range(len(rendered)))


def prepare_query_state_row(
    rendered: QueryStateRenderedRow,
    *,
    dino_regions: torch.Tensor,
    source_manifest_identity: str,
) -> QueryStatePreparedRow:
    """Bind one encoded real response to a detached original-observation target."""

    if rendered.schema != QUERY_STATE_RENDERED_ROW_SCHEMA:
        raise ValueError("Query-State rendered row schema mismatch")
    if not _is_sha256(source_manifest_identity):
        raise ValueError("Query-State source manifest identity must be SHA256")
    encoded = rendered.encoded_tensors
    required = {"input_ids", "labels"}
    if not required <= set(encoded):
        raise ValueError("Query-State encoded row requires input_ids and labels")
    forbidden = sorted(_FORBIDDEN_STUDENT_KEYS & set(encoded))
    if forbidden:
        raise ValueError("Query-State row may not contain cached student tensors: " + forbidden[0])
    for name, value in encoded.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Query-State encoded inputs must be named tensors")
        if value.requires_grad:
            raise ValueError("Query-State encoded inputs may not own autograd graphs")
    if encoded["input_ids"].ndim != 1 or encoded["labels"].shape != encoded["input_ids"].shape:
        raise ValueError("Query-State encoded token rows are malformed")
    if dino_regions.shape != (16, 1024) or dino_regions.requires_grad:
        raise ValueError("Query-State DINO target must be detached (16,1024)")
    if not torch.isfinite(dino_regions).all():
        raise ValueError("Query-State DINO target must be finite")
    row = rendered.row
    image = Path(row.original_image_path)
    if not image.is_file() or sha256_file(image) != row.original_image_sha256:
        raise ValueError("Query-State original observation identity changed")
    require_archived_assistant_response(row.archived_assistant_response, source="archived")
    return QueryStatePreparedRow(
        schema=QUERY_STATE_PREPARED_ROW_SCHEMA,
        source_manifest_identity=source_manifest_identity,
        record_id=row.record_id,
        step_index=row.step_index,
        split=row.split,
        original_image_path=row.original_image_path,
        original_image_sha256=row.original_image_sha256,
        archived_assistant_response=row.archived_assistant_response,
        executed_action_index=row.executed_action_index,
        encoded_tensors={name: value.detach() for name, value in encoded.items()},
        dino_regions=dino_regions.detach().float(),
    )


__all__ = [
    "FreshQueryStateDINOTeacher",
    "QUERY_STATE_PREPARED_ROW_SCHEMA",
    "QUERY_STATE_RENDERED_ROW_SCHEMA",
    "QueryStatePreparedRow",
    "QueryStateRenderedRow",
    "prepare_query_state_row",
    "render_query_state_row",
]
