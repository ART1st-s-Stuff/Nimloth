from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

from nimloth.agent import NimlothPromptTemplate
from nimloth.backbone.base import BackboneBatch, LoadedBackbone
from nimloth.backbone.dino_grid import (
    DINOV2_LARGE_IDENTITY,
    FrozenDINOGridTargets,
)
from nimloth.backbone.qwen25vl.model import Qwen25VLBackbone
from nimloth.backbone.qwen25vl.state_training import QwenStateTrainingOutput
from nimloth.backbone.qwen25vl.tuning import configure_qwen_tuning
from nimloth.latent import (
    LatentActionTokens,
    find_all_latent_state_blocks,
    latent_state_block,
    latent_state_tokens,
    special_token_ids,
)
from nimloth.training.sft1 import query_state_adapter
from nimloth.training.sft1.data import sha256_file
from nimloth.training.sft1.query_state import (
    QueryStateNormalization,
    SFT1QueryStateObjective,
    SFT1QueryStateTrainingRoot,
    query_state_trainable_parameter_groups,
)
from nimloth.training.sft1.query_state_adapter import (
    QUERY_STATE_DATAPROTO_SCHEMA,
    build_query_state_dataproto,
    query_state_update_inputs,
)
from nimloth.training.sft1.query_state_checkpoint import (
    QUERY_STATE_CHECKPOINT_SCHEMA,
    QueryStateResumeControl,
    QueryStateResumeIdentity,
    export_direct_query_state_artifact,
    load_direct_query_state_artifact,
    load_query_state_resume_checkpoint,
    save_query_state_resume_checkpoint,
)
from nimloth.training.sft1.query_state_data import (
    QUERY_STATE_PREPARED_ROW_SCHEMA,
    FreshQueryStateDINOTeacher,
    prepare_query_state_row,
    render_query_state_row,
)
from nimloth.training.sft1.query_state_runtime import (
    QueryStateProductionContract,
    assemble_query_state_training_root,
    construct_query_state_production_root,
)
from nimloth.training.sft1.real_rows import EARLY4_ROW_SCHEMA, SFT1V2Early4Row
from nimloth.training.verl.runtime import MixedPrecisionConfig
from nimloth.wm.grid import DirectSlotProjector
from tests.training.sft1._state_v2_fixtures import pre_rl_trajectory_record


class _Tokenizer:
    unk_token_id = -1
    pad_token_id = 0

    def __init__(self) -> None:
        tokens = LatentActionTokens()
        all_tokens = (
            *latent_state_tokens(16),
            tokens.action_start,
            tokens.action_end,
            *tokens.action_tokens,
        )
        self.ids = {token: 10 + index for index, token in enumerate(all_tokens)}
        self.ids["<|image_pad|>"] = 700

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
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[Any]]:
        assert not add_special_tokens and not padding
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        specials = sorted(self.ids, key=len, reverse=True)
        index = 0
        while index < len(text):
            special = next(
                (token for token in specials if text.startswith(token, index)),
                None,
            )
            if special is None:
                ids.append(1000 + ord(text[index]))
                offsets.append((index, index + 1))
                index += 1
            else:
                ids.append(self.ids[special])
                offsets.append((index, index + len(special)))
                index += len(special)
        if truncation and max_length is not None:
            ids = ids[:max_length]
            offsets = offsets[:max_length]
        result: dict[str, list[Any]] = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        parts: list[str] = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                payload = content
            else:
                payload = "".join(
                    item["text"] if item["type"] == "text" else "<image>"
                    for item in content
                )
            parts.append(f"<{message['role']}>{payload}</{message['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, *, text, images, padding, truncation, return_tensors):
        assert padding is False and truncation is False and return_tensors == "pt"
        encoded = self.tokenizer(
            text[0], add_special_tokens=False, return_offsets_mapping=True
        )
        # Mimic Qwen's multimodal processor expanding an image placeholder so
        # tokenizer-only character offsets are not processed input_ids indices.
        image_starts = {
            index
            for index in range(len(text[0]))
            if text[0].startswith("<image>", index)
        }
        expanded: list[int] = []
        for token_id, (start, _stop) in zip(
            encoded["input_ids"], encoded["offset_mapping"], strict=True
        ):
            if start in image_starts:
                expanded.extend((700, 700, 700))
            expanded.append(token_id)
        length = len(expanded)
        return {
            "input_ids": torch.tensor([expanded], dtype=torch.long),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }


class _FakeDataProto:
    def __init__(self, *, batch, non_tensor_batch, meta_info) -> None:
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    @classmethod
    def from_dict(cls, *, tensors, non_tensors, meta_info):
        return cls(batch=tensors, non_tensor_batch=non_tensors, meta_info=meta_info)

    def __len__(self) -> int:
        return int(next(iter(self.batch.values())).shape[0])


class _InputBuilder:
    def collate_encoded(self, rows, *, include_labels: bool) -> BackboneBatch:
        assert include_labels is True
        max_length = max(int(row["input_ids"].numel()) for row in rows)
        input_ids = torch.zeros(len(rows), max_length, dtype=torch.long)
        labels = torch.full((len(rows), max_length), -100, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), max_length, dtype=torch.long)
        for index, row in enumerate(rows):
            length = int(row["input_ids"].numel())
            input_ids[index, :length] = row["input_ids"]
            labels[index, :length] = row["labels"]
            attention_mask[index, :length] = 1
        return BackboneBatch(
            {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        )


class _DINO(FrozenDINOGridTargets):
    def __init__(self) -> None:
        self.grid_size = 4
        self.identity = DINOV2_LARGE_IDENTITY
        self.model = nn.Linear(1, 1).requires_grad_(False).eval()
        self.paths: list[str] = []

    def load(self, paths, *, device: torch.device) -> torch.Tensor:
        self.paths = [str(value) for value in paths]
        return torch.arange(len(paths) * 16 * 1024, dtype=torch.float32).reshape(
            len(paths), 16, 1024
        ).to(device)


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.query_seed = nn.Parameter(torch.randn(16, 2048))
        self.model.visual = nn.Module()
        self.model.visual.merger = nn.Linear(2, 2)
        self.lm_head = nn.Linear(2048, 32, bias=False)


class _TransportBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = configure_qwen_tuning(
            _FakeQwen(),
            argparse.Namespace(lora=False, llm_tune="full", vision_tune="freeze"),
        )
        self.calls = 0

    def forward_state_training(self, batch) -> QwenStateTrainingOutput:
        self.calls += 1
        labels = batch.backbone_batch.tensors["labels"]
        batch_size = int(labels.shape[0])
        hidden = self.language_model.model.language_model.query_seed.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        logits = self.language_model.lm_head(hidden.mean(dim=1))
        return QwenStateTrainingOutput(
            query_hidden=hidden,
            action_logits=logits[:, :8],
            lm_loss_sum=logits.float().square().sum(),
            lm_valid_token_count=int((labels != -100).sum().item()),
        )


def _early_row(tmp_path: Path, *, step_index: int = 0) -> SFT1V2Early4Row:
    record, _image = pre_rl_trajectory_record(tmp_path, latent_token_count=8)
    if step_index == 1:
        prompt = NimlothPromptTemplate(latent_token_count=8, action_count=8)
        record["action_indices"].append(2)
        record["actions"].append("move_right")
        record["assistant_responses"].append(
            prompt.assistant_response(
                2,
                thought="The second archived observation supports moving right.",
            )
        )
        record["think_texts"].append(
            "The second archived observation supports moving right."
        )
        next_image = tmp_path / "row-1-after-second.png"
        next_image.write_bytes(b"after-second")
        record["image_paths"].append(str(next_image))
        record["observation_texts"].append(
            "After your action. The environment feedback is: "
            "Last action is executed successfully.\n<image>"
        )
        record["step"] = 2
    elif step_index != 0:
        raise ValueError("test fixture supports only step 0 or 1")
    for path in record["image_paths"]:
        Image.new("RGB", (2, 2), color=(1, 2, 3)).save(path)
    image = Path(record["image_paths"][step_index])
    response = record["assistant_responses"][step_index]
    instruction = "Find the target object."
    observation = record["observation_texts"][0]
    start = observation.index("Human Instruction: ") + len("Human Instruction: ")
    stop = observation.index("\nDecide your next action(s).", start)
    return SFT1V2Early4Row(
        schema=EARLY4_ROW_SCHEMA,
        ordinal=0,
        source_path="train.jsonl",
        source_sha256="a" * 64,
        split="train",
        record_id=str(record["id"]),
        step_index=step_index,
        original_image_path=str(image),
        original_image_sha256=sha256_file(image),
        image_content_group=sha256_file(image),
        instruction=instruction,
        instruction_char_span=(start, stop),
        instruction_equivalence_group=hashlib.sha256(instruction.encode()).hexdigest(),
        archived_assistant_response=response,
        executed_action_index=int(record["action_indices"][step_index]),
        movement_success=True,
        external_eligible=True,
        record=record,
    )


def _loaded_backbone(
    *,
    query_adapter=None,
    dtype: torch.dtype = torch.float32,
) -> LoadedBackbone:
    model = configure_qwen_tuning(
        _FakeQwen().to(dtype=dtype),
        argparse.Namespace(lora=False, llm_tune="full", vision_tune="freeze"),
    )
    backbone = Qwen25VLBackbone(
        model,
        token_id_map={},
        device=torch.device("cpu"),
        latent_token_count=16,
        lora=False,
        vision_tune="freeze",
    )
    return LoadedBackbone(
        backbone=backbone,
        processor=object(),
        token_id_map={},
        added_special_token_count=0,
        base_model_path="fake",
        query_adapter=query_adapter,
    )


def test_full_archived_response_renderer_masks_exact_queries_and_labels_final_span(
    tmp_path: Path,
) -> None:
    row = _early_row(tmp_path)
    processor = _Processor()
    rendered = render_query_state_row(row, processor=processor, max_length=8192)
    tokens = LatentActionTokens()
    assert len(rendered.diagnostic_image_token_indices) == 3
    assert all(
        rendered.input_ids[index].item() == 700
        for index in rendered.diagnostic_image_token_indices
    )
    start, stop = rendered.diagnostic_instruction_token_span
    assert stop > start
    assert rendered.input_ids[start:stop].numel() > 0
    normalized = row.archived_assistant_response.replace(
        latent_state_block(8), latent_state_block(16)
    )

    assert normalized in rendered.rendered_text
    assert tokens.action_start + tokens.action_tokens[0] + tokens.action_end in normalized
    labels = rendered.encoded_tensors["labels"]
    input_ids = rendered.encoded_tensors["input_ids"]
    token_map = special_token_ids(processor.tokenizer, latent_token_count=16)
    blocks = find_all_latent_state_blocks(
        input_ids, token_map, tokens, latent_token_count=16
    )
    assert blocks
    assert all(torch.all(labels[list(block)] == -100) for block in blocks)
    assert labels[0].item() == -100
    boundary = rendered.action_boundary_index
    assert torch.equal(labels[boundary : boundary + 3], input_ids[boundary : boundary + 3])
    cot_probe = 1000 + ord("T")
    cot_positions = torch.nonzero(input_ids == cot_probe, as_tuple=False).flatten()
    assert any(labels[int(index)] != -100 for index in cot_positions)

    multiturn = render_query_state_row(
        _early_row(tmp_path, step_index=1), processor=processor, max_length=8192
    )
    multi_ids = multiturn.encoded_tensors["input_ids"]
    multi_labels = multiturn.encoded_tensors["labels"]
    action_start_id = token_map[tokens.action_start]
    boundaries = torch.nonzero(
        multi_ids == action_start_id, as_tuple=False
    ).flatten().tolist()
    assert len(boundaries) == 2
    assert multi_labels[boundaries[0] + 1].item() == -100
    assert torch.equal(
        multi_labels[boundaries[-1] : boundaries[-1] + 3],
        multi_ids[boundaries[-1] : boundaries[-1] + 3],
    )


def test_rendered_forensic_row_binds_complete_prompt_and_encoding_provenance(
    tmp_path: Path,
) -> None:
    rendered = render_query_state_row(
        _early_row(tmp_path, step_index=1),
        processor=_Processor(),
        max_length=8192,
    )
    required = {
        "prompt_history_identity",
        "messages_identity",
        "renderer_identity",
        "template_identity",
        "encoded_input_identity",
        "response_source",
    }
    missing = sorted(name for name in required if not hasattr(rendered, name))

    assert not missing, f"forensic row provenance is incomplete: {missing}"
    assert rendered.response_source == "archived"
    for name in required - {"response_source"}:
        identity = getattr(rendered, name)
        assert isinstance(identity, str) and len(identity) == 64
        assert set(identity) <= set("0123456789abcdef")


def test_original_observation_dino_and_distinct_dataproto_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rendered = render_query_state_row(
        _early_row(tmp_path), processor=_Processor(), max_length=8192
    )
    dino = _DINO()
    targets = FreshQueryStateDINOTeacher(dino).build_many((rendered,))
    cached_like = type(
        "_CachedLike",
        (),
        {"grid_size": 4, "identity": DINOV2_LARGE_IDENTITY},
    )()
    with pytest.raises(TypeError, match="rejects cached"):
        FreshQueryStateDINOTeacher(cached_like)  # type: ignore[arg-type]
    prepared = prepare_query_state_row(
        rendered,
        dino_regions=targets[0],
        source_manifest_identity="b" * 64,
    )
    monkeypatch.setattr(
        query_state_adapter,
        "_load_pinned_dataproto_type",
        lambda: _FakeDataProto,
    )
    data = build_query_state_dataproto((prepared,))
    inputs = query_state_update_inputs(data, input_builder=_InputBuilder())
    diagnostic_inputs = query_state_update_inputs(
        data, input_builder=_InputBuilder(), include_diagnostics=True
    )

    assert inputs.student_batch.diagnostic_image_token_indices is None
    assert diagnostic_inputs.student_batch.diagnostic_image_token_indices == (
        rendered.diagnostic_image_token_indices,
    )
    assert diagnostic_inputs.student_batch.diagnostic_instruction_token_spans == (
        rendered.diagnostic_instruction_token_span,
    )
    assert dino.paths == [rendered.row.original_image_path]
    assert prepared.schema == QUERY_STATE_PREPARED_ROW_SCHEMA
    assert data.meta_info["schema"] == QUERY_STATE_DATAPROTO_SCHEMA
    assert "query_hidden" not in data.batch and "state" not in data.batch
    assert "labels" in inputs.student_batch.backbone_batch.tensors
    assert inputs.student_batch.response_sources == ("archived",)
    assert inputs.targets.dino_regions.shape == (1, 16, 1024)
    assert not inputs.targets.dino_regions.requires_grad
    transport_backbone = _TransportBackbone()
    root = SFT1QueryStateTrainingRoot(
        transport_backbone,
        SFT1QueryStateObjective(projector=DirectSlotProjector()),
    )
    lm_count = int(
        (inputs.student_batch.backbone_batch.tensors["labels"] != -100).sum().item()
    )
    output = root(
        inputs.student_batch,
        inputs.targets,
        QueryStateNormalization(
            global_state_valid_element_count=16 * 1024,
            global_lm_valid_token_count=lm_count,
            gradient_average_world_size=1,
        ),
    )
    assert transport_backbone.calls == 1
    assert output.state.shape == (1, 16, 1024)
    assert set(output.losses) == {"direct_state_mse", "lm_ce"}

    encoded = dict(prepared.encoded_tensors)
    encoded["student_state_cache_v2"] = torch.zeros(16, 2048)
    bad = type(prepared)(**{**prepared.__dict__, "encoded_tensors": encoded})
    with pytest.raises(ValueError, match="cached student"):
        build_query_state_dataproto((bad,))


def test_query_state_update_inputs_accepts_production_tensordict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from verl import DataProto

    rendered = render_query_state_row(
        _early_row(tmp_path), processor=_Processor(), max_length=8192
    )
    prepared = prepare_query_state_row(
        rendered,
        dino_regions=torch.zeros(16, 1024),
        source_manifest_identity="c" * 64,
    )
    monkeypatch.setattr(
        query_state_adapter,
        "_load_pinned_dataproto_type",
        lambda: DataProto,
    )

    data = build_query_state_dataproto((prepared,))
    assert type(data.batch).__name__ == "TensorDict"
    inputs = query_state_update_inputs(data, input_builder=_InputBuilder())

    assert inputs.record_ids == (prepared.record_id,)
    assert inputs.targets.dino_regions.shape == (1, 16, 1024)
    assert inputs.student_batch.response_sources == ("archived",)


def test_production_constructor_aligns_direct_head_to_loaded_qwen_dtype() -> None:
    constructed = construct_query_state_production_root(
        _loaded_backbone(dtype=torch.bfloat16)
    )

    floating_dtypes = {
        parameter.dtype
        for parameter in constructed.root.backbone.parameters()
        if parameter.is_floating_point()
    }
    assert floating_dtypes == {torch.bfloat16}
    assert constructed.root.objective.projector.linear.weight.dtype == torch.bfloat16

    mixed = _loaded_backbone(dtype=torch.bfloat16)
    mixed.backbone.model.lm_head.to(dtype=torch.float32)
    with pytest.raises(ValueError, match="loaded Qwen has mixed floating parameter dtypes"):
        construct_query_state_production_root(mixed)


def test_production_constructor_and_pre_wrap_optimizer_groups_fail_closed() -> None:
    constructed = construct_query_state_production_root(_loaded_backbone())
    assembly = assemble_query_state_training_root(
        constructed=constructed,
        device=torch.device("cpu"),
        repo_root=Path.cwd(),
        wrap_policy={"disable": False},
        mixed_precision=MixedPrecisionConfig(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        language_learning_rate=1e-5,
        direct_state_learning_rate=1e-4,
        weight_decay=0.0,
        adam_betas=(0.9, 0.95),
        adam_epsilon=1e-8,
        wrap=lambda module: module,
    )

    assert tuple(group["group_name"] for group in assembly.optimizer.param_groups) == (
        "language",
        "direct_state",
    )
    parameter_names = {
        id(parameter): name for name, parameter in assembly.root.named_parameters()
    }
    assert tuple(
        parameter_names[id(parameter)]
        for parameter in assembly.optimizer.param_groups[0]["params"]
    ) == constructed.inventory.language_trainable
    assert tuple(
        parameter_names[id(parameter)]
        for parameter in assembly.optimizer.param_groups[1]["params"]
    ) == constructed.inventory.direct_state_trainable
    optimizer_ids = {
        id(parameter)
        for group in assembly.optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_ids == {
        id(parameter) for parameter in assembly.root.parameters() if parameter.requires_grad
    }
    assert constructed.contract == QueryStateProductionContract()
    assert constructed.root.objective.projector.linear.bias is None

    tampered = construct_query_state_production_root(_loaded_backbone())
    tampered.root.objective.projector.to(dtype=torch.bfloat16)
    wrapper_called = False

    def forbidden_wrapper(module: nn.Module) -> nn.Module:
        nonlocal wrapper_called
        wrapper_called = True
        return module

    with pytest.raises(ValueError, match="mixed floating parameter dtypes"):
        assemble_query_state_training_root(
            constructed=tampered,
            device=torch.device("cpu"),
            repo_root=Path.cwd(),
            wrap_policy={"disable": False},
            mixed_precision=MixedPrecisionConfig(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            ),
            language_learning_rate=1e-5,
            direct_state_learning_rate=1e-4,
            weight_decay=0.0,
            adam_betas=(0.9, 0.95),
            adam_epsilon=1e-8,
            wrap=forbidden_wrapper,
        )
    assert wrapper_called is False
    assert tampered.root.objective.projector.linear.weight.dtype == torch.bfloat16

    with pytest.raises(ValueError, match="input-forward device"):
        assemble_query_state_training_root(
            constructed=constructed,
            device=torch.device("meta"),
            repo_root=Path.cwd(),
            wrap_policy={"disable": False},
            mixed_precision=MixedPrecisionConfig(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            ),
            language_learning_rate=1e-5,
            direct_state_learning_rate=1e-4,
            weight_decay=0.0,
            adam_betas=(0.9, 0.95),
            adam_epsilon=1e-8,
            wrap=lambda module: module,
        )
    with pytest.raises(ValueError, match="query additive adapter"):
        construct_query_state_production_root(_loaded_backbone(query_adapter=object()))
    loaded = _loaded_backbone()
    loaded.backbone.model.model.visual.merger.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="visual parameter is trainable"):
        construct_query_state_production_root(loaded)
    with pytest.raises(ValueError, match="requires full language"):
        QueryStateProductionContract(llm_tune="freeze")


def test_direct_artifact_and_same_identity_resume_round_trip_reject_legacy(
    tmp_path: Path,
) -> None:
    constructed = construct_query_state_production_root(_loaded_backbone())
    root = constructed.root
    groups = [
        {"params": group.parameters, "group_name": group.name, "lr": 1e-4}
        for group in query_state_trainable_parameter_groups(root)
    ]
    optimizer = torch.optim.AdamW(groups)
    identity = QueryStateResumeIdentity(
        source_commit="1" * 40,
        source_manifest_identity="2" * 64,
        config_identity="3" * 64,
        run_identity="4" * 64,
        world_size=1,
    )
    artifact = tmp_path / "direct_state.pt"
    export_direct_query_state_artifact(
        artifact,
        projector=root.objective.projector,
        source_identity=identity,
        metadata={"query_mode": "inject"},
    )
    loaded_projector, metadata = load_direct_query_state_artifact(
        artifact, expected_source_identity=identity
    )
    torch.testing.assert_close(
        loaded_projector.linear.weight, root.objective.projector.linear.weight
    )
    assert metadata == {"query_mode": "inject"}

    local_shard = DirectSlotProjector()
    local_shard.linear.weight = nn.Parameter(torch.zeros(17))
    with pytest.raises(ValueError, match="full finite unsharded"):
        export_direct_query_state_artifact(
            tmp_path / "invalid-shard.pt",
            projector=local_shard,
            source_identity=identity,
        )

    checkpoint = tmp_path / "resume.pt"
    expected = {
        name: parameter.detach().clone()
        for name, parameter in root.named_parameters()
        if parameter.requires_grad
    }
    save_query_state_resume_checkpoint(
        checkpoint,
        root=root,
        optimizer=optimizer,
        control=QueryStateResumeControl(
            identity=identity, global_step=7, data_cursor={"ordinal": 11}
        ),
        scheduler_state={"last_epoch": 7},
    )
    with torch.no_grad():
        for parameter in root.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    control, scheduler = load_query_state_resume_checkpoint(
        checkpoint,
        root=root,
        optimizer=optimizer,
        expected_identity=identity,
    )
    assert control.global_step == 7 and control.data_cursor == {"ordinal": 11}
    assert scheduler == {"last_epoch": 7}
    for name, parameter in root.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, expected[name])

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model"]["objective.projector.linear.weight"] = payload["model"][
        "objective.projector.linear.weight"
    ].to(dtype=torch.bfloat16)
    dtype_mismatch = tmp_path / "resume-dtype-mismatch.pt"
    torch.save(payload, dtype_mismatch)
    with torch.no_grad():
        for parameter in root.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    before_mismatched_restore = {
        name: parameter.detach().clone()
        for name, parameter in root.named_parameters()
        if parameter.requires_grad
    }
    with pytest.raises(ValueError, match="checkpoint tensor dtype mismatch"):
        load_query_state_resume_checkpoint(
            dtype_mismatch,
            root=root,
            optimizer=optimizer,
            expected_identity=identity,
        )
    for name, parameter in root.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, before_mismatched_restore[name])

    legacy = tmp_path / "legacy.pt"
    torch.save(
        {
            "schema": "nimloth_sft1_state_v2_checkpoint_v2",
            "shared_slot_projector": True,
        },
        legacy,
    )
    with pytest.raises(ValueError, match="legacy/cross-stage"):
        load_query_state_resume_checkpoint(
            legacy, root=root, optimizer=optimizer, expected_identity=identity
        )
    with pytest.raises(ValueError, match="SharedSlotProjector state artifact"):
        load_direct_query_state_artifact(legacy)
    mismatched_identity = QueryStateResumeIdentity(
        source_commit="1" * 40,
        source_manifest_identity="2" * 64,
        config_identity="3" * 64,
        run_identity="5" * 64,
        world_size=1,
    )
    with pytest.raises(ValueError, match="resume identity mismatch"):
        load_query_state_resume_checkpoint(
            checkpoint,
            root=root,
            optimizer=optimizer,
            expected_identity=mismatched_identity,
        )
    assert QUERY_STATE_CHECKPOINT_SCHEMA != "nimloth_sft1_state_v2_checkpoint_v2"
