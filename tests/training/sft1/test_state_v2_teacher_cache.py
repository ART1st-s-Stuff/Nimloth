from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nimloth.training.sft1.data import sha256_file
from nimloth.training.sft1.real_rows import (
    EARLY4_ROW_SCHEMA,
    SFT1V2Early4Row,
    SFT1V2RenderedRow,
)
from nimloth.training.sft1.teacher_cache import (
    SFT1V2FreshTargets,
    SFT1V2TeacherCacheIdentity,
    SFT1V2TeacherCacheReader,
    deterministic_shard_ownership,
    finalize_teacher_cache,
    inspect_teacher_cache,
    prepare_teacher_cache_shard,
)


def _identity() -> SFT1V2TeacherCacheIdentity:
    return SFT1V2TeacherCacheIdentity(
        source_commit="f" * 40,
        actor_checkpoint_sha256="a" * 64,
        actor_config_sha256="0" * 64,
        actor_model_index_sha256="8" * 64,
        actor_action_head_sha256="9" * 64,
        actor_shards_sha256=("b" * 64, "c" * 64),
        processor_sha256="d" * 64, tokenizer_sha256="e" * 64,
        chat_template_sha256="f" * 64, prompt_renderer_sha256="1" * 64,
        token_table_sha256="2" * 64, query_action_contract_sha256="3" * 64,
        dino_checkpoint_sha256="4" * 64, dino_processor_sha256="5" * 64,
        train_trajectory_sha256="6" * 64, validation_trajectory_sha256="7" * 64,
    )


def _rows(tmp_path: Path, count: int = 4) -> tuple[SFT1V2RenderedRow, ...]:
    result = []
    for ordinal in range(count):
        image = tmp_path / f"image-{ordinal}.png"
        image.write_bytes(f"original-image-{ordinal}".encode())
        row = SFT1V2Early4Row(
            schema=EARLY4_ROW_SCHEMA, ordinal=ordinal, source_path="source.jsonl",
            source_sha256="8" * 64, split="train", record_id=f"record-{ordinal}",
            step_index=ordinal % 4, original_image_path=str(image),
            original_image_sha256=sha256_file(image),
            image_content_group=sha256_file(image), instruction=f"instruction-{ordinal}",
            instruction_equivalence_group=f"group-{ordinal}",
            archived_assistant_response="<think>real cot</think><|latent_state|><|action_start|>",
            executed_action_index=ordinal % 8, movement_success=True,
            external_eligible=True, record={},
        )
        result.append(SFT1V2RenderedRow(
            row=row, rendered_text=f"rendered-{ordinal}",
            input_ids=torch.tensor([1, 2, 3, 4]), instruction_token_span=(1, 3),
            action_boundary_index=3, encoded_tensors={"input_ids": torch.tensor([1, 2, 3, 4])},
        ))
    return tuple(result)


class _FakeTeacher:
    def __init__(self, fail_ordinal: int | None = None) -> None:
        self.fail_ordinal = fail_ordinal
        self.calls: list[int] = []

    def build(self, rendered: SFT1V2RenderedRow) -> SFT1V2FreshTargets:
        ordinal = rendered.row.ordinal
        self.calls.append(ordinal)
        assert rendered.instruction_token_span == (1, 3)
        assert Path(rendered.row.original_image_path).read_bytes().startswith(b"original-image")
        if ordinal == self.fail_ordinal:
            raise RuntimeError("interrupted fake teacher")
        logits = torch.arange(8, dtype=torch.float32) + ordinal
        return SFT1V2FreshTargets(
            dino_regions=torch.full((16, 1024), float(ordinal)),
            instruction_teacher=torch.full((2048,), float(ordinal)),
            actor_teacher_log_probs=torch.log_softmax(logits, dim=-1),
        )


def test_cache_shards_have_deterministic_ownership_exact_prefix_resume_and_atomic_root(
    tmp_path: Path,
) -> None:
    rows = _rows(tmp_path)
    output = tmp_path / "cache"
    assert [deterministic_shard_ownership(value, 2) for value in range(4)] == [0, 1, 0, 1]

    interrupted = _FakeTeacher(fail_ordinal=2)
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare_teacher_cache_shard(
            output, rows, shard_index=0, shard_count=2,
            identity=_identity(), teacher=interrupted,
        )
    assert interrupted.calls == [0, 2]
    assert not (output / "COMPLETED").exists()

    resumed = _FakeTeacher()
    first = prepare_teacher_cache_shard(
        output, rows, shard_index=0, shard_count=2,
        identity=_identity(), teacher=resumed,
    )
    second = prepare_teacher_cache_shard(
        output, rows, shard_index=1, shard_count=2,
        identity=_identity(), teacher=_FakeTeacher(),
    )
    assert resumed.calls == [2]
    assert first.row_count == second.row_count == 2
    summary = finalize_teacher_cache(
        output, identity=_identity(), shard_count=2, expected_row_count=4
    )
    assert summary.row_count == 4
    assert inspect_teacher_cache(output) == summary
    assert (output / "manifest.json").is_file()
    assert (output / "COMPLETED").read_text().strip() == summary.root_manifest_sha256

    payload = torch.load(
        output / "shards/shard_00000_of_00002.pt",
        map_location="cpu", weights_only=False,
    )
    forbidden = {"hidden", "query_hidden", "student_hidden", "state", "projected_state", "encoded_tensors"}
    assert all(not (forbidden & set(row)) for row in payload["rows"])
    with pytest.raises(FileExistsError, match="immutable"):
        prepare_teacher_cache_shard(
            output, rows, shard_index=0, shard_count=2,
            identity=_identity(), teacher=_FakeTeacher(),
        )


class _BatchTeacher(_FakeTeacher):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def build_many(self, rendered):
        self.batch_sizes.append(len(rendered))
        return tuple(self.build(row) for row in rendered)


def test_cache_batches_fresh_forwards_and_reader_validates_once(
    tmp_path: Path,
) -> None:
    rows = _rows(tmp_path, count=5)
    output = tmp_path / "cache"
    teacher = _BatchTeacher()
    prepare_teacher_cache_shard(
        output,
        rows,
        shard_index=0,
        shard_count=1,
        identity=_identity(),
        teacher=teacher,
        teacher_batch_size=2,
    )
    assert teacher.batch_sizes == [2, 2, 1]
    finalize_teacher_cache(
        output,
        identity=_identity(),
        shard_count=1,
        expected_row_count=5,
    )
    reader = SFT1V2TeacherCacheReader(
        output,
        manifest_identity="9" * 64,
    )
    assert reader.summary.row_count == 5
    assert reader.load(4).record_id == "record-4"
    assert reader.load(4).manifest_identity == "9" * 64
