from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nimloth.training.sft1.driver import (
    SFT1V2DataCursor,
    assert_gradient_checkpointing_train_mode,
    deterministic_epoch_schedule,
    deterministic_update_schedule,
    resume_schedule,
    run_sft1_v2_epochs,
)
from nimloth.training.sft1.experiment_config import load_sft1_v2_config
from nimloth.training.sft1.real_rows import EARLY4_ROW_SCHEMA, SFT1V2Early4Row
from nimloth.training.sft1.verl_worker import SFT1V2UpdateResult


def test_deterministic_schedule_partitions_whole_rows_pads_ranks_and_resumes_exactly() -> None:
    schedules = []
    identities = []
    for rank in range(3):
        schedule, identity = deterministic_epoch_schedule(
            tuple(range(8)), epoch=2, seed=176, rank=rank, world_size=3
        )
        schedules.append(schedule)
        identities.append(identity)
    assert len(set(identities)) == 1
    assert len({len(schedule) for schedule in schedules}) == 1
    real = [item.ordinal for schedule in schedules for item in schedule if item.row_valid]
    assert sorted(real) == list(range(8))
    assert sum(not item.row_valid for schedule in schedules for item in schedule) == 1

    cursor = SFT1V2DataCursor(
        epoch=2, update_index=1, consumed_rank_rows=1,
        schedule_identity=identities[0], world_size=3, rank=0,
    )
    remaining = resume_schedule(
        schedules[0], cursor, expected_identity=identities[0], rank=0, world_size=3
    )
    assert remaining == schedules[0][1:]
    with pytest.raises(ValueError, match="identity"):
        resume_schedule(
            schedules[0], cursor, expected_identity="0" * 64, rank=0, world_size=3
        )


def test_update_schedule_keeps_b2_contrastive_and_one_global_movement_label() -> None:
    movement = frozenset(range(6))
    schedules = []
    identities = []
    for rank in range(2):
        schedule, identity = deterministic_update_schedule(
            tuple(range(10)),
            movement_ordinals=movement,
            epoch=0,
            seed=7,
            rank=rank,
            world_size=2,
            rows_per_rank_update=2,
        )
        schedules.append(schedule)
        identities.append(identity)
    assert identities[0] == identities[1]
    assert all(len(schedule) % 2 == 0 for schedule in schedules)
    for start in range(0, len(schedules[0]), 2):
        global_update = [
            item for schedule in schedules for item in schedule[start : start + 2]
        ]
        assert any(item.ordinal in movement for item in global_update if item.row_valid)


def test_epoch_runner_executes_epoch0_three_epochs_checkpoints_and_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_sft1_v2_config(
        root / "configs/training/sft1/state_interface_v2_early4_report_first.yaml"
    )
    config = replace(
        config,
        selection=replace(config.selection, train_records=1, train_rows=4),
        checkpoint=replace(config.checkpoint, cadence_steps=100),
    )
    rows = tuple(
        SFT1V2Early4Row(
            schema=EARLY4_ROW_SCHEMA,
            ordinal=index,
            source_path="train.jsonl",
            source_sha256="a" * 64,
            split="train",
            record_id=f"record-{index}",
            step_index=0,
            original_image_path="image.png",
            original_image_sha256="b" * 64,
            image_content_group=f"image-{index}",
            instruction="navigate to the Mug in the room and be as close as possible to it",
            instruction_equivalence_group="group",
            archived_assistant_response="<think>real</think><|latent_state|><|action_start|>",
            executed_action_index=0,
            movement_success=True,
            external_eligible=True,
            record={},
        )
        for index in range(4)
    )

    class Core:
        device = torch.device("cpu")
        def update(self, _data):
            return SFT1V2UpdateResult(
                metrics={
                    **{f"loss/{name}": 0.1 for name in (
                        "visual_content", "visual_relation", "instruction_cosine",
                        "instruction_contrastive", "observed_feasibility", "actor_kl",
                        "state_policy_kl",
                    )},
                    **{f"count/{name}": 2.0 for name in (
                        "visual_content", "visual_relation", "instruction_cosine",
                        "instruction_contrastive", "observed_feasibility", "actor_kl",
                        "state_policy_kl",
                    )},
                },
                gradient_norm=0.5,
                micro_batch_count=1,
            )

    assembly = SimpleNamespace(
        worker=SimpleNamespace(core=Core()),
        loaded_backbone=SimpleNamespace(processor=object()),
    )
    monkeypatch.setattr(
        "nimloth.training.sft1.driver.build_update_dataproto",
        lambda *args, **kwargs: SimpleNamespace(
            batch={"token_counts": torch.tensor([10, 11])}
        ),
    )
    validations: list[int] = []
    checkpoints: list[int] = []
    updates: list[int] = []
    result = run_sft1_v2_epochs(
        assembly=assembly,
        config=config,
        rows=rows,
        cache_reader=object(),
        manifest=object(),
        repo_root=tmp_path,
        rank=0,
        world_size=1,
        seed=7,
        checkpoint_callback=lambda epoch, step, cursor: (
            checkpoints.append(epoch) or tmp_path / f"ckpt-{epoch}-{step}"
        ),
        validation_callback=lambda epoch, step, runtime: (
            validations.append(epoch) or tmp_path / f"report-{epoch}", False
        ),
        update_callback=lambda epoch, step, metrics: updates.append(step),
    )
    assert result.final_epoch == 3
    assert result.global_step == 6
    assert validations == [0, 1, 2, 3]
    assert checkpoints == [1, 2, 3]
    assert updates == list(range(1, 7))


class _CheckpointGate(nn.Module):
    def __init__(self, active: bool) -> None:
        super().__init__()
        self.gradient_checkpointing = active
        self.weight = nn.Parameter(torch.ones(1))


def test_gradient_checkpointing_requires_the_actual_qwen_train_mode_gate() -> None:
    active = _CheckpointGate(True).train()
    assert_gradient_checkpointing_train_mode(active)
    active.eval()
    with pytest.raises(RuntimeError, match="train mode"):
        assert_gradient_checkpointing_train_mode(active)
    with pytest.raises(RuntimeError, match="train mode"):
        assert_gradient_checkpointing_train_mode(_CheckpointGate(False).train())
