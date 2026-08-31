from __future__ import annotations

import contextlib
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from nimloth.backbone.base import BackboneBatch
from nimloth.training.sft1 import query_state_distributed
from nimloth.training.sft1.query_state import (
    QueryStateNormalization,
    QueryStateObjectiveOutput,
    QueryStateTargets,
)
from nimloth.training.sft1.query_state_adapter import (
    QUERY_STATE_DATAPROTO_SCHEMA,
)
from nimloth.training.sft1.query_state_checkpoint import (
    QUERY_STATE_RANK_CHECKPOINT_SCHEMA,
    QueryStateDistributedControl,
    QueryStateRankState,
    QueryStateResumeIdentity,
    export_query_state_deployable_bundle,
    finalize_query_state_rank_checkpoint,
    load_query_state_forensic_model_for_debug,
    load_query_state_rank_state,
    restore_query_state_rank_state,
    save_query_state_rank_state,
)
from nimloth.training.sft1.query_state_config import (
    QUERY_STATE_CODE_CANARY_CONFIG_SCHEMA,
    bind_query_state_code_canary_identity,
    parse_query_state_code_canary_config,
)
from nimloth.training.sft1.query_state_distributed import (
    QueryStateUpdateCore,
    query_state_global_normalization,
)
from nimloth.training.sft1.query_state_driver import (
    QueryStateDataCursor,
    deterministic_query_state_schedule,
    iter_query_state_updates,
    restore_query_state_distributed_checkpoint,
    resume_query_state_schedule,
    save_query_state_distributed_checkpoint,
)
from nimloth.training.sft1.query_state_validation import (
    QueryStateDiagnosticAccumulator,
)
from nimloth.training.sft1.query_state_data import QUERY_STATE_PREPARED_ROW_SCHEMA
from nimloth.training.sft1.manifest import PINNED_VAGEN_COMMIT, PINNED_VERL_COMMIT
from nimloth.wm.grid import DirectSlotProjector


_RESPONSE = (
    "<think>Use the actual archived observation.</think>"
    "<|latent_state|><|action_start|><|action_(0)|><|action_end|>"
)


class _FakeDataProto:
    def __init__(self, *, batch, non_tensor_batch, meta_info) -> None:
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    def __len__(self) -> int:
        return int(next(iter(self.batch.values())).shape[0])

    def __getitem__(self, indices: torch.Tensor) -> "_FakeDataProto":
        selected = indices.tolist()
        return _FakeDataProto(
            batch={name: value.index_select(0, indices) for name, value in self.batch.items()},
            non_tensor_batch={
                name: np.asarray([value[index] for index in selected], dtype=object)
                for name, value in self.non_tensor_batch.items()
            },
            meta_info=dict(self.meta_info),
        )


class _InputBuilder:
    def collate_encoded(self, rows, *, include_labels: bool) -> BackboneBatch:
        assert include_labels
        length = max(int(row["input_ids"].numel()) for row in rows)
        ids = torch.zeros(len(rows), length, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        attention = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            size = int(row["input_ids"].numel())
            ids[index, :size] = row["input_ids"]
            labels[index, :size] = row["labels"]
            attention[index, :size] = 1
        return BackboneBatch(
            {"input_ids": ids, "labels": labels, "attention_mask": attention}
        )


def _data() -> _FakeDataProto:
    rows = []
    for length in (5, 4, 3):
        ids = torch.arange(1, length + 1, dtype=torch.long)
        labels = ids.clone()
        labels[0] = -100
        rows.append({"input_ids": ids, "labels": labels})
    return _FakeDataProto(
        batch={
            "dino_regions": torch.zeros(3, 16, 1024),
            "token_counts": torch.tensor([5, 4, 3]),
            "step_indices": torch.tensor([0, 1, 2]),
            "row_valid": torch.tensor([True, False, True]),
            "executed_action_indices": torch.tensor([0, 0, 0]),
        },
        non_tensor_batch={
            "encoded_rows": np.asarray(rows, dtype=object),
            "archived_assistant_responses": np.asarray([_RESPONSE] * 3, dtype=object),
            "response_sources": np.asarray(["archived"] * 3, dtype=object),
            "record_ids": np.asarray(["a", "b", "c"], dtype=object),
            "splits": np.asarray(["train"] * 3, dtype=object),
            "original_image_paths": np.asarray(["a.png", "b.png", "c.png"], dtype=object),
            "original_image_sha256": np.asarray(["a" * 64, "b" * 64, "c" * 64], dtype=object),
            "diagnostic_image_token_indices": np.asarray([(), (), ()], dtype=object),
            "diagnostic_instruction_token_spans": np.asarray([(), (), ()], dtype=object),
        },
        meta_info={
            "schema": QUERY_STATE_DATAPROTO_SCHEMA,
            "row_schema": QUERY_STATE_PREPARED_ROW_SCHEMA,
            "training_schema": "nimloth_sft1_query_state_v1",
            "objective_version": "direct_query_state_dino_lm_v1",
            "state_artifact_schema": "nimloth_direct_k16_state_v1",
            "source_manifest_identity": "d" * 64,
            "query_count": 16,
            "state_dim": 1024,
            "vagen_commit": PINNED_VAGEN_COMMIT,
            "verl_commit": PINNED_VERL_COMMIT,
        },
    )


class _FrameworkRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.calls: list[tuple[int, int, QueryStateNormalization]] = []
        self.no_sync_calls = 0
        self.clip_calls = 0

    @contextlib.contextmanager
    def no_sync(self):
        self.no_sync_calls += 1
        yield

    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        self.clip_calls += 1
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    def forward(self, batch, targets, normalization):
        labels = batch.backbone_batch.tensors["labels"]
        lm_count = int((labels != -100).sum().item())
        state_count = int(targets.sample_valid.sum().item()) * 16 * 1024
        self.calls.append((state_count, lm_count, normalization))
        state_sum = self.weight.square() * float(state_count)
        lm_sum = self.weight.square() * float(lm_count)
        total = (
            2.0 * state_sum * normalization.gradient_average_world_size
            / normalization.global_state_valid_element_count
            + lm_sum * normalization.gradient_average_world_size
            / normalization.global_lm_valid_token_count
        )
        size = int(labels.shape[0])
        return QueryStateObjectiveOutput(
            raw_query_hidden=torch.zeros(size, 16, 2048),
            state=torch.zeros(size, 16, 1024),
            action_logits=torch.zeros(size, 8),
            losses={"direct_state_mse": total * 0, "lm_ce": total * 0},
            total_loss=total,
            loss_sums={"direct_state_mse": state_sum, "lm_ce": lm_sum},
            local_valid_counts={
                "direct_state_mse": state_count,
                "lm_ce": lm_count,
            },
        )


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters) -> None:
        super().__init__(parameters, lr=0.01)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def test_global_normalization_reduces_exact_state_and_lm_counts_in_fixed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reduced: list[int] = []

    def global_sum(value: int, device: torch.device) -> int:
        assert device == torch.device("cpu")
        reduced.append(value)
        return value * 2

    monkeypatch.setattr(query_state_distributed, "_global_sum_int", global_sum)
    monkeypatch.setattr(query_state_distributed, "_distributed_world_size", lambda: 2)

    normalization = query_state_global_normalization(
        _data(),
        device=torch.device("cpu"),
    )

    assert reduced == [2 * 16 * 1024, 6]
    assert normalization == QueryStateNormalization(
        global_state_valid_element_count=4 * 16 * 1024,
        global_lm_valid_token_count=12,
        gradient_average_world_size=2,
    )


def test_update_core_uses_update_global_denominators_and_equal_padding_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _FrameworkRoot()
    optimizer = _CountingSGD(root.parameters())
    monkeypatch.setattr(query_state_distributed, "_global_max_int", lambda value, device: 4)
    core = QueryStateUpdateCore(
        root=root,
        optimizer=optimizer,
        input_builder=_InputBuilder(),
        device=torch.device("cpu"),
        max_padded_tokens=5,
        max_rows=1,
        max_grad_norm=1.0,
    )

    result = core.update(_data())

    assert optimizer.step_calls == 1
    assert root.clip_calls == 1
    assert len(root.calls) == 4
    assert root.no_sync_calls == 3
    assert all(
        call[2].global_state_valid_element_count == 2 * 16 * 1024
        and call[2].global_lm_valid_token_count == 6
        for call in root.calls
    )
    assert [(state, lm) for state, lm, _ in root.calls].count((0, 0)) == 2
    assert result.micro_batch_count == 4
    assert result.metrics["count/direct_state_mse"] == 2 * 16 * 1024
    assert result.metrics["count/lm_ce"] == 6


def _config_raw() -> dict[str, Any]:
    return {
        "schema": QUERY_STATE_CODE_CANARY_CONFIG_SCHEMA,
        "optimizer": {
            "name": "adamw",
            "language_learning_rate": 1e-5,
            "direct_state_learning_rate": 1e-4,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "epsilon": 1e-8,
            "scheduler": "constant",
        },
        "runtime": {
            "max_padded_tokens": 1024,
            "max_rows_per_micro_batch": 2,
            "rows_per_rank_update": 4,
            "max_grad_norm": 1.0,
            "world_size": 2,
            "gradient_checkpointing": True,
            "train_mode": True,
            "fsdp_sharding": "full_shard",
            "fsdp_use_orig_params": True,
            "launch_authorized": False,
        },
        "checkpoint": {
            "cadence_updates": 3,
            "at_update_boundary": True,
            "exact_resume": True,
            "immutable_rank_shards": True,
            "save_optimizer": True,
            "save_rng": True,
            "save_data_cursor": True,
            "save_metric_cursor": True,
        },
        "validation": {
            "cadence_updates": 2,
            "report_only": True,
            "model_quality_gate": False,
            "diagnostics": [
                "raw_query_hidden",
                "canonical_state",
                "lm_ce",
                "action_logits",
            ],
        },
    }


def test_code_canary_config_is_explicit_identity_bound_and_non_launching() -> None:
    raw = _config_raw()
    config = parse_query_state_code_canary_config(raw)
    assert len(config.identity) == 64
    identity = bind_query_state_code_canary_identity(
        config,
        source_commit="1" * 40,
        source_manifest_identity="2" * 64,
        run_identity="3" * 64,
    )
    assert identity.world_size == config.runtime.world_size

    missing = _config_raw()
    del missing["optimizer"]["language_learning_rate"]
    with pytest.raises(ValueError, match="missing Query-State config field"):
        parse_query_state_code_canary_config(missing)
    launching = _config_raw()
    launching["runtime"]["launch_authorized"] = True
    with pytest.raises(ValueError, match="non-launching"):
        parse_query_state_code_canary_config(launching)
    unknown = _config_raw()
    unknown["runtime"]["gpu_count"] = 8
    with pytest.raises(ValueError, match="unknown Query-State config field"):
        parse_query_state_code_canary_config(unknown)


def test_raw_row_schedule_partitions_pads_and_resumes_deterministically() -> None:
    schedules = []
    identities = []
    for rank in range(3):
        schedule, identity = deterministic_query_state_schedule(
            tuple(range(8)), epoch=2, seed=17, rank=rank, world_size=3
        )
        schedules.append(schedule)
        identities.append(identity)
    assert len(set(identities)) == 1
    assert len({len(value) for value in schedules}) == 1
    assert sorted(
        item.ordinal
        for schedule in schedules
        for item in schedule
        if item.row_valid
    ) == list(range(8))
    assert sum(not item.row_valid for schedule in schedules for item in schedule) == 1
    assert list(iter_query_state_updates(schedules[0], rows_per_rank_update=2))

    cursor = QueryStateDataCursor(
        epoch=2,
        update_index=1,
        consumed_rank_rows=2,
        schedule_identity=identities[0],
        world_size=3,
        rank=0,
        metric_cursor={"lm_tokens": 6},
    )
    assert resume_query_state_schedule(
        schedules[0],
        cursor,
        expected_identity=identities[0],
        expected_epoch=2,
        rank=0,
        world_size=3,
        rows_per_rank_update=2,
    ) == schedules[0][2:]
    with pytest.raises(ValueError, match="identity"):
        resume_query_state_schedule(
            schedules[0],
            cursor,
            expected_identity="0" * 64,
            expected_epoch=2,
            rank=0,
            world_size=3,
            rows_per_rank_update=2,
        )
    mid_update = QueryStateDataCursor(
        **{**cursor.__dict__, "consumed_rank_rows": 1}
    )
    with pytest.raises(ValueError, match="update boundary"):
        resume_query_state_schedule(
            schedules[0],
            mid_update,
            expected_identity=identities[0],
            expected_epoch=2,
            rank=0,
            world_size=3,
            rows_per_rank_update=2,
        )


def test_diagnostics_accumulate_only_direct_query_state_lm_and_action() -> None:
    accumulator = QueryStateDiagnosticAccumulator()
    output = QueryStateObjectiveOutput(
        raw_query_hidden=torch.ones(2, 16, 2048),
        state=torch.ones(2, 16, 1024),
        action_logits=torch.tensor([[3.0, 1.0, 0, 0, 0, 0, 0, 0], [0, 2.0, 0, 0, 0, 0, 0, 0]]),
        losses={"direct_state_mse": torch.tensor(0.0), "lm_ce": torch.tensor(0.0)},
        total_loss=torch.tensor(0.0),
        loss_sums={
            "direct_state_mse": torch.tensor(float(16 * 1024)),
            "lm_ce": torch.tensor(6.0),
        },
        local_valid_counts={"direct_state_mse": 16 * 1024, "lm_ce": 3},
    )
    accumulator.add(
        output,
        QueryStateTargets(
            dino_regions=torch.zeros(2, 16, 1024),
            sample_valid=torch.tensor([True, False]),
        ),
        executed_action_indices=torch.tensor([0, 1]),
        record_ids=("a", "b"),
    )
    report = accumulator.finalize(
        config_identity="1" * 64,
        source_manifest_identity="2" * 64,
        checkpoint_identity="3" * 64,
        checkpoint_step=4,
    )
    assert report.diagnostic_only
    assert report.automatic_model_quality_pass is None
    assert report.sample_count == 1
    assert report.record_ids == ("a",)
    assert {
        "direct_state/mse",
        "direct_state/cosine",
        "raw_query/norm_mean",
        "canonical_state/norm_mean",
        "lm/ce",
        "action/logit_std",
        "action/executed_margin_mean",
    } <= set(report.metrics)


def _resume_identity(
    world_size: int = 2,
    *,
    experiment_mode: str = "mechanics",
) -> QueryStateResumeIdentity:
    return QueryStateResumeIdentity(
        source_commit="1" * 40,
        source_manifest_identity="2" * 64,
        config_identity="3" * 64,
        run_identity="4" * 64,
        world_size=world_size,
        experiment_mode=experiment_mode,
    )


def _rank_state(
    rank: int,
    identity: QueryStateResumeIdentity | None = None,
) -> QueryStateRankState:
    return QueryStateRankState(
        identity=identity or _resume_identity(),
        model={"objective.projector.linear.weight": torch.tensor([float(rank)])},
        optimizer={"step": rank + 1},
        scheduler={"last_epoch": rank},
        rng={
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        },
    )


def test_rank_sharded_checkpoint_transaction_binds_metric_and_data_cursor(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    for rank in range(2):
        save_query_state_rank_state(
            checkpoint, rank=rank, world_size=2, state=_rank_state(rank)
        )
    control = QueryStateDistributedControl(
        identity=_resume_identity(),
        global_step=7,
        data_cursor={"epoch": 1, "consumed_rank_rows": 4},
        metric_cursor={"lm_tokens": 19, "validation_step": 6},
    )
    finalize_query_state_rank_checkpoint(checkpoint, control=control)
    state, restored = load_query_state_rank_state(
        checkpoint,
        rank=1,
        expected_identity=_resume_identity(),
    )
    assert restored == control
    assert state.optimizer == {"step": 2}
    assert (checkpoint / "COMPLETED").is_file()
    assert (
        json.loads((checkpoint / "control.json").read_text())["schema"]
        == QUERY_STATE_RANK_CHECKPOINT_SCHEMA
    )

    with pytest.raises(FileExistsError, match="immutable"):
        save_query_state_rank_state(
            checkpoint, rank=0, world_size=2, state=_rank_state(0)
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        load_query_state_rank_state(
            checkpoint,
            rank=0,
            expected_identity=QueryStateResumeIdentity(
                **{**asdict(_resume_identity()), "run_identity": "5" * 64}
            ),
        )
    control_path = checkpoint / "control.json"
    original_control = control_path.read_bytes()
    control_path.write_bytes(original_control.replace(b'"global_step": 7', b'"global_step": 8'))
    with pytest.raises(ValueError, match="control hash mismatch"):
        load_query_state_rank_state(
            checkpoint, rank=0, expected_identity=_resume_identity()
        )
    control_path.write_bytes(original_control)
    (checkpoint / "rank_00001_of_00002.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="shard hash mismatch"):
        load_query_state_rank_state(
            checkpoint, rank=0, expected_identity=_resume_identity()
        )

    partial = tmp_path / "partial"
    save_query_state_rank_state(
        partial, rank=0, world_size=2, state=_rank_state(0)
    )
    with pytest.raises(FileNotFoundError, match="incomplete"):
        finalize_query_state_rank_checkpoint(partial, control=control)

    mixed = tmp_path / "mixed-identity"
    save_query_state_rank_state(
        mixed, rank=0, world_size=2, state=_rank_state(0)
    )
    wrong_identity = QueryStateResumeIdentity(
        **{**asdict(_resume_identity()), "run_identity": "5" * 64}
    )
    save_query_state_rank_state(
        mixed, rank=1, world_size=2, state=_rank_state(1, wrong_identity)
    )
    with pytest.raises(ValueError, match="manifest identity mismatch"):
        finalize_query_state_rank_checkpoint(mixed, control=control)


def test_rank_checkpoint_restore_rejects_dtype_before_mutating_any_parameter() -> None:
    root = nn.Module()
    root.language = nn.Linear(1, 1, bias=False)
    root.objective = nn.Module()
    root.objective.projector = DirectSlotProjector()
    optimizer = torch.optim.AdamW(root.parameters(), lr=1e-4)
    with torch.no_grad():
        for parameter in root.parameters():
            parameter.zero_()
    before = {
        name: parameter.detach().clone()
        for name, parameter in root.named_parameters()
    }
    direct = root.objective.projector.linear.weight
    state = QueryStateRankState(
        identity=_resume_identity(world_size=1),
        model={
            "language.weight": torch.ones_like(root.language.weight),
            "objective.projector.linear.weight": direct.detach()
            .clone()
            .to(dtype=torch.bfloat16),
        },
        optimizer=optimizer.state_dict(),
        scheduler={"last_epoch": 0},
        rng={
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        },
    )

    with pytest.raises(ValueError, match="rank checkpoint tensor dtype mismatch"):
        restore_query_state_rank_state(root, optimizer, state)
    for name, parameter in root.named_parameters():
        torch.testing.assert_close(parameter, before[name])


def test_distributed_checkpoint_driver_restores_model_optimizer_rng_and_cursors(
    tmp_path: Path,
) -> None:
    root = nn.Module()
    root.objective = nn.Module()
    root.objective.projector = DirectSlotProjector()
    optimizer = torch.optim.AdamW(root.parameters(), lr=1e-4)
    root.objective.projector(torch.ones(1, 16, 2048)).sum().backward()
    optimizer.step()
    expected = root.objective.projector.linear.weight.detach().clone()
    control = QueryStateDistributedControl(
        identity=_resume_identity(world_size=1),
        global_step=3,
        data_cursor={"schedule_identity": "a" * 64, "consumed_rank_rows": 2},
        metric_cursor={"count/lm_ce": 9},
    )
    checkpoint = tmp_path / "driver-checkpoint"
    save_query_state_distributed_checkpoint(
        checkpoint,
        root=root,
        optimizer=optimizer,
        scheduler_state={"last_epoch": 3},
        control=control,
        rank=0,
    )
    with torch.no_grad():
        root.objective.projector.linear.weight.zero_()
    optimizer.state.clear()
    restored, scheduler = restore_query_state_distributed_checkpoint(
        checkpoint,
        root=root,
        optimizer=optimizer,
        expected_identity=control.identity,
        rank=0,
    )
    torch.testing.assert_close(root.objective.projector.linear.weight, expected)
    assert optimizer.state
    assert restored.metric_cursor == {"count/lm_ce": 9}
    assert scheduler == {"last_epoch": 3}


def test_forensic_checkpoint_is_debug_loadable_but_rejected_for_training_resume(
    tmp_path: Path,
) -> None:
    root = nn.Module()
    root.objective = nn.Module()
    root.objective.projector = DirectSlotProjector()
    optimizer = torch.optim.AdamW(root.parameters(), lr=1e-4)
    root.objective.projector(torch.ones(1, 16, 2048)).sum().backward()
    optimizer.step()
    expected = root.objective.projector.linear.weight.detach().clone()
    control = QueryStateDistributedControl(
        identity=_resume_identity(world_size=1, experiment_mode="formal"),
        global_step=321,
        data_cursor={"next_update": 322},
        metric_cursor={"validation": 321},
        forensic_only=True,
    )
    run_root = tmp_path / "run"
    forensic = run_root / "forensics" / "unsafe_update_00000321"
    save_query_state_distributed_checkpoint(
        forensic,
        root=root,
        optimizer=optimizer,
        scheduler_state={"last_epoch": 321},
        control=control,
        rank=0,
    )
    failure_path = run_root / "durable" / "failures" / "unsafe_00000000_00000321.json"
    failure_path.parent.mkdir(parents=True)
    control_sha = hashlib.sha256((forensic / "control.json").read_bytes()).hexdigest()
    failure_path.write_text(
        json.dumps(
            {
                "schema": "nimloth_sft1_query_state_segment_v1",
                "run_identity": control.identity.run_identity,
                "mode": "formal",
                "end_update": 321,
                "forensic_checkpoint": {
                    "path": str(forensic.resolve()),
                    "control_sha256": control_sha,
                    "forensic_only": True,
                    "resumable": False,
                    "authoritative": False,
                },
                "resumable": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with torch.no_grad():
        root.objective.projector.linear.weight.zero_()
    optimizer.state.clear()
    with pytest.raises(ValueError, match="forensic checkpoint cannot be used.*resume"):
        restore_query_state_distributed_checkpoint(
            forensic,
            root=root,
            optimizer=optimizer,
            expected_identity=control.identity,
            rank=0,
        )
    assert not optimizer.state
    restored = load_query_state_forensic_model_for_debug(
        forensic,
        root=root,
        rank=0,
        expected_identity=control.identity,
        failure_manifest_path=failure_path,
    )
    torch.testing.assert_close(root.objective.projector.linear.weight, expected)
    assert not optimizer.state
    assert restored.global_step == 321
    assert restored.forensic_only is True
    wrong_run_failure = run_root / "durable" / "failures" / "wrong-run.json"
    wrong = json.loads(failure_path.read_text())
    wrong["run_identity"] = "f" * 64
    wrong_run_failure.write_text(json.dumps(wrong) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forensic checkpoint provenance"):
        load_query_state_forensic_model_for_debug(
            forensic,
            root=root,
            rank=0,
            expected_identity=control.identity,
            failure_manifest_path=wrong_run_failure,
        )


def _actor_exporter(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"full-qwen-state")


def _processor_exporter(path: Path) -> None:
    path.mkdir()
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


def test_deployable_bundle_separates_qwen_processor_and_direct_state(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "deployable"
    export_query_state_deployable_bundle(
        destination,
        actor_exporter=_actor_exporter,
        processor_exporter=_processor_exporter,
        projector=DirectSlotProjector(),
        source_identity=_resume_identity(),
        metadata={"role": "local_boundary_test_not_model_quality_evidence"},
    )
    assert {path.name for path in destination.iterdir()} == {
        "actor",
        "processor",
        "direct_state.pt",
        "bundle.json",
    }
    bundle = json.loads((destination / "bundle.json").read_text())
    assert bundle["direct_state_schema"] == "nimloth_sft1_query_state_deployable_v1"
    assert bundle["training_schema"] == "nimloth_sft1_query_state_v1"
    assert not any("optimizer" in path.name.lower() for path in destination.rglob("*"))
