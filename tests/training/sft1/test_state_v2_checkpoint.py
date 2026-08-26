from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nimloth.training.sft1.checkpoint import (
    SFT1V2ControlState,
    SFT1V2RankState,
    capture_sft1_v2_rank_state,
    export_sft1_v2_deployable,
    finalize_sft1_v2_checkpoint,
    load_sft1_v2_rank_state,
    restore_sft1_v2_rank_state,
    save_sft1_v2_rank_state,
)
from nimloth.training.sft1.config import STATE_INTERFACE_OBJECTIVE_VERSION


MANIFEST_ID = "a" * 64
CONFIG_ID = "b" * 64
RUN_ID = "c" * 64
SOURCE_COMMIT = "d" * 40


def _rank_state(rank: int) -> SFT1V2RankState:
    return SFT1V2RankState(
        model={
            "backbone.nimloth_query_embedding_adapter.delta": torch.tensor([rank + 1.0]),
            "objective.projector.net.0.weight": torch.tensor([rank + 2.0]),
            "objective.visual_readout.weight": torch.tensor([rank + 3.0]),
        },
        optimizer={"step": rank + 4},
        scheduler={"last_epoch": rank + 5},
        rng={"cpu": torch.tensor([rank], dtype=torch.uint8)},
    )


def _control(world_size: int = 2) -> SFT1V2ControlState:
    return SFT1V2ControlState(
        global_step=7,
        data_cursor={"epoch": 1, "shard": 3, "row": 11},
        manifest_identity=MANIFEST_ID,
        config_identity=CONFIG_ID,
        objective_version=STATE_INTERFACE_OBJECTIVE_VERSION,
        world_size=world_size,
        run_identity=RUN_ID,
        source_commit=SOURCE_COMMIT,
    )


def test_local_trainable_optimizer_and_rng_state_restore_exactly() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    model[1].requires_grad_(False)
    optimizer = torch.optim.AdamW(model[0].parameters(), lr=1e-3)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    captured = capture_sft1_v2_rank_state(model, optimizer)
    expected = {name: value.clone() for name, value in captured.model.items()}
    with torch.no_grad():
        for parameter in model[0].parameters():
            parameter.add_(10)
    optimizer.state.clear()
    restore_sft1_v2_rank_state(model, optimizer, captured)
    restored = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert set(restored) == set(expected)
    assert all(torch.equal(restored[name], expected[name]) for name in expected)
    assert optimizer.state


def test_resume_checkpoint_round_trips_full_training_and_control_state(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    for rank in range(2):
        save_sft1_v2_rank_state(
            checkpoint,
            rank=rank,
            world_size=2,
            state=_rank_state(rank),
        )
    finalize_sft1_v2_checkpoint(checkpoint, control=_control())

    restored, control = load_sft1_v2_rank_state(
        checkpoint,
        rank=1,
        expected_world_size=2,
        expected_manifest_identity=MANIFEST_ID,
        expected_config_identity=CONFIG_ID,
        expected_run_identity=RUN_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert control.global_step == 7
    assert control.data_cursor == {"epoch": 1, "shard": 3, "row": 11}
    assert restored.optimizer == {"step": 5}
    assert restored.scheduler == {"last_epoch": 6}
    torch.testing.assert_close(
        restored.model["objective.visual_readout.weight"],
        torch.tensor([4.0]),
    )
    assert (checkpoint / "COMPLETED").is_file()
    with pytest.raises(FileExistsError, match="immutable"):
        save_sft1_v2_rank_state(
            checkpoint,
            rank=0,
            world_size=2,
            state=_rank_state(0),
        )


def test_resume_checkpoint_rejects_partial_or_mismatched_identity(tmp_path) -> None:
    partial = tmp_path / "partial"
    save_sft1_v2_rank_state(
        partial,
        rank=0,
        world_size=2,
        state=_rank_state(0),
    )
    with pytest.raises(FileNotFoundError, match="rank shards"):
        finalize_sft1_v2_checkpoint(partial, control=_control())

    complete = tmp_path / "complete"
    save_sft1_v2_rank_state(
        complete,
        rank=0,
        world_size=1,
        state=_rank_state(0),
    )
    finalize_sft1_v2_checkpoint(complete, control=_control(world_size=1))
    with pytest.raises(ValueError, match="manifest identity"):
        load_sft1_v2_rank_state(
            complete,
            rank=0,
            expected_world_size=1,
            expected_manifest_identity="0" * 64,
            expected_config_identity=CONFIG_ID,
            expected_run_identity=RUN_ID,
            expected_source_commit=SOURCE_COMMIT,
        )


def _actor_exporter(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text('{"query_mode":"inject"}\n')
    (path / "model.safetensors").write_bytes(b"actor-query")


def _processor_exporter(path: Path) -> None:
    path.mkdir()
    (path / "tokenizer_config.json").write_text('{"action_count":8}\n')


def test_deployable_export_contains_only_actor_processor_projector_and_metadata(
    tmp_path,
) -> None:
    output = tmp_path / "deployable"
    export_sft1_v2_deployable(
        output,
        actor_exporter=_actor_exporter,
        processor_exporter=_processor_exporter,
        projector_state={"net.0.weight": torch.ones(2, 2)},
        state_metadata={
            "manifest_identity": MANIFEST_ID,
            "query_mode": "inject",
            "action_token_ids": list(range(8)),
        },
    )

    assert {path.name for path in output.iterdir()} == {
        "actor",
        "processor",
        "slot_projector.pt",
        "state_interface_config.json",
    }
    metadata = json.loads((output / "state_interface_config.json").read_text())
    assert metadata["grid_tokens"] == 16
    assert metadata["state_dim"] == 1024
    all_names = {path.name.lower() for path in output.rglob("*")}
    assert not any(
        term in name
        for name in all_names
        for term in ("training_head", "optimizer", "world_model", "value_head")
    )

    with pytest.raises(ValueError, match="training-only keys"):
        export_sft1_v2_deployable(
            tmp_path / "bad-export",
            actor_exporter=_actor_exporter,
            processor_exporter=_processor_exporter,
            projector_state={"state_policy_head.weight": torch.ones(1)},
            state_metadata={"manifest_identity": MANIFEST_ID},
        )
