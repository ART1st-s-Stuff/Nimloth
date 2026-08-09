from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nimloth.training.rl.planner_verl_driver import (
    PlannerVERLUpdateDriver,
    validate_planner_fsdp_checkpoint,
)


class _Batch:
    def __init__(
        self,
        update_id: str,
        rows: int = 1,
        *,
        diagnostic_only: bool = False,
    ) -> None:
        self.meta_info = {
            "update_id": update_id,
            "behavior_matched": not diagnostic_only,
            "diagnostic_only": diagnostic_only,
        }
        self.rows = rows

    def __len__(self) -> int:
        return self.rows


class _Collector:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.committed = False

    def begin_consumption(self, *, output_dir: Path, global_step: int) -> str:
        self.events.append(("claim", output_dir, global_step))
        return "consumption-1"

    def commit_consumption(
        self,
        consumption_id: str,
        *,
        checkpoint_path: Path,
        global_step: int,
    ) -> None:
        self.events.append(
            ("commit", consumption_id, checkpoint_path, global_step)
        )
        self.committed = True


class _Workers:
    world_size = 2

    def __init__(self, events: list[object], *, fail_save: bool = False) -> None:
        self.events = events
        self.fail_save = fail_save

    def begin_planner_update(self, update_id: str):  # type: ignore[no-untyped-def]
        self.events.append(("begin", update_id))
        return [True, True]

    def backward_planner_micro_batch(self, rank_batches):  # type: ignore[no-untyped-def]
        self.events.append(("backward", tuple(rank_batches)))
        return [{"loss": 1.0}, {"loss": 2.0}]

    def finish_planner_update(self, update_id: str):  # type: ignore[no-untyped-def]
        self.events.append(("finish", update_id))
        return [{"loss": 1.0}, {"loss": 2.0}]

    def save_planner_checkpoint(
        self,
        path: str,
        update_id: str,
        global_step: int,
    ):  # type: ignore[no-untyped-def]
        checkpoint = Path(path)
        self.events.append(("save", checkpoint, update_id, global_step))
        if self.fail_save:
            raise RuntimeError("rank checkpoint failed")
        checkpoint.mkdir(parents=True)
        for rank in range(self.world_size):
            for kind in ("model", "optim", "extra_state"):
                torch.save(
                    {"rank": rank},
                    checkpoint
                    / f"{kind}_world_size_{self.world_size}_rank_{rank}.pt",
                )
        torch.save(
            {
                "checkpoint_schema_version": 1,
                "optimizer_state_layout": "rank_sharded_fsdp",
                "optimizer_world_size": self.world_size,
                "training_world_size": self.world_size,
                "global_step": global_step,
                "update_id": update_id,
                "completed_update_ids": [update_id],
            },
            checkpoint / "rl_state.pt",
        )
        return [True, True]

    def mark_planner_checkpoint_succeeded(
        self,
        update_id: str,
    ):  # type: ignore[no-untyped-def]
        self.events.append(("mark", update_id))
        return [True, True]


def test_validate_planner_fsdp_checkpoint_requires_every_rank(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint"
    path.mkdir()
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "optimizer_state_layout": "rank_sharded_fsdp",
            "optimizer_world_size": 2,
            "training_world_size": 2,
            "global_step": 1,
            "update_id": "update-1",
            "completed_update_ids": ["update-1"],
        },
        path / "rl_state.pt",
    )

    with pytest.raises(FileNotFoundError, match="rank checkpoint shard"):
        validate_planner_fsdp_checkpoint(
            path,
            world_size=2,
            global_step=1,
            update_id="update-1",
        )


def test_driver_rejects_nonbehavior_diagnostic_before_claim(tmp_path: Path) -> None:
    events: list[object] = []
    driver = PlannerVERLUpdateDriver(
        worker_group=_Workers(events),
        collector=_Collector(events),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="reject nonbehavior diagnostics"):
        driver.run_update(
            output_dir=tmp_path,
            current_global_step=0,
            rank_rounds=(
                (
                    _Batch("diagnostic", diagnostic_only=True),
                    _Batch("diagnostic", diagnostic_only=True),
                ),
            ),
        )

    assert events == []


def test_driver_publishes_checkpoint_before_consumption_commit(
    tmp_path: Path,
) -> None:
    events: list[object] = []
    workers = _Workers(events)
    collector = _Collector(events)
    driver = PlannerVERLUpdateDriver(
        worker_group=workers,
        collector=collector,  # type: ignore[arg-type]
    )
    rounds = (
        (_Batch("update-1"), _Batch("update-1")),
        (_Batch("update-1"), _Batch("update-1")),
    )

    result = driver.run_update(
        output_dir=tmp_path,
        current_global_step=0,
        rank_rounds=rounds,
    )

    checkpoint = tmp_path / "global_step_000001"
    assert result.checkpoint_path == checkpoint
    assert result.global_step == 1
    assert result.update_id == "consumption-1"
    assert collector.committed is True
    assert checkpoint.is_dir()
    assert [event[0] for event in events] == [
        "claim",
        "begin",
        "backward",
        "backward",
        "finish",
        "save",
        "commit",
        "mark",
    ]
    assert not tuple(tmp_path.glob(".global_step_000001.tmp-*"))


def test_driver_leaves_claim_uncommitted_after_checkpoint_failure(
    tmp_path: Path,
) -> None:
    events: list[object] = []
    workers = _Workers(events, fail_save=True)
    collector = _Collector(events)
    driver = PlannerVERLUpdateDriver(
        worker_group=workers,
        collector=collector,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="rank checkpoint failed"):
        driver.run_update(
            output_dir=tmp_path,
            current_global_step=0,
            rank_rounds=((_Batch("update-1"), _Batch("update-1")),),
        )

    assert collector.committed is False
    assert [event[0] for event in events] == [
        "claim",
        "begin",
        "backward",
        "finish",
        "save",
    ]


def test_driver_validates_rank_rounds_before_claim(tmp_path: Path) -> None:
    events: list[object] = []
    driver = PlannerVERLUpdateDriver(
        worker_group=_Workers(events),
        collector=_Collector(events),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="one nonempty batch per rank"):
        driver.run_update(
            output_dir=tmp_path,
            current_global_step=0,
            rank_rounds=((_Batch("update-1"),),),
        )

    assert events == []


def test_driver_rejects_unequal_rank_row_counts_before_claim(
    tmp_path: Path,
) -> None:
    events: list[object] = []
    driver = PlannerVERLUpdateDriver(
        worker_group=_Workers(events),
        collector=_Collector(events),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="equal row counts"):
        driver.run_update(
            output_dir=tmp_path,
            current_global_step=0,
            rank_rounds=((_Batch("update-1", 2), _Batch("update-1", 1)),),
        )

    assert events == []
