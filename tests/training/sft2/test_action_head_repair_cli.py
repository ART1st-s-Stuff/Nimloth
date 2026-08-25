from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nimloth.training.sft2.action_head_repair_cli import (
    _HIDDEN_SCHEMA,
    _atomic_csv,
    _atomic_torch_save,
    _combine_hidden_shards,
    _parse_args,
    _per_action_metrics,
)


def _identity(index: int, action: int) -> dict[str, object]:
    return {
        "sample_index": index,
        "record_id": f"trajectory-{index}",
        "step_index": index,
        "action_index": action,
    }


def test_combine_hidden_shards_restores_selection_order(tmp_path: Path) -> None:
    output = tmp_path / "run"
    selection = [_identity(0, 0), _identity(1, 1), _identity(2, 0)]
    shards = (
        ([selection[0], selection[2]], [[1.0, 2.0], [5.0, 6.0]], [0, 0]),
        ([selection[1]], [[3.0, 4.0]], [1]),
    )
    for rank, (identities, hidden, targets) in enumerate(shards):
        _atomic_torch_save(
            output / "hidden" / f"train_rank_{rank:03d}.pt",
            {
                "schema": _HIDDEN_SCHEMA,
                "split": "train",
                "rank": rank,
                "world_size": 2,
                "identities": identities,
                "hidden": torch.tensor(hidden),
                "targets": torch.tensor(targets),
            },
        )

    hidden, targets = _combine_hidden_shards(
        output,
        split="train",
        world_size=2,
        expected_selection=selection,
    )

    assert hidden.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert targets.tolist() == [0, 1, 0]


def test_atomic_training_log_has_one_row_per_epoch(tmp_path: Path) -> None:
    path = tmp_path / "train_step_log.csv"
    _atomic_csv(
        path,
        (
            {"epoch": 1, "training_nll": 2.0, "validation_nll": 2.1},
            {"epoch": 2, "training_nll": 1.8, "validation_nll": 1.9},
        ),
    )

    assert path.read_text().splitlines() == [
        "epoch,training_nll,validation_nll",
        "1,2.0,2.1",
        "2,1.8,1.9",
    ]


def test_per_action_metrics_reports_each_action_without_reweighting() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
    targets = torch.tensor([0, 1, 0])

    metrics = _per_action_metrics(logits, targets)

    assert metrics["0"]["count"] == 2
    assert metrics["1"]["count"] == 1
    assert metrics["0"]["accuracy"] == 1.0
    assert metrics["1"]["accuracy"] == 1.0


def test_cli_has_no_joint_training_numeric_defaults() -> None:
    with pytest.raises(SystemExit):
        _parse_args([])

    source = Path(
        "src/nimloth/training/sft2/action_head_repair_cli.py"
    ).read_text(encoding="utf-8")
    for flag in (
        "train-examples-per-action",
        "validation-examples-per-action",
        "fit-learning-rate",
        "fit-weight-decay",
        "fit-max-epochs",
        "fit-early-stopping-patience",
        "minimum-validation-nll-improvement",
        "minimum-bf16-median-spread",
        "resume-mode",
        "git-commit",
        "experiment-purpose",
    ):
        assert f'"{flag}"' in source
    assert "required=True" in source
