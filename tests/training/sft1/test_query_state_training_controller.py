from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nimloth.training.sft1.query_state_training_controller import (
    QueryStateTrainingController,
)


_SHA = "a" * 64


def test_controller_claim_is_nonoverwrite_and_publishes_durable_readme_first(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    controller = QueryStateTrainingController(
        run_root=root,
        controller_root=tmp_path / "controller",
        run_identity=_SHA,
        mode="pilot",
    )
    controller.claim(
        resolved_config={"identity": _SHA, "mode": "pilot"},
        command_manifest={"identity": "b" * 64, "argv": ["python"]},
    )
    assert (root / "README.md").is_file()
    assert (root / "resolved_config.json").is_file()
    assert (root / "command_manifest.json").is_file()
    assert not (root / "COMPLETED.json").exists()
    controller.verify_existing_claim()
    with pytest.raises(FileExistsError, match="exists|claimed"):
        QueryStateTrainingController(
            run_root=root,
            controller_root=tmp_path / "other",
            run_identity=_SHA,
            mode="pilot",
        ).claim(resolved_config={}, command_manifest={})


def test_authoritative_pause_receipt_is_nonterminal_and_restartable(
    tmp_path: Path,
) -> None:
    controller = QueryStateTrainingController(
        run_root=tmp_path / "formal",
        controller_root=tmp_path / "controller",
        run_identity=_SHA,
        mode="formal",
    )
    controller.claim(resolved_config={"mode": "formal"}, command_manifest={"argv": []})
    command_manifest_text = '{"schema":"test"}\n'
    process_path = controller.record_process(
        process_identity="c" * 64,
        details={
            "config_identity": "d" * 64,
            "command_identity": hashlib.sha256(
                command_manifest_text.encode()
            ).hexdigest(),
            "resume_mode": "fresh",
            "approved_pause_update": 1605,
            "resume_checkpoint": None,
            "resolved_config": {"schema": "test"},
            "command_manifest_text": command_manifest_text,
        },
    )
    process = json.loads(process_path.read_text())
    assert process["approved_pause_update"] == 1605
    assert process["run_identity"] == _SHA
    details = {
        "checkpoint": "/run/checkpoints/update_00001605",
        "checkpoint_control_hash": "b" * 64,
        "terminal_primary": False,
    }
    path = controller.record_pause(update=1605, details=details)
    assert path.name == "pause_update_00001605.json"
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "paused"
    assert receipt["terminal_primary"] is False
    assert not (controller.run_root / "COMPLETED.json").exists()
    controller.verify_existing_claim()
    with pytest.raises(FileExistsError, match="immutable"):
        controller.record_pause(update=1605, details=details)


def test_failure_preemption_and_completion_are_distinct_terminal_states(tmp_path: Path) -> None:
    controller = QueryStateTrainingController(
        run_root=tmp_path / "formal",
        controller_root=tmp_path / "controller",
        run_identity=_SHA,
        mode="formal",
    )
    controller.claim(resolved_config={"mode": "formal"}, command_manifest={"argv": []})
    path = controller.record_terminal(status="preempted", details={"resume_update": 8})
    assert path.name == "PREEMPTED.json"
    with pytest.raises(RuntimeError, match="terminal"):
        controller.verify_existing_claim()
    assert json.loads(path.read_text())["status"] == "preempted"
    with pytest.raises(RuntimeError, match="terminal status"):
        controller.record_terminal(status="completed", details={})


def test_controller_never_auto_extends_pilot_to_formal_or_sft2(tmp_path: Path) -> None:
    controller = QueryStateTrainingController(
        run_root=tmp_path / "pilot",
        controller_root=tmp_path / "controller",
        run_identity=_SHA,
        mode="pilot",
    )
    controller.claim(resolved_config={"mode": "pilot"}, command_manifest={"argv": []})
    with pytest.raises(ValueError, match="automatic.*formal|next stage"):
        controller.record_terminal(status="completed", details={"next_stage": "formal"})
    with pytest.raises(ValueError, match="automatic.*SFT2|next stage"):
        controller.record_terminal(status="completed", details={"next_stage": "sft2a"})
