from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from nimloth.training.sft1.query_state_checkpoint import (
    QueryStateDistributedControl,
    QueryStateRankState,
    QueryStateResumeIdentity,
    finalize_query_state_rank_checkpoint,
    save_query_state_rank_state,
)
from nimloth.training.sft1.query_state_export import (
    QUERY_STATE_EXPORT_CONFIG_SCHEMA,
    materialize_gated_query_state_export,
    parse_query_state_export_contract,
    validate_full_state_for_export,
    verify_query_state_export_gate,
    verify_query_state_state_interface,
)
from nimloth.training.sft1.query_state_training_runtime import current_process_identity
from nimloth.wm.grid import DirectSlotProjector


_SHA = "a" * 64
_COMMIT = "b" * 40


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(tmp_path: Path, *, passed: bool = True) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    identity = QueryStateResumeIdentity(
        source_commit=_COMMIT,
        source_manifest_identity="5" * 64,
        config_identity="e" * 64,
        run_identity="6" * 64,
        world_size=2,
        experiment_mode="formal",
    )
    for rank in range(2):
        save_query_state_rank_state(
            checkpoint,
            rank=rank,
            world_size=2,
            state=QueryStateRankState(
                identity=identity,
                model={"objective.projector.linear.weight": torch.tensor([float(rank)])},
                optimizer={"step": rank},
                scheduler={"last_epoch": 16},
                rng={
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(),
                },
            ),
        )
    finalize_query_state_rank_checkpoint(
        checkpoint,
        control=QueryStateDistributedControl(
            identity=identity,
            global_step=16,
            data_cursor={"next_update": 17},
            metric_cursor={"validation": 16, "log": 16, "wandb": 16},
            terminal_primary=True,
        ),
    )
    control = json.loads((checkpoint / "control.json").read_text(encoding="utf-8"))
    receipt = tmp_path / "human-gate.json"
    receipt.write_text(json.dumps({
        "decision": "pass" if passed else "fail",
        "checkpoint_identity": control["checkpoint_identity"],
        "checkpoint_control_hash": control["control_hash"],
        "config_identity": "e" * 64,
        "source_commit": _COMMIT,
        "terminal_primary": True,
        "automatic_sft2_authorization": False,
    }), encoding="utf-8")
    return checkpoint, receipt


def _raw(tmp_path: Path, *, passed: bool = True) -> dict:
    checkpoint, receipt = _files(tmp_path, passed=passed)
    control = json.loads((checkpoint / "control.json").read_text(encoding="utf-8"))
    return {
        "schema": QUERY_STATE_EXPORT_CONFIG_SCHEMA,
        "source": {
            "commit": _COMMIT,
            "config_identity": "e" * 64,
            "source_manifest_identity": "5" * 64,
            "run_identity": "6" * 64,
            "world_size": 2,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "identity": control["checkpoint_identity"],
            "control_sha256": _hash(checkpoint / "control.json"),
            "control_identity": control["control_hash"],
            "terminal_update": 16,
            "terminal_primary": True,
        },
        "human_gate": {
            "receipt_path": str(receipt),
            "receipt_sha256": _hash(receipt),
            "decision": "pass" if passed else "fail",
        },
        "export": {
            "approval_id": "export-approval-1",
            "approval_sha256": "f" * 64,
            "command": ["python", "query_state_export.py", "materialize"],
            "command_identity": "1" * 64,
            "output_path": str(tmp_path / "bundle"),
            "overwrite": False,
        },
        "model": {
            "processor_identity": "2" * 64,
            "tokenizer_identity": "3" * 64,
            "template_identity": "4" * 64,
            "direct_head_shape": [1024, 2048],
            "state_interface": [16, 1024],
        },
        "boundary": {
            "official_fsdp_full_state": True,
            "include_optimizer": False,
            "include_scheduler": False,
            "include_rng": False,
            "automatic_formal_export": False,
            "automatic_sft2_authorization": False,
        },
    }


def test_finalized_terminal_rank_checkpoint_is_consumed_by_export_gate(tmp_path: Path) -> None:
    contract = parse_query_state_export_contract(_raw(tmp_path))
    evidence = verify_query_state_export_gate(contract)
    assert evidence.terminal_update == 16
    assert evidence.human_decision == "pass"

    missing = _raw(tmp_path / "missing")
    del missing["human_gate"]["receipt_sha256"]
    with pytest.raises(ValueError, match="missing.*human_gate.receipt_sha256"):
        parse_query_state_export_contract(missing)


def test_pre_gate_failure_happens_before_any_full_state_or_export_callback(tmp_path: Path) -> None:
    contract = parse_query_state_export_contract(_raw(tmp_path, passed=False))
    called = False

    def actor_exporter(*_args, **_kwargs):
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="human.*pass"):
        materialize_gated_query_state_export(
            contract,
            fsdp_root=object(),
            actor_exporter=actor_exporter,
            processor_exporter=lambda _path: None,
            expected_actor_state_keys=("model.weight",),
            invoked_from_formal_job=False,
        )
    assert called is False
    assert not Path(contract.output_path).exists()


def test_formal_job_auto_export_existing_output_and_identity_mismatch_fail_closed(tmp_path: Path) -> None:
    contract = parse_query_state_export_contract(_raw(tmp_path))
    with pytest.raises(ValueError, match="formal.*automatic export"):
        materialize_gated_query_state_export(
            contract,
            fsdp_root=object(),
            actor_exporter=lambda *_args: None,
            processor_exporter=lambda *_args: None,
            expected_actor_state_keys=("model.weight",),
            invoked_from_formal_job=True,
        )
    Path(contract.output_path).mkdir()
    with pytest.raises(FileExistsError, match="output"):
        verify_query_state_export_gate(contract)


def test_full_state_validation_rejects_rank_shards_missing_head_and_training_payloads() -> None:
    actor = {"weight": torch.ones(2, 2)}
    direct = torch.ones(1024, 2048)
    validated_actor, validated_direct = validate_full_state_for_export(
        {
            "backbone.language_model.weight": actor["weight"],
            "objective.projector.linear.weight": direct,
        },
        expected_actor_state_keys=("weight",),
    )
    assert set(validated_actor) == {"weight"}
    assert validated_direct.shape == (1024, 2048)

    for bad, message in (
        ({"backbone.language_model.weight": actor["weight"]}, "direct head"),
        ({"backbone.language_model.weight": actor["weight"], "objective.projector.linear.weight": torch.ones(1, 2048)}, "1024"),
        ({"backbone.language_model.weight": actor["weight"], "objective.projector.linear.weight": direct, "optimizer.state": torch.ones(1)}, "training-only"),
        ({"_local_shard.backbone.language_model.weight": actor["weight"], "objective.projector.linear.weight": direct}, "rank-local|actor"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_full_state_for_export(bad, expected_actor_state_keys=("weight",))


def test_fresh_load_state_interface_is_exact_b16x1024_and_has_no_optimizer() -> None:
    projector = DirectSlotProjector()
    hidden = torch.randn(2, 16, 2048)
    identities = {
        "loaded_processor_identity": "2" * 64,
        "loaded_tokenizer_identity": "3" * 64,
        "loaded_template_identity": "4" * 64,
        "expected_processor_identity": "2" * 64,
        "expected_tokenizer_identity": "3" * 64,
        "expected_template_identity": "4" * 64,
        "materialization_process_identity": "9" * 64,
        "verifier_process_identity": current_process_identity(),
    }
    state = verify_query_state_state_interface(
        projector,
        query_hidden=hidden,
        bundle_files=("actor/config.json", "processor/tokenizer.json", "direct_state.pt", "bundle.json"),
        **identities,
    )
    assert state.shape == (2, 16, 1024)
    with pytest.raises(ValueError, match="training-only"):
        verify_query_state_state_interface(
            projector,
            query_hidden=hidden,
            bundle_files=("actor/config.json", "optimizer.pt"),
            **identities,
        )
    with pytest.raises(ValueError, match="fresh process"):
        verify_query_state_state_interface(
            projector,
            query_hidden=hidden,
            bundle_files=("actor/config.json", "processor/tokenizer.json", "direct_state.pt", "bundle.json"),
            **{
                **identities,
                "materialization_process_identity": current_process_identity(),
            },
        )
    with pytest.raises(ValueError, match="processor|tokenizer|template"):
        verify_query_state_state_interface(
            projector,
            query_hidden=hidden,
            bundle_files=("actor/config.json", "processor/tokenizer.json", "direct_state.pt", "bundle.json"),
            **{
                **identities,
                "loaded_tokenizer_identity": "8" * 64,
            },
        )
