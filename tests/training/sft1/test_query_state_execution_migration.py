from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from nimloth.training.sft1.query_state_training_backend import (
    _process_execution_provenance,
)
from nimloth.training.sft1.query_state_training_config import (
    parse_query_state_training_config,
    query_state_training_lineage_identity,
    query_state_training_run_identity,
)
from nimloth.training.sft1.query_state_training_preflight import (
    _authenticate_execution_migration,
    _authenticated_exact_restart_update,
)
from nimloth.training.sft1.query_state_training_migration import (
    parse_legacy_prior_process_config,
    query_state_execution_provenance,
    validate_query_state_execution_migration_contract,
)
from tests.training.sft1.test_query_state_training_config import _raw


def _disabled_migration() -> dict:
    return {
        "enabled": False,
        "anchor_run_identity": "disabled",
        "anchor_source_commit": "disabled",
        "anchor_source_manifest_path": "disabled",
        "anchor_source_manifest_identity": "disabled",
        "anchor_partition": "disabled",
        "prior_process_path": "disabled",
        "prior_process_sha256": "disabled",
        "anchor_checkpoint_path": "disabled",
        "anchor_control_sha256": "disabled",
        "anchor_index_path": "disabled",
        "anchor_index_sha256": "disabled",
        "execution_source_commit": "disabled",
        "execution_source_manifest_path": "disabled",
        "execution_source_manifest_identity": "disabled",
        "execution_partition": "disabled",
        "approval_sha256": "disabled",
    }


def _migration_pair() -> tuple[dict, dict]:
    prior = _raw(mode="visual_only_forensic_fork", resume_mode="exact_restart")
    prior["resources"]["partition"] = "preempt"
    prior["initialization"]["resume_checkpoint"] = (
        prior["output"]["run_root"] + "/checkpoints/update_00004815"
    )
    prior["schema"] = "nimloth_sft1_query_state_training_v4"
    prior.pop("execution_migration")
    anchor = parse_legacy_prior_process_config(prior)

    current = deepcopy(prior)
    current["source"]["commit"] = "e" * 40
    current["source"]["source_manifest_path"] = "/manifests/migration-source.json"
    current["source"]["source_manifest_identity"] = "f" * 64
    current["resources"]["partition"] = "normal"
    current["tracking"]["resume"] = "must"
    current["execution_migration"] = {
        "enabled": True,
        "anchor_run_identity": query_state_training_run_identity(anchor),
        "anchor_source_commit": prior["source"]["commit"],
        "anchor_source_manifest_path": prior["source"]["source_manifest_path"],
        "anchor_source_manifest_identity": prior["source"]["source_manifest_identity"],
        "anchor_partition": "preempt",
        "prior_process_path": "/controllers/visual_only_forensic_fork/processes/process_" + "1" * 64 + ".json",
        "prior_process_sha256": "2" * 64,
        "anchor_checkpoint_path": current["initialization"]["resume_checkpoint"],
        "anchor_control_sha256": "3" * 64,
        "anchor_index_path": current["output"]["run_root"] + "/durable/authoritative_index.json",
        "anchor_index_sha256": "4" * 64,
        "execution_source_commit": current["source"]["commit"],
        "execution_source_manifest_path": current["source"]["source_manifest_path"],
        "execution_source_manifest_identity": current["source"]["source_manifest_identity"],
        "execution_partition": "normal",
        "approval_sha256": current["authorization"]["approval_sha256"],
    }
    current["artifacts"]["file_sha256"].update({
        current["source"]["source_manifest_path"]: current["source"]["source_manifest_identity"],
        current["execution_migration"]["prior_process_path"]: "2" * 64,
        current["execution_migration"]["anchor_checkpoint_path"] + "/control.json": "3" * 64,
        current["execution_migration"]["anchor_index_path"]: "4" * 64,
    })
    return prior, current


def test_job542431_process_fixture_authenticates_documented_anchor_identity() -> None:
    fixture = Path(__file__).parent / "fixtures" / "job542431_process.json"
    process = json.loads(fixture.read_text(encoding="utf-8"))
    prior_raw = process["resolved_config"]
    prior = parse_legacy_prior_process_config(prior_raw)
    assert process["resume_mode"] == "fresh"
    assert process["config_identity"] == (
        "58ef701a49899b1f9c3fe48d70816c3c7b2ad4fc23da3c1665643cf68829ddb0"
    )
    assert prior.identity == process["config_identity"]
    assert prior.source["commit"] == "f65ed859f9377584af7e1bb450e7e9de99e02b95"
    assert prior.resources["partition"] == "preempt"
    assert query_state_training_run_identity(prior) == (
        "ca1003f306f0337a33dee11790ce983788c0522f5e4022776a1655a9aeb41487"
    ) == process["run_identity"]

    current = deepcopy(prior_raw)
    current["source"]["commit"] = "e" * 40
    current["source"]["source_manifest_path"] = "/contracts/restart43-source.json"
    current["source"]["source_manifest_identity"] = "f" * 64
    current["resources"]["partition"] = "normal"
    current["initialization"]["resume_mode"] = "exact_restart"
    current["initialization"]["resume_checkpoint"] = (
        current["output"]["run_root"] + "/checkpoints/update_00004815"
    )
    current["tracking"]["resume"] = "must"
    current["execution_migration"] = {
        "enabled": True,
        "anchor_run_identity": process["run_identity"],
        "anchor_source_commit": prior_raw["source"]["commit"],
        "anchor_source_manifest_path": prior_raw["source"]["source_manifest_path"],
        "anchor_source_manifest_identity": prior_raw["source"]["source_manifest_identity"],
        "anchor_partition": "preempt",
        "prior_process_path": str(fixture.resolve()),
        "prior_process_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "anchor_checkpoint_path": current["initialization"]["resume_checkpoint"],
        "anchor_control_sha256": "3" * 64,
        "anchor_index_path": "/contracts/job542431-update4815-index.json",
        "anchor_index_sha256": "4" * 64,
        "execution_source_commit": current["source"]["commit"],
        "execution_source_manifest_path": current["source"]["source_manifest_path"],
        "execution_source_manifest_identity": current["source"]["source_manifest_identity"],
        "execution_partition": "normal",
        "approval_sha256": current["authorization"]["approval_sha256"],
    }
    current["artifacts"]["file_sha256"].update({
        current["source"]["source_manifest_path"]: "f" * 64,
        str(fixture.resolve()): current["execution_migration"]["prior_process_sha256"],
        current["initialization"]["resume_checkpoint"] + "/control.json": "3" * 64,
        current["execution_migration"]["anchor_index_path"]: "4" * 64,
    })
    migrated = parse_query_state_training_config(current)
    validate_query_state_execution_migration_contract(migrated, prior_raw)
    assert query_state_training_lineage_identity(migrated) == process["run_identity"]


def test_execution_migration_is_explicit_and_native_identity_stays_strict() -> None:
    native = _raw(mode="visual_only_forensic_fork", resume_mode="exact_restart")
    native["initialization"]["resume_checkpoint"] = (
        native["output"]["run_root"] + "/checkpoints/update_00004815"
    )
    native["execution_migration"] = _disabled_migration()
    preempt = deepcopy(native)
    preempt["resources"]["partition"] = "preempt"
    normal_config = parse_query_state_training_config(native)
    preempt_config = parse_query_state_training_config(preempt)
    assert query_state_training_run_identity(normal_config) != query_state_training_run_identity(
        preempt_config
    )
    assert query_state_training_lineage_identity(normal_config) == query_state_training_run_identity(
        normal_config
    )

    missing = deepcopy(native)
    del missing["execution_migration"]
    with pytest.raises(ValueError, match="execution_migration"):
        parse_query_state_training_config(missing)


def test_only_visual_exact_restart_preempt_to_normal_can_enable_migration() -> None:
    prior, current = _migration_pair()
    config = parse_query_state_training_config(current)
    validate_query_state_execution_migration_contract(config, prior)
    assert query_state_training_lineage_identity(config) == config.execution_migration[
        "anchor_run_identity"
    ]
    assert query_state_training_run_identity(config) != query_state_training_lineage_identity(
        config
    )

    for section, field, value in (
        ("objective", "state_weight", 3.0),
        ("optimizer", "language_learning_rate", 2e-6),
        ("resources", "world_size", 4),
        ("environment", "nccl_socket_ifname", "other0"),
        ("tracking", "run_id", "different"),
    ):
        drifted = deepcopy(current)
        drifted[section][field] = value
        with pytest.raises(ValueError):
            validate_query_state_execution_migration_contract(
                parse_query_state_training_config(drifted), prior
            )

    for mode, resume_mode in (("formal", "exact_restart"), ("visual_only_forensic_fork", "fresh")):
        invalid = deepcopy(current)
        invalid["mode"] = mode
        invalid["initialization"]["resume_mode"] = resume_mode
        if resume_mode == "fresh":
            invalid["initialization"]["resume_checkpoint"] = "none"
            invalid["tracking"]["resume"] = "never"
        with pytest.raises(ValueError):
            parse_query_state_training_config(invalid)


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_preflight_authenticates_prior_process_control_index_and_actual_partition(
    tmp_path: Path,
) -> None:
    prior, current = _migration_pair()
    run_root = tmp_path / "run"
    controller_root = tmp_path / "controller"
    checkpoint = run_root / "checkpoints" / "update_00004815"
    checkpoint.mkdir(parents=True)
    (run_root / "durable").mkdir()
    process_path = controller_root / "processes" / ("process_" + "1" * 64 + ".json")
    process_path.parent.mkdir(parents=True)

    for raw in (prior, current):
        raw["output"]["run_root"] = str(run_root)
        raw["output"]["controller_root"] = str(controller_root)
    prior["initialization"]["resume_checkpoint"] = str(checkpoint)
    prior_config = parse_legacy_prior_process_config(prior)
    anchor_run = query_state_training_run_identity(prior_config)
    process_path.write_text(json.dumps({
        "run_identity": anchor_run,
        "mode": "visual_only_forensic_fork",
        "config_identity": prior_config.identity,
        "resolved_config": prior,
    }) + "\n", encoding="utf-8")

    identity = {
        "config_identity": anchor_run,
        "source_commit": prior["source"]["commit"],
        "source_manifest_identity": prior["source"]["source_manifest_identity"],
        "world_size": 8,
        "experiment_mode": "visual_only_forensic_fork",
        "run_identity": anchor_run,
    }
    base = {
        "identity": identity,
        "config_identity": anchor_run,
        "source_commit": prior["source"]["commit"],
        "global_step": 4815,
        "data_cursor": {"next_update": 4816},
    }
    checkpoint_identity = _canonical_hash(base)
    with_checkpoint = {**base, "checkpoint_identity": checkpoint_identity}
    control = {**with_checkpoint, "control_hash": _canonical_hash(with_checkpoint)}
    control_path = checkpoint / "control.json"
    control_path.write_text(json.dumps(control, sort_keys=True) + "\n", encoding="utf-8")
    control_sha = hashlib.sha256(control_path.read_bytes()).hexdigest()
    (checkpoint / "COMPLETED").write_text(
        f"control_sha256={control_sha}\n", encoding="utf-8"
    )
    index_path = run_root / "durable" / "authoritative_index.json"
    anchor_index_path = tmp_path / "immutable_update4815_index.json"
    anchor_index = {
        "mode": "visual_only_forensic_fork",
        "run_identity": anchor_run,
        "entries": [{
            "run_identity": anchor_run,
            "end_update": 4815,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_control_hash": control_sha,
            "checkpoint_payload_present": True,
            "resumable": True,
        }],
    }
    index_path.write_text(json.dumps(anchor_index) + "\n", encoding="utf-8")
    anchor_index_path.write_text(
        json.dumps(anchor_index) + "\n", encoding="utf-8"
    )

    current["initialization"]["resume_checkpoint"] = str(checkpoint)
    current["execution_migration"].update({
        "anchor_run_identity": anchor_run,
        "prior_process_path": str(process_path),
        "prior_process_sha256": hashlib.sha256(process_path.read_bytes()).hexdigest(),
        "anchor_checkpoint_path": str(checkpoint),
        "anchor_control_sha256": control_sha,
        "anchor_index_path": str(anchor_index_path),
        "anchor_index_sha256": hashlib.sha256(
            anchor_index_path.read_bytes()
        ).hexdigest(),
    })
    current["artifacts"]["file_sha256"].update({
        str(process_path): current["execution_migration"]["prior_process_sha256"],
        str(control_path): control_sha,
        str(anchor_index_path): current["execution_migration"]["anchor_index_sha256"],
    })
    config = parse_query_state_training_config(current)
    _authenticate_execution_migration(
        config,
        checkpoint=checkpoint,
        run_root=run_root,
        environ={},
        require_actual_partition=False,
    )
    _authenticate_execution_migration(
        config,
        checkpoint=checkpoint,
        run_root=run_root,
        environ={"SLURM_JOB_PARTITION": "normal"},
    )
    assert _authenticated_exact_restart_update(
        config, checkpoint=checkpoint, run_root=run_root
    ) == 4815

    # The live index advances after every successor checkpoint; the immutable
    # update4815 snapshot must remain sufficient to authenticate the migration.
    advanced = deepcopy(anchor_index)
    advanced["entries"].append({
        "run_identity": anchor_run,
        "end_update": 5136,
        "checkpoint_path": str(run_root / "checkpoints/update_00005136"),
        "checkpoint_control_hash": "8" * 64,
        "checkpoint_payload_present": True,
        "resumable": True,
    })
    index_path.write_text(json.dumps(advanced) + "\n", encoding="utf-8")
    _authenticate_execution_migration(
        config,
        checkpoint=checkpoint,
        run_root=run_root,
        environ={"SLURM_JOB_PARTITION": "normal"},
    )
    with pytest.raises(ValueError, match="actual Slurm partition"):
        _authenticate_execution_migration(
            config,
            checkpoint=checkpoint,
            run_root=run_root,
            environ={"SLURM_JOB_PARTITION": "preempt"},
        )


def test_migrated_checkpoint_provenance_preserves_anchor_and_execution_chain() -> None:
    _, current = _migration_pair()
    config = parse_query_state_training_config(current)
    first = query_state_execution_provenance(config)
    assert first == {
        "schema": "nimloth_query_state_execution_provenance_v1",
        "anchor": {
            "run_identity": config.execution_migration["anchor_run_identity"],
            "source_commit": config.execution_migration["anchor_source_commit"],
            "source_manifest_path": config.execution_migration["anchor_source_manifest_path"],
            "source_manifest_identity": config.execution_migration[
                "anchor_source_manifest_identity"
            ],
            "partition": "preempt",
        },
        "execution_chain": [{
            "config_identity": config.identity,
            "source_commit": config.source["commit"],
            "source_manifest_path": config.source["source_manifest_path"],
            "source_manifest_identity": config.source["source_manifest_identity"],
            "partition": "normal",
            "approval_sha256": config.authorization["approval_sha256"],
        }],
    }
    assert query_state_execution_provenance(config, previous=first) == first

    lost = deepcopy(first)
    lost["anchor"]["run_identity"] = "0" * 64
    with pytest.raises(ValueError, match="lost|rewrote"):
        query_state_execution_provenance(config, previous=lost)


def test_process_evidence_inherits_checkpoint_execution_chain(tmp_path: Path) -> None:
    _, first_raw = _migration_pair()
    first = parse_query_state_training_config(first_raw)
    prior_provenance = query_state_execution_provenance(first)

    run_root = tmp_path / "run"
    checkpoint = run_root / "checkpoints" / "update_00005136"
    checkpoint.mkdir(parents=True)
    (checkpoint / "control.json").write_text(
        json.dumps({"execution_provenance": prior_provenance}) + "\n",
        encoding="utf-8",
    )
    _, current = _migration_pair()
    current["output"]["run_root"] = str(run_root)
    current["output"]["controller_root"] = str(tmp_path / "controller")
    current["initialization"]["resume_checkpoint"] = str(checkpoint)
    current["authorization"]["approval_sha256"] = "9" * 64
    current["execution_migration"]["approval_sha256"] = "9" * 64
    config = parse_query_state_training_config(current)
    inherited = _process_execution_provenance(config)
    assert inherited is not None
    assert inherited["execution_chain"][:-1] == prior_provenance["execution_chain"]
    assert inherited["execution_chain"][-1]["config_identity"] == config.identity


def test_migration_contract_rejects_wrong_anchor_and_actual_partition() -> None:
    prior, current = _migration_pair()
    config = parse_query_state_training_config(current)

    wrong_anchor = deepcopy(prior)
    wrong_anchor["source"]["commit"] = "9" * 40
    with pytest.raises(ValueError, match="anchor|migration"):
        validate_query_state_execution_migration_contract(config, wrong_anchor)

    with pytest.raises(ValueError, match="actual.*partition"):
        validate_query_state_execution_migration_contract(
            config, prior, actual_partition="preempt"
        )
