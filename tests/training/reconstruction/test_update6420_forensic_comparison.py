from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from nimloth.training.reconstruction.update6420_forensic_comparison import (
    BASELINE_INVARIANTS_SHA256,
    LOCKED_UPDATE6420_EXPECTED,
    UPDATE6420_CACHE_SCHEMA,
    _epoch1_metric_input,
    build_comparison_artifact,
    build_inspection_contract,
    build_matched_cfm_invariants,
    canonical_identity,
    validate_cache_manifest,
    validate_checkpoint_evidence,
    validate_matched_rows,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _checkpoint(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    checkpoint = tmp_path / "checkpoints" / "update_00006420"
    checkpoint.mkdir(parents=True)
    files: dict[str, str] = {}
    control = checkpoint / "control.json"
    control.write_text('{"forensic_only":false,"global_step":6420,"run_identity":"' + "1" * 64 + '","source_commit":"' + "a" * 40 + '"}\n')
    completed = checkpoint / "COMPLETED"
    completed.write_text("complete\n")
    entry = {"update": 6420, "epoch_final": True, "checkpoint_payload_present": True, "resumable": True, "run_identity": "1" * 64}
    index = tmp_path / "authoritative_index.json"
    index.write_text(__import__("json").dumps({"entries": [entry]}) + "\n")
    for name, payload in (
        ("resolved_config", {"identity": "1" * 64}),
        ("anchor_manifest", {"identity": "6" * 64}),
        ("migration_manifest", {"identity": "8" * 64}),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(__import__("json").dumps(payload) + "\n")
        files[name] = str(path)
    files.update({"control": str(control), "completed": str(completed), "authoritative_index": str(index)})
    segment_files = {}
    for index_value, name in enumerate(("commit", "cursors", "mirror", "owner", "safety", "updates", "validation"), start=1):
        path = tmp_path / f"segment-{name}.json"
        path.write_text(f"{name}\n")
        segment_files[name] = str(path)
    files["segment"] = segment_files  # type: ignore[assignment]
    ranks = []
    for rank in range(8):
        payload = checkpoint / f"rank_{rank:05d}_of_00008.pt"
        sidecar = checkpoint / f"rank_{rank:05d}_of_00008.json"
        payload.write_bytes(f"rank-{rank}".encode())
        sidecar.write_bytes(f"sidecar-{rank}".encode())
        ranks.append({
            "rank": rank,
            "payload_path": str(payload),
            "payload_sha256": _sha_bytes(payload.read_bytes()),
            "sidecar_path": str(sidecar),
            "sidecar_sha256": _sha_bytes(sidecar.read_bytes()),
        })
    evidence = {
        "update": 6420,
        "epoch_final": True,
        "checkpoint_payload_present": True,
        "resumable": True,
        "forensic_only": False,
        "terminal_primary": False,
        "source_commit": "a" * 40,
        "execution_source_commit": "b" * 40,
        "run_identity": "1" * 64,
        "config_identity": "1" * 64,
        "control_sha256": _sha_bytes(control.read_bytes()),
        "completed_sha256": _sha_bytes(completed.read_bytes()),
        "index_entry_sha256": canonical_identity(entry),
        "resolved_config_sha256": _sha_bytes(Path(files["resolved_config"]).read_bytes()),
        "anchor_manifest_sha256": _sha_bytes(Path(files["anchor_manifest"]).read_bytes()),
        "anchor_manifest_identity": "6" * 64,
        "migration_manifest_sha256": _sha_bytes(Path(files["migration_manifest"]).read_bytes()),
        "migration_manifest_identity": "8" * 64,
        "migration_approval_sha256": "9" * 64,
        "segment_sidecar_sha256": {name: _sha_bytes(Path(path).read_bytes()) for name, path in segment_files.items()},
        "actor": {"unsafe": True, "deployable": False, "kl": 1.5, "top1": 0.7, "rms_ratio": 2.0},
        "rank_payloads": ranks,
        "files": files,
    }
    expected = {key: value for key, value in evidence.items() if key not in {"rank_payloads", "files"}}
    expected["evidence_paths"] = files
    expected["rank_payload_sha256"] = [item["payload_sha256"] for item in ranks]
    expected["rank_sidecar_sha256"] = [item["sidecar_sha256"] for item in ranks]
    return evidence, expected


def test_locked_update6420_segment_paths_match_live_owner_filenames() -> None:
    segment = LOCKED_UPDATE6420_EXPECTED["evidence_paths"]["segment"]
    assert Path(segment["mirror"]).name == "mirror_batch.json"
    assert Path(segment["updates"]).name == "updates.jsonl"


def test_checkpoint_rejects_identity_payload_optimizer_and_consumer_confusion(tmp_path: Path) -> None:
    evidence, expected = _checkpoint(tmp_path)
    validated = validate_checkpoint_evidence(evidence, expected=expected)
    assert validated["forensic_only"] is False
    assert validated["actor"] == evidence["actor"]

    for field, bad in (("update", 6419), ("source_commit", "f" * 40), ("control_sha256", "f" * 64)):
        drift = dict(evidence)
        drift[field] = bad
        with pytest.raises(ValueError):
            validate_checkpoint_evidence(drift, expected=expected)

    compacted = dict(evidence)
    compacted["checkpoint_payload_present"] = False
    with pytest.raises(ValueError):
        validate_checkpoint_evidence(compacted, expected=expected)
    Path(evidence["rank_payloads"][0]["payload_path"]).unlink()  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_checkpoint_evidence(evidence, expected=expected)
    live_optimizer = dict(evidence)
    live_optimizer["optimizer"] = object()
    with pytest.raises(ValueError):
        validate_checkpoint_evidence(live_optimizer, expected=expected)
    deployable = dict(evidence)
    deployable["actor"] = {**evidence["actor"], "deployable": True}  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_checkpoint_evidence(deployable, expected=expected)


def _row(index: int, role: str) -> dict[str, object]:
    row = {
        "selection_ordinal": index,
        "selection_role": role,
        "row_identity": hashlib.sha256(f"row-{index}".encode()).hexdigest(),
        "record_id": f"record-{index}",
        "step_index": index,
        "original_image_path": f"/images/{index}.png",
        "original_image_sha256": hashlib.sha256(f"image-{index}".encode()).hexdigest(),
        "archived_assistant_response_sha256": hashlib.sha256(f"response-{index}".encode()).hexdigest(),
        "response_source": "archived",
        "encoded_input_identity": hashlib.sha256(f"encoded-{index}".encode()).hexdigest(),
        "messages_identity": hashlib.sha256(f"messages-{index}".encode()).hexdigest(),
        "prompt_history_identity": hashlib.sha256(f"prompt-{index}".encode()).hexdigest(),
        "renderer_identity": hashlib.sha256(f"renderer-{index}".encode()).hexdigest(),
        "template_identity": hashlib.sha256(f"template-{index}".encode()).hexdigest(),
    }
    row["observation_identity"] = canonical_identity({
        key: row[key] for key in ("record_id", "step_index", "original_image_path", "original_image_sha256")
    })
    row["archived_response_identity"] = row["archived_assistant_response_sha256"]
    return row


def test_cache_requires_exact_archived_rows_and_rejects_deployable_reader() -> None:
    baseline = [_row(0, "all_train"), _row(1, "external_validation")]
    comparison = [dict(row) for row in baseline]
    digests = validate_matched_rows(comparison, baseline_rows=baseline, expected_counts={"all_train": 1, "external_validation": 1})
    manifest = {
        "schema": UPDATE6420_CACHE_SCHEMA,
        "actor_unsafe": True,
        "deployable": False,
        "forensic_only": False,
        "state_shape": [16, 1024],
        "state_dtype": "float32",
        "count": 2,
        "ordered_identity_digests": digests,
    }
    assert validate_cache_manifest(manifest, expected_digests=digests, expected_count=2)["actor_unsafe"] is True
    synthetic = [dict(row) for row in comparison]
    synthetic[0]["response_source"] = "synthetic"
    with pytest.raises(ValueError):
        validate_matched_rows(synthetic, baseline_rows=baseline, expected_counts={"all_train": 1, "external_validation": 1})
    drift = [dict(row) for row in comparison]
    drift.reverse()
    with pytest.raises(ValueError):
        validate_matched_rows(drift, baseline_rows=baseline, expected_counts={"all_train": 1, "external_validation": 1})
    with pytest.raises(ValueError):
        validate_cache_manifest({**manifest, "deployable": True}, expected_digests=digests, expected_count=2)


def _baseline_invariants() -> dict[str, object]:
    return {
        "state_shape": [16, 1024], "image_size": 128, "input_channels": 3, "output_channels": 3,
        "base_channels": 64, "condition_dim": 256, "time_dim": 512, "batch_size": 32,
        "learning_rate": 1e-4, "weight_decay": 1e-4, "gradient_clip": 1.0,
        "max_steps": 4000, "evaluation_interval": 1000, "save_interval": 1000,
        "seed": 20260921, "noise_seeds": [20260931, 20260932, 20260933],
        "sample_items": 16, "sample_ode_steps": 50, "sample_noise_seed": 20260921,
        "sample_batch_size": 8, "shuffle_algorithm": "global_cyclic_shift_v1",
        "correct_and_shuffled_share_noise_and_time": True,
        "metric_unit": "mean conditional-flow velocity MSE per normalized [-1,1] RGB element",
        "checkpoint_selection": "final_step4000_only", "pass_min_delta": 0.01,
        "pass_min_aggregate_ratio": 1.05, "image_preprocessing": {"color_space": "sRGB", "resample": "bicubic", "range": [-1, 1]},
    }


def test_actual_epoch1_terminal_record_projects_locked_metrics_and_curve() -> None:
    per_seed = [
        {"seed": seed, "correct": correct, "shuffled": shuffled}
        for seed, correct, shuffled in (
            (20260931, 0.05068443708190324, 0.05725616668018144),
            (20260932, 0.04561192902452289, 0.05730876067135238),
            (20260933, 0.05056978969901315, 0.06146096816235034),
        )
    ]
    terminal = {
        "schema": "nimloth_formal38_forensic_stage_b_cfm_task_end_v1",
        "recorded_at": "locked", "slurm": {"state": "COMPLETED", "exit_code": "0:0"},
        "source": {"commit": "cd1c002358b6b78e4607a1c7e5ecad6dad3b0e86"},
        "input": {"all_train_items": 12836, "external_validation_items": 1413, "state_shape": [16, 1024]},
        "training": {
            "final_step": 4000,
            "train_flow_mse_at_steps": {
                "1000": 0.06458581984043121, "2000": 0.045169197022914886,
                "3000": 0.05132240056991577, "4000": 0.058027178049087524,
            },
        },
        "final_external_gate": {"passed": False, "per_seed": per_seed},
        "artifacts": {"final_checkpoint_sha256": "52bf18e22aba3dd5055a51b07c94c4488ded9a5134df87c48b0818cb31798929"},
        "wandb": {}, "scientific_status": "publication_gate_failed",
        "resume": "terminal", "validity": "forensic only",
    }
    assert _epoch1_metric_input(terminal) == {
        "checkpoint_sha256": terminal["artifacts"]["final_checkpoint_sha256"],
        "per_seed": per_seed,
    }


def test_matched_cfm_comparison_and_actual_gate_inspection_contract() -> None:
    baseline = _baseline_invariants()
    invariants = build_matched_cfm_invariants(
        baseline, cache_fingerprint="a" * 64, checkpoint_identity="b" * 64,
        row_identity_digest="c" * 64,
    )
    assert invariants["baseline_invariants_sha256"] == BASELINE_INVARIANTS_SHA256
    drift = dict(baseline)
    drift["max_steps"] = 4001
    with pytest.raises(ValueError):
        build_matched_cfm_invariants(drift, cache_fingerprint="a" * 64, checkpoint_identity="b" * 64, row_identity_digest="c" * 64)

    epoch1 = {
        "checkpoint_sha256": "52bf18e22aba3dd5055a51b07c94c4488ded9a5134df87c48b0818cb31798929",
        "per_seed": [
            {"seed": 20260931, "correct": 0.05068443708190324, "shuffled": 0.05725616668018144},
            {"seed": 20260932, "correct": 0.04561192902452289, "shuffled": 0.05730876067135238},
            {"seed": 20260933, "correct": 0.05056978969901315, "shuffled": 0.06146096816235034},
        ],
    }
    update = {
        "checkpoint_sha256": "e" * 64,
        "per_seed": [
            {"seed": 20260931, "correct": .04, "shuffled": .06},
            {"seed": 20260932, "correct": .035, "shuffled": .056},
            {"seed": 20260933, "correct": .043, "shuffled": .059},
        ],
    }
    artifact = build_comparison_artifact(epoch1=epoch1, update6420=update, cache_manifest_sha256="f" * 64)
    assert artifact["claim_boundary"] == "representation_decodability_and_condition_use_only"
    assert artifact["update6420_minus_epoch1"]["correct_mse"] < 0

    passed = build_inspection_contract(gate={"passed": True}, decoder_checkpoint_sha256="e" * 64, cache_manifest_sha256="f" * 64)
    assert "publication_gate_failed" not in passed["watermarks"]
    assert passed["correct_condition_only"] is True
    failed = build_inspection_contract(gate={"passed": False}, decoder_checkpoint_sha256="e" * 64, cache_manifest_sha256="f" * 64)
    assert "publication_gate_failed" in failed["watermarks"]
    with pytest.raises(ValueError):
        build_inspection_contract(gate={}, decoder_checkpoint_sha256="e" * 64, cache_manifest_sha256="f" * 64)
