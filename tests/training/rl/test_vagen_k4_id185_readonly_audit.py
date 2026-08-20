from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/training/rl/finalize_vagen_k4_id185_readonly_audit.py"
SPEC = importlib.util.spec_from_file_location("id185_readonly_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _build_complete_source(root: Path) -> Path:
    phase = root / "full_eval_test300"
    journal = phase / "validation_batch_journal"
    journal.mkdir(parents=True)
    _json(phase / "phase_status.json", {"status": "failed", "exit_code": 1})
    final_rows = []
    manifest_rows = []
    batch_sizes = [40] * 7 + [20]
    cursor = 0
    source_counts = {source: 60 for source in MODULE.EXPECTED_SOURCES}
    for batch_index, batch_size in enumerate(batch_sizes):
        rows = []
        for _ in range(batch_size):
            source = MODULE.EXPECTED_SOURCES[cursor // 60]
            sample_id = f"sample-{cursor:03d}"
            success = float(cursor % 3 == 0)
            row = {
                "input": f"input-{cursor}",
                "output": f"output-{cursor}",
                "gts": {},
                "score": success,
                "reward": success,
                "traj_success": success,
                "step": 20,
                "uid": f"uid-{cursor}",
                "data_source": source,
                "rollout_sample_id": sample_id,
                "rollout_repeat_index": 0,
            }
            rows.append(row)
            final_rows.append(row)
            manifest_rows.append(
                {
                    "rollout_sample_id": sample_id,
                    "data_source": source,
                    "seed": cursor % 60 + 1,
                }
            )
            cursor += 1
        payload = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
        rows_file = f"batch_{batch_index:04d}.jsonl"
        (journal / rows_file).write_bytes(payload)
        _json(
            journal / f"batch_{batch_index:04d}.complete.json",
            {
                "schema": "vagen_validation_batch_journal_v1",
                "global_step": 20,
                "batch_index": batch_index,
                "row_count": batch_size,
                "rows_file": rows_file,
                "rows_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            },
        )
    _json(
        journal / "complete.json",
        {
            "schema": "vagen_validation_batch_journal_complete_v1",
            "global_step": 20,
            "batch_count": 8,
            "row_count": 300,
            "data_source_counts": source_counts,
        },
    )
    _json(phase / "dataset_manifest.json", {"validation_rows": manifest_rows})
    validation = root / "validation"
    validation.mkdir()
    (validation / "20.jsonl").write_text(
        "\n".join(json.dumps(row) for row in final_rows) + "\n"
    )
    return root


def test_id185_readonly_audit_accepts_one_complete_immutable_evaluation(
    tmp_path: Path,
) -> None:
    source = _build_complete_source(tmp_path / "source")
    result = MODULE.validate_evaluation_artifacts(source)
    assert result["metrics"]["row_count"] == 300
    assert result["metrics"]["success_count"] == 100
    assert len(result["source_artifacts"]) == 20


def test_id185_readonly_audit_rejects_journal_corruption(tmp_path: Path) -> None:
    source = _build_complete_source(tmp_path / "source")
    with (source / "full_eval_test300/validation_batch_journal/batch_0007.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="batch hash"):
        MODULE.validate_evaluation_artifacts(source)


def test_id185_readonly_audit_refuses_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "audit"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        MODULE._publish_atomic_directory(destination, {"final_status.json": {}})
