#!/usr/bin/env python3
"""Strict read-only finalization of the complete ID185 retry4 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_SOURCES = (
    "navigation_base_test_id185",
    "navigation_common_sense_test_id185",
    "navigation_complex_instruction_test_id185",
    "navigation_visual_appearance_test_id185",
    "navigation_long_horizon_test_id185",
)
EXPECTED_SNAPSHOT_ID = (
    "sha256:6648780b3791cb4b937974b151b9e119ed9bf74602d1bc21dabfc30a3914d969"
)
EXPECTED_WANDB_PATH = (
    "art2nd-hong-kong-university-of-science-and-technology/vagen/"
    "nimloth-id185-k4-full-eval-test300-retry4"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, Any] = {}
    for source in EXPECTED_SOURCES:
        selected = [row for row in rows if row["data_source"] == source]
        if len(selected) != 60:
            raise ValueError(f"ID185 audit source count mismatch: {source}")
        success = sum(float(row["traj_success"]) == 1.0 for row in selected)
        by_source[source] = {
            "row_count": 60,
            "success_count": success,
            "success_rate": success / 60,
            "reward_mean": sum(float(row["score"]) for row in selected) / 60,
        }
    success = sum(item["success_count"] for item in by_source.values())
    return {
        "step": 20,
        "row_count": len(rows),
        "success_count": success,
        "success_rate": success / len(rows),
        "reward_mean": sum(float(row["score"]) for row in rows) / len(rows),
        "by_data_source": by_source,
    }


def validate_evaluation_artifacts(source_run: Path) -> dict[str, Any]:
    """Validate complete immutable journal/dump evidence without writing it."""

    phase = source_run / "full_eval_test300"
    phase_status = _read_json(phase / "phase_status.json")
    if phase_status.get("status") != "failed" or phase_status.get("exit_code") != 1:
        raise ValueError("ID185 audit expects the disclosed post-finalizer failure")
    if list((source_run / "checkpoints").glob("global_step_*")):
        raise ValueError("ID185 eval-only source wrote a checkpoint")

    final_files = sorted((source_run / "validation").glob("*.jsonl"))
    if [path.name for path in final_files] != ["20.jsonl"]:
        raise ValueError("ID185 audit validation step mismatch")
    final_rows = _read_jsonl(final_files[0])
    if len(final_rows) != 300:
        raise ValueError("ID185 audit final dump is not 300 rows")

    journal = phase / "validation_batch_journal"
    complete = _read_json(journal / "complete.json")
    expected_counts = {source: 60 for source in EXPECTED_SOURCES}
    if complete != {
        **complete,
        "schema": "vagen_validation_batch_journal_complete_v1",
        "global_step": 20,
        "batch_count": 8,
        "row_count": 300,
        "data_source_counts": expected_counts,
    }:
        raise ValueError("ID185 audit journal complete marker mismatch")

    marker_paths = sorted(journal.glob("batch_*.complete.json"))
    if len(marker_paths) != 8:
        raise ValueError("ID185 audit journal marker count mismatch")
    journal_rows: list[dict[str, Any]] = []
    artifact_paths = [phase / "phase_status.json", final_files[0], journal / "complete.json"]
    expected_batch_rows = [40] * 7 + [20]
    for batch_index, marker_path in enumerate(marker_paths):
        marker = _read_json(marker_path)
        if (
            marker.get("schema") != "vagen_validation_batch_journal_v1"
            or marker.get("global_step") != 20
            or marker.get("batch_index") != batch_index
            or marker.get("row_count") != expected_batch_rows[batch_index]
        ):
            raise ValueError(f"ID185 audit batch marker mismatch: {batch_index}")
        rows_path = journal / marker["rows_file"]
        if _sha256(rows_path) != marker["rows_sha256"]:
            raise ValueError(f"ID185 audit batch hash mismatch: {batch_index}")
        batch_rows = _read_jsonl(rows_path)
        if len(batch_rows) != marker["row_count"]:
            raise ValueError(f"ID185 audit batch row mismatch: {batch_index}")
        journal_rows.extend(batch_rows)
        artifact_paths.extend((marker_path, rows_path))

    identities = [
        (row["rollout_sample_id"], int(row["rollout_repeat_index"]))
        for row in journal_rows
    ]
    if len(journal_rows) != 300 or len(set(identities)) != 300:
        raise ValueError("ID185 audit journal identities are not 300 unique rows")
    if any(str(row["uid"]).startswith("__vagen_validation_padding__") for row in journal_rows):
        raise ValueError("ID185 audit journal retained a padding row")
    if Counter(row["data_source"] for row in journal_rows) != Counter(expected_counts):
        raise ValueError("ID185 audit journal source counts mismatch")

    manifest = _read_json(phase / "dataset_manifest.json")
    expected_ids = {row["rollout_sample_id"] for row in manifest["validation_rows"]}
    if len(expected_ids) != 300 or expected_ids != {item[0] for item in identities}:
        raise ValueError("ID185 audit dataset/sample identity mismatch")
    artifact_paths.append(phase / "dataset_manifest.json")

    final_by_id = {row["rollout_sample_id"]: row for row in final_rows}
    journal_by_id = {row["rollout_sample_id"]: row for row in journal_rows}
    if len(final_by_id) != 300 or set(final_by_id) != set(journal_by_id):
        raise ValueError("ID185 audit final/journal identity mismatch")
    for sample_id, final_row in final_by_id.items():
        journal_row = journal_by_id[sample_id]
        if (
            final_row["step"] != 20
            or final_row["score"] != journal_row["score"]
            or final_row["data_source"] != journal_row["data_source"]
            or final_row["traj_success"] != journal_row["traj_success"]
        ):
            raise ValueError("ID185 audit final/journal row mismatch")

    return {
        "metrics": _compute_metrics(final_rows),
        "source_artifacts": {
            str(path.relative_to(source_run)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        },
    }


def _validate_external_evidence(
    slurm: dict[str, Any],
    wandb: dict[str, Any],
) -> None:
    if slurm != {
        "job_id": "524485",
        "state": "FAILED",
        "elapsed": "02:10:39",
        "exit_code": "1:0",
        "nodes": "dgx-[10,14,21,46]",
    }:
        raise ValueError("ID185 audit Slurm evidence mismatch")
    if (
        wandb.get("path") != EXPECTED_WANDB_PATH
        or wandb.get("state") != "finished"
        or wandb.get("name")
        != "185_eval_k4schemeb_dp8_tp8_source20_test5x60_t20_s100_c1_a1_b85p78297006578457_t1_cot07p095_retry4"
        or wandb.get("steps") != [20]
    ):
        raise ValueError("ID185 audit W&B evidence mismatch")


def _validate_wandb_metrics(
    wandb: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    history = wandb.get("history")
    if not isinstance(history, dict) or history.get("_step") != 20:
        raise ValueError("ID185 audit W&B history evidence mismatch")
    for source, source_metrics in metrics["by_data_source"].items():
        expected = {
            f"val-core/{source}/reward/mean@1": source_metrics["reward_mean"],
            f"val-aux/{source}/traj_success/mean@1": source_metrics["success_rate"],
        }
        for key, value in expected.items():
            actual = history.get(key)
            if not isinstance(actual, (int, float)) or abs(actual - value) > 1e-12:
                raise ValueError(f"ID185 audit W&B metric mismatch: {key}")


def _publish_atomic_directory(destination: Path, files: dict[str, Any]) -> None:
    if destination.exists():
        raise FileExistsError(f"ID185 audit output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        for name, payload in files.items():
            path = temporary / name
            if isinstance(payload, str):
                encoded = payload
            else:
                encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            path.write_text(encoded)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.rename(temporary, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--slurm-evidence", type=Path, required=True)
    parser.add_argument("--wandb-evidence", type=Path, required=True)
    args = parser.parse_args()

    if not str(args.source_run).endswith("cot07p095_retry4"):
        raise ValueError("ID185 audit source run mismatch")
    if not str(args.source_checkpoint).endswith("checkpoints/global_step_20"):
        raise ValueError("ID185 audit source checkpoint mismatch")
    slurm = _read_json(args.slurm_evidence)
    wandb = _read_json(args.wandb_evidence)
    _validate_external_evidence(slurm, wandb)
    artifact_audit = validate_evaluation_artifacts(args.source_run)
    _validate_wandb_metrics(wandb, artifact_audit["metrics"])

    import torch
    from nimloth.training.rl.joint_planner import load_frozen_planning_snapshot_file
    from vagen.joint_policy.checkpoint import load_complete_joint_checkpoint

    checkpoint = load_complete_joint_checkpoint(args.source_checkpoint)
    actor = checkpoint["actor_critic"]
    owner = checkpoint["frozen_q_owner"]
    active = owner["active_snapshot_state"]
    if (
        checkpoint["global_step"] != 20
        or actor["source_step"] != 796
        or active["snapshot_source_step"] != 796
        or active["snapshot_id"] != EXPECTED_SNAPSHOT_ID
        or actor["snapshot_id"] != EXPECTED_SNAPSHOT_ID
        or owner["activation_version"] != 20
    ):
        raise ValueError("ID185 audit source checkpoint identity mismatch")
    snapshot_path = Path(active["transport_path"])
    snapshot = load_frozen_planning_snapshot_file(snapshot_path, device=torch.device("cpu"))
    if snapshot.source_step != 796 or snapshot.snapshot_id != EXPECTED_SNAPSHOT_ID:
        raise ValueError("ID185 audit active source transport mismatch")

    source_log = (args.source_run / "full_eval_test300" / "train.log").read_text()
    for marker in (
        "ID185_K4_FULL_EVAL_RESTORE_OK global_step=20",
        "VALIDATION_BATCH_JOURNAL_COMPLETE batches=8 rows=300",
    ):
        if marker not in source_log:
            raise ValueError(f"ID185 audit runtime marker missing: {marker}")

    summary = {
        "status": "passed",
        "phase": "readonly_finalization_audit",
        "source_evaluation_job": slurm,
        "source_evaluation_status": "failed_postrun_finalizer_only",
        "source_run": str(args.source_run),
        "source_checkpoint": str(args.source_checkpoint),
        "global_step": 20,
        "source_step": 796,
        "snapshot_id": EXPECTED_SNAPSHOT_ID,
        "checkpoint_steps_written": [],
        "wandb": wandb,
        **artifact_audit,
    }
    readme = (
        "# ID185 retry4 strict read-only finalization audit\n\n"
        "This independent audit does not modify or resume the failed source output. "
        "It verifies the one complete 300-row evaluation, immutable source796 snapshot, "
        "all journal hashes and identities, final dump agreement, Slurm disclosure, and "
        "the existing finished W&B step20.\n"
    )
    _publish_atomic_directory(
        args.audit_output,
        {
            "validator.json": summary,
            "final_status.json": summary,
            "slurm_evidence.json": slurm,
            "wandb_evidence.json": wandb,
            "README.md": readme,
        },
    )
    print(json.dumps({"status": "ID185_READONLY_FINALIZATION_ALL_OK", **summary["metrics"]}, allow_nan=False))


if __name__ == "__main__":
    main()
