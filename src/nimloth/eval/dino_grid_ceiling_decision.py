"""Publish a human-scored decision bound to one immutable grid-ceiling report."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nimloth.eval.dino_grid_reconstruction_ceiling import (
    _has_symlink_component,
    _mapping_identity,
    _reject_protected_output,
    _sha256_file,
    apply_human_structure_judgments,
    validate_dino_grid_ceiling_report,
)
from nimloth.training.reconstruction.forensic_query_state_cache import (
    _publish_noreplace,
)

DINO_GRID_CEILING_JUDGMENTS_SCHEMA = (
    "nimloth_dino_grid_ceiling_human_judgments_v1"
)
DINO_GRID_CEILING_DECISION_SCHEMA = "nimloth_dino_grid_ceiling_decision_v1"


def validate_human_judgments(
    payload: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
) -> dict[str, tuple[bool, ...]]:
    """Bind every boolean judgment to the fixed ID198 row and contact sheet."""

    contact = report.get("contact_sheet")
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    expected_rows = (
        list(zip(contact.get("row_identities", ()), contact.get("image_sha256", ())))
        if isinstance(contact, Mapping)
        else []
    )
    if (
        not isinstance(payload, Mapping)
        or set(payload)
        != {
            "schema",
            "report_artifact_identity",
            "contact_sheet_identity",
            "contact_sheet_sha256",
            "rows",
            "identity",
        }
        or payload.get("schema") != DINO_GRID_CEILING_JUDGMENTS_SCHEMA
        or payload.get("report_artifact_identity")
        != report.get("artifact_identity")
        or payload.get("contact_sheet_identity") != contact.get("identity")
        or payload.get("contact_sheet_sha256") != contact.get("png_sha256")
        or not isinstance(rows, list)
        or len(rows) != 16
        or len(expected_rows) != 16
        or payload.get("identity")
        != _mapping_identity(
            {key: value for key, value in payload.items() if key != "identity"}
        )
    ):
        raise ValueError("human judgments are not bound to the exact report/contact sheet")
    for item, (row_identity, image_sha256) in zip(rows, expected_rows, strict=True):
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "row_identity",
                "original_image_sha256",
                "grid8_recognizable",
                "grid16_recognizable",
            }
            or item.get("row_identity") != row_identity
            or item.get("original_image_sha256") != image_sha256
            or type(item.get("grid8_recognizable")) is not bool
            or type(item.get("grid16_recognizable")) is not bool
        ):
            raise ValueError("human judgment row/order/image identity is invalid")
    return {
        "grid8": tuple(bool(item["grid8_recognizable"]) for item in rows),
        "grid16": tuple(bool(item["grid16_recognizable"]) for item in rows),
    }


def publish_human_decision(
    output: str | Path,
    *,
    report_root: str | Path,
    judgments_path: str | Path,
) -> Mapping[str, Any]:
    """Publish one non-overwriting, manifest-last mixed human/statistical decision."""

    destination = Path(output)
    report_path = Path(report_root)
    judgment_file = Path(judgments_path)
    _reject_protected_output(destination, (report_path, judgment_file))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("human decision output already exists")
    if (
        not report_path.is_absolute()
        or _has_symlink_component(report_path)
        or not report_path.is_dir()
        or not judgment_file.is_absolute()
        or _has_symlink_component(judgment_file)
        or not judgment_file.is_file()
    ):
        raise ValueError("human decision inputs must be absolute regular paths")
    report = validate_dino_grid_ceiling_report(report_path)
    try:
        judgments = json.loads(judgment_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("human judgment input is unreadable") from error
    if not isinstance(judgments, Mapping):
        raise TypeError("human judgment input must be a mapping")
    boolean_judgments = validate_human_judgments(judgments, report=report)
    decision = apply_human_structure_judgments(
        report["comparison"], judgments=boolean_judgments
    )
    decision_payload: dict[str, Any] = {
        "decision": decision,
        "human_judgments_identity": judgments["identity"],
        "source_report_artifact_identity": report["artifact_identity"],
    }
    decision_payload["identity"] = _mapping_identity(decision_payload)
    manifest: dict[str, Any] = {
        "schema": DINO_GRID_CEILING_DECISION_SCHEMA,
        "status": "completed_human_and_statistical_decision",
        "source_report": {
            "path": str(report_path.resolve()),
            "manifest_sha256": _sha256_file(report_path / "manifest.json"),
            "artifact_identity": report["artifact_identity"],
            "contact_sheet_identity": report["contact_sheet"]["identity"],
            "contact_sheet_sha256": report["contact_sheet"]["png_sha256"],
        },
        "human_judgments": {
            "path": str(judgment_file.resolve()),
            "sha256": _sha256_file(judgment_file),
            "identity": judgments["identity"],
        },
        "decision": decision,
        "decision_payload": {
            "file": "decision.json",
            "identity": decision_payload["identity"],
        },
        "claim_scope": "forensic_representation_decodability_only",
        "actor_sft1_sft2_deployability_verdicts_unchanged": True,
    }
    manifest["artifact_identity"] = _mapping_identity(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging.", dir=destination.parent
        )
    )
    try:
        decision_path = staging / "decision.json"
        with decision_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(decision_payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        manifest["decision_payload"]["sha256"] = _sha256_file(decision_path)
        manifest["artifact_identity"] = _mapping_identity(
            {key: value for key, value in manifest.items() if key != "artifact_identity"}
        )
        with (staging / "manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _publish_noreplace(staging, destination)
        return validate_human_decision(destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def validate_human_decision(root: str | Path) -> Mapping[str, Any]:
    supplied = Path(root)
    if _has_symlink_component(supplied):
        raise ValueError("human decision root must not traverse symlinks")
    decision_root = supplied.resolve()
    manifest_path = decision_root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("human decision requires a regular manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("human decision manifest is unreadable") from error
    if (
        not isinstance(manifest, Mapping)
        or set(manifest)
        != {
            "schema",
            "status",
            "source_report",
            "human_judgments",
            "decision",
            "decision_payload",
            "claim_scope",
            "actor_sft1_sft2_deployability_verdicts_unchanged",
            "artifact_identity",
        }
        or manifest.get("schema") != DINO_GRID_CEILING_DECISION_SCHEMA
        or manifest.get("status") != "completed_human_and_statistical_decision"
        or manifest.get("claim_scope")
        != "forensic_representation_decodability_only"
        or manifest.get("actor_sft1_sft2_deployability_verdicts_unchanged")
        is not True
        or manifest.get("artifact_identity")
        != _mapping_identity(
            {key: value for key, value in manifest.items() if key != "artifact_identity"}
        )
    ):
        raise ValueError("human decision manifest schema/status/identity is invalid")
    source = manifest["source_report"]
    judgments_owner = manifest["human_judgments"]
    decision_owner = manifest["decision_payload"]
    if (
        not isinstance(source, Mapping)
        or not isinstance(judgments_owner, Mapping)
        or not isinstance(decision_owner, Mapping)
    ):
        raise TypeError("human decision source owners are invalid")
    report_root = Path(str(source.get("path", "")))
    judgment_file = Path(str(judgments_owner.get("path", "")))
    decision_path = decision_root / str(decision_owner.get("file", ""))
    report = validate_dino_grid_ceiling_report(report_root)
    try:
        judgments = json.loads(judgment_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("human decision live judgments are unreadable") from error
    if (
        set(source)
        != {
            "path",
            "manifest_sha256",
            "artifact_identity",
            "contact_sheet_identity",
            "contact_sheet_sha256",
        }
        or source.get("manifest_sha256")
        != _sha256_file(report_root / "manifest.json")
        or source.get("artifact_identity") != report["artifact_identity"]
        or source.get("contact_sheet_identity") != report["contact_sheet"]["identity"]
        or source.get("contact_sheet_sha256") != report["contact_sheet"]["png_sha256"]
        or set(judgments_owner) != {"path", "sha256", "identity"}
        or judgments_owner.get("sha256") != _sha256_file(judgment_file)
        or judgments_owner.get("identity") != judgments.get("identity")
        or set(decision_owner) != {"file", "sha256", "identity"}
        or decision_owner.get("file") != "decision.json"
        or not decision_path.is_file()
        or decision_path.is_symlink()
        or decision_owner.get("sha256") != _sha256_file(decision_path)
    ):
        raise ValueError("human decision live report/judgment identity drift")
    try:
        decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("human decision payload is unreadable") from error
    if (
        not isinstance(decision_payload, Mapping)
        or set(decision_payload)
        != {
            "decision",
            "human_judgments_identity",
            "source_report_artifact_identity",
            "identity",
        }
        or decision_payload.get("identity") != decision_owner.get("identity")
        or decision_payload.get("identity")
        != _mapping_identity(
            {
                key: value
                for key, value in decision_payload.items()
                if key != "identity"
            }
        )
        or decision_payload.get("human_judgments_identity")
        != judgments.get("identity")
        or decision_payload.get("source_report_artifact_identity")
        != report.get("artifact_identity")
    ):
        raise ValueError("human decision payload identity is invalid")
    boolean_judgments = validate_human_judgments(judgments, report=report)
    expected_decision = apply_human_structure_judgments(
        report["comparison"], judgments=boolean_judgments
    )
    if (
        manifest.get("decision") != expected_decision
        or decision_payload.get("decision") != expected_decision
    ):
        raise ValueError("human decision recomputation mismatch")
    return manifest


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a report/contact-bound DINO-grid human decision"
    )
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_cli_parser().parse_args(argv)
    result = publish_human_decision(
        args.output_dir,
        report_root=args.report_root,
        judgments_path=args.judgments,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DINO_GRID_CEILING_DECISION_SCHEMA",
    "DINO_GRID_CEILING_JUDGMENTS_SCHEMA",
    "build_cli_parser",
    "publish_human_decision",
    "validate_human_decision",
    "validate_human_judgments",
]
