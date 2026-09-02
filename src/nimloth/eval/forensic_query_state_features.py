"""Strict direct-feature diagnostics for Formal38 unsafe update1605.

This entry point is deliberately separate from the deployable Query-State feature
CLI. It accepts only typed Stage A/B forensic cache owners and preserves the actor
safety failure as the highest-level evidence.  No operation here promotes the
checkpoint, cache, or report to a deployable artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from nimloth.eval.query_state_features import (
    NIMLOTH_SHARED_BASIS_METHOD,
    QueryStateFeatureRecord,
    SharedFeatureBasis,
    SharedFeatureBasisIdentity,
    _build_pinned_dino_teacher,
    _canonical_json,
    _feature_row_set_identity,
    _feature_split_identity,
    _fit_shared_feature_basis_from_records,
    _is_sha256,
    _render_query_state_feature_report_from_records,
    _sha256_file,
    aggregate_direct_feature_metrics,
    dino_feature_identity,
    extract_dino_feature_records,
    load_shared_feature_basis,
)
from nimloth.training.reconstruction.forensic_query_state_cache import (
    FORENSIC_QUERY_STATE_CACHE_SCHEMA,
    FORENSIC_QUERY_STATE_OWNER_ROLE,
    FORENSIC_SELECTION_ALL_TRAIN,
    FORENSIC_SELECTION_EXTERNAL_VALIDATION,
    FORENSIC_SELECTION_MECHANICS_TRAIN,
    FORENSIC_SELECTION_MECHANICS_VALIDATION,
    ForensicExperimentStage,
    ForensicQueryStateCacheDataset,
)

FORMAL38_FORENSIC_SOURCE_COMMIT = "4838e5fdb469dffb78909e307cf11a808cb2d29e"
FORMAL38_UNSAFE_UPDATE1605_CONTROL_SHA256 = (
    "414daefe2b501a22805691aa101d76fcc0f5b28447a1332d81b19b3434e838af"
)
FORENSIC_FEATURE_REPORT_SCHEMA = "nimloth_forensic_query_state_feature_report_v1"
_FORENSIC_BASIS_RECEIPT_SCHEMA = "nimloth_forensic_query_state_feature_basis_receipt_v1"
_STAGE_A_ROLE_COUNTS = {
    FORENSIC_SELECTION_MECHANICS_TRAIN: 48,
    FORENSIC_SELECTION_MECHANICS_VALIDATION: 16,
}
_STAGE_B_ROLE_COUNTS = {
    FORENSIC_SELECTION_ALL_TRAIN: 12_836,
    FORENSIC_SELECTION_EXTERNAL_VALIDATION: 1_413,
}
_STAGE_B_VISUAL_COUNT = 16
_STAGE_B_VISUAL_SEED = 20260921


def _stage_roles(stage: str) -> tuple[str, str, Mapping[str, int]]:
    if stage == ForensicExperimentStage.MECHANICS_ONLY.value:
        return (
            FORENSIC_SELECTION_MECHANICS_TRAIN,
            FORENSIC_SELECTION_MECHANICS_VALIDATION,
            _STAGE_A_ROLE_COUNTS,
        )
    if stage == ForensicExperimentStage.STAGE_B_DIAGNOSTIC.value:
        return (
            FORENSIC_SELECTION_ALL_TRAIN,
            FORENSIC_SELECTION_EXTERNAL_VALIDATION,
            _STAGE_B_ROLE_COUNTS,
        )
    raise ValueError("unsupported forensic feature experiment stage")
_BASIS_RECEIPT_FIELDS = {
    "schema", "basis_file", "basis_file_sha256", "basis_artifact_sha256",
    "global_scale_sha256", "cache_schema", "cache_fingerprint",
    "checkpoint_identity", "selection_identity", "fit_role",
    "transform_only_role", "forensic_only", "unsafe_actor_checkpoint",
    "not_deployable", "mechanics_only", "not_heldout",
    "formal38_calibration_80_aggregation_reproduced",
    "deep_sight_exact_colorization",
}
_PROVENANCE_FIELDS = (
    "prompt_history_identity",
    "messages_identity",
    "renderer_identity",
    "template_identity",
    "encoded_input_identity",
    "response_source",
)


def _role_identity(
    *, cache_fingerprint: str, selection_identity: str, role: str, rows: Sequence[Mapping[str, Any]]
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "cache_fingerprint": cache_fingerprint,
                "selection_identity": selection_identity,
                "role": role,
                "ordered_rows": [
                    {
                        "selection_ordinal": row["selection_ordinal"],
                        "row_identity": row["row_identity"],
                        "image_sha256": row["original_image_sha256"],
                        "archived_response_sha256": row[
                            "archived_assistant_response_sha256"
                        ],
                    }
                    for row in rows
                ],
            }
        )
    ).hexdigest()


def _matches_formal38_actor_failure(value: object) -> bool:
    if not isinstance(value, Mapping) or value.get("passed") is not False:
        return False
    kl = value.get("kl")
    top1 = value.get("top1_agreement")
    return (
        isinstance(kl, (int, float))
        and not isinstance(kl, bool)
        and isinstance(top1, (int, float))
        and not isinstance(top1, bool)
        and math.isclose(float(kl), 1.057509183883667, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(float(top1), 0.675000011920929, rel_tol=0.0, abs_tol=1e-6)
    )


def _strict_forensic_records(
    cache_dir: str | Path,
) -> tuple[
    Mapping[str, Any],
    dict[str, list[QueryStateFeatureRecord]],
    dict[str, dict[str, Mapping[str, Any]]],
]:
    """Load and rebind every record through the strict forensic cache reader."""

    dataset = ForensicQueryStateCacheDataset(cache_dir)
    manifest = dataset.manifest
    checkpoint = manifest.get("checkpoint")
    selection = manifest.get("selection")
    if (
        manifest.get("schema") != FORENSIC_QUERY_STATE_CACHE_SCHEMA
        or manifest.get("owner_role") != FORENSIC_QUERY_STATE_OWNER_ROLE
        or manifest.get("forensic_only") is not True
        or any(
            manifest.get(field) is not False
            for field in ("authoritative", "terminal_primary", "deployable", "sft2_ready")
        )
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("source_commit") != FORMAL38_FORENSIC_SOURCE_COMMIT
        or checkpoint.get("control_sha256")
        != FORMAL38_UNSAFE_UPDATE1605_CONTROL_SHA256
        or Path(str(checkpoint.get("checkpoint_path", ""))).name
        != "unsafe_update_00001605"
        or checkpoint.get("world_size") != 8
        or not _is_sha256(checkpoint.get("failure_manifest_sha256"))
        or not _matches_formal38_actor_failure(checkpoint.get("actor_failure"))
        or not isinstance(selection, Mapping)
    ):
        raise ValueError(
            "forensic feature adapter requires exact Formal38 unsafe update1605 "
            "checkpoint/failure evidence"
        )
    _fit_role, _transform_role, role_counts = _stage_roles(str(selection.get("stage")))
    if selection.get("roles") != role_counts:
        raise ValueError("forensic feature stage/role/count contract mismatch")
    cache_fingerprint = str(manifest.get("cache_fingerprint"))
    source_manifest_identity = str(manifest["source_jsonl"]["source_manifest_identity"])
    source_sha256_by_role = {
        role: str(manifest["source_jsonl"]["train"]["sha256"])
        for role in role_counts
    }
    if selection.get("stage") == ForensicExperimentStage.STAGE_B_DIAGNOSTIC.value:
        source_sha256_by_role[_transform_role] = str(
            manifest["source_jsonl"]["validation"]["sha256"]
        )
    selection_identity = str(selection["identity"])
    if any(
        not _is_sha256(value)
        for value in (
            cache_fingerprint,
            source_manifest_identity,
            selection_identity,
            *source_sha256_by_role.values(),
        )
    ):
        raise ValueError("forensic feature cache/source/selection identity is invalid")

    raw_by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in role_counts}
    state_by_identity: dict[str, torch.Tensor] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        state = item.pop("state")
        role = str(item.get("selection_role"))
        if role not in raw_by_role:
            raise ValueError("forensic feature adapter rejects a cross-stage selection role")
        row = dict(item)
        raw_by_role[role].append(row)
        state_by_identity[str(row["row_identity"])] = state
    if any(len(raw_by_role[role]) != count for role, count in role_counts.items()):
        raise ValueError("forensic feature stage role count mismatch")
    train_images = {row["original_image_sha256"] for row in raw_by_role[_fit_role]}
    validation_images = {row["original_image_sha256"] for row in raw_by_role[_transform_role]}
    if train_images & validation_images:
        raise ValueError("forensic feature roles must be exact-image disjoint")

    records: dict[str, list[QueryStateFeatureRecord]] = {}
    provenance: dict[str, dict[str, Mapping[str, Any]]] = {}
    for role, raw_rows in raw_by_role.items():
        split_identity = _role_identity(
            cache_fingerprint=cache_fingerprint,
            selection_identity=selection_identity,
            role=role,
            rows=raw_rows,
        )
        records[role] = []
        provenance[role] = {}
        for row in raw_rows:
            row_identity = str(row["row_identity"])
            if row.get("response_source") != "archived" or any(
                not _is_sha256(row.get(field)) for field in _PROVENANCE_FIELDS[:-1]
            ):
                raise ValueError("forensic feature row prompt/render/encoded provenance is invalid")
            records[role].append(
                QueryStateFeatureRecord(
                    row_identity=row_identity,
                    split=role,
                    image_path=str(row["original_image_path"]),
                    image_sha256=str(row["original_image_sha256"]),
                    archived_response_sha256=str(
                        row["archived_assistant_response_sha256"]
                    ),
                    bundle_fingerprint=cache_fingerprint,
                    source_jsonl_sha256=source_sha256_by_role[role],
                    source_manifest_identity=source_manifest_identity,
                    selection_role=role,
                    cache_split_identity=split_identity,
                    state=state_by_identity[row_identity],
                )
            )
            provenance[role][row_identity] = {
                "record_id": row["record_id"],
                "step_index": row["step_index"],
                "selection_ordinal": row["selection_ordinal"],
                "selection_role": role,
                **{field: row[field] for field in _PROVENANCE_FIELDS},
            }
    return manifest, records, provenance


def _expected_basis_identity(
    train_records: Sequence[QueryStateFeatureRecord],
    *,
    fit_role: str,
    dino_identity: str,
    interpolation: str,
) -> SharedFeatureBasisIdentity:
    first = train_records[0]
    return SharedFeatureBasisIdentity(
        method=NIMLOTH_SHARED_BASIS_METHOD,
        bundle_fingerprint=first.bundle_fingerprint,
        source_jsonl_sha256=first.source_jsonl_sha256,
        source_manifest_identity=first.source_manifest_identity,
        fit_split=fit_role,
        fit_split_identity=_feature_split_identity(
            train_records,
            source_manifest_identity=first.source_manifest_identity,
            expected_selection_role=fit_role,
        ),
        fit_row_set_identity=_feature_row_set_identity(train_records),
        dino_identity=dino_identity,
        state_shape=(16, 1024),
        interpolation=interpolation,
    )


def _basis_receipt_path(path: str | Path) -> Path:
    basis = Path(path)
    return basis.with_name(f"{basis.name}.forensic.json")


def _write_basis_receipt(
    basis_path: Path,
    *,
    basis: SharedFeatureBasis,
    manifest: Mapping[str, Any],
) -> None:
    receipt = _basis_receipt_path(basis_path)
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError(f"forensic basis receipt already exists: {receipt}")
    mechanics_only = (
        manifest["selection"]["stage"] == ForensicExperimentStage.MECHANICS_ONLY.value
    )
    payload = {
        "schema": _FORENSIC_BASIS_RECEIPT_SCHEMA,
        "basis_file": basis_path.name,
        "basis_file_sha256": _sha256_file(basis_path),
        "basis_artifact_sha256": basis.artifact_sha256,
        "global_scale_sha256": basis.global_scale_sha256,
        "cache_schema": FORENSIC_QUERY_STATE_CACHE_SCHEMA,
        "cache_fingerprint": manifest["cache_fingerprint"],
        "checkpoint_identity": manifest["checkpoint"],
        "selection_identity": manifest["selection"]["identity"],
        "fit_role": _stage_roles(str(manifest["selection"]["stage"]))[0],
        "transform_only_role": _stage_roles(str(manifest["selection"]["stage"]))[1],
        "forensic_only": True,
        "unsafe_actor_checkpoint": True,
        "not_deployable": True,
        "mechanics_only": mechanics_only,
        "not_heldout": mechanics_only,
        "formal38_calibration_80_aggregation_reproduced": False,
        "deep_sight_exact_colorization": False,
    }
    temporary = receipt.with_name(f".{receipt.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(receipt)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_basis_receipt(
    basis_path: str | Path,
    *,
    basis: SharedFeatureBasis,
    manifest: Mapping[str, Any],
) -> None:
    path = Path(basis_path)
    receipt_path = _basis_receipt_path(path)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("forensic basis receipt is missing or invalid") from error
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != _BASIS_RECEIPT_FIELDS
        or receipt.get("schema") != _FORENSIC_BASIS_RECEIPT_SCHEMA
        or receipt.get("basis_file") != path.name
        or receipt.get("basis_file_sha256") != _sha256_file(path)
        or receipt.get("basis_artifact_sha256") != basis.artifact_sha256
        or receipt.get("global_scale_sha256") != basis.global_scale_sha256
        or receipt.get("cache_schema") != FORENSIC_QUERY_STATE_CACHE_SCHEMA
        or receipt.get("cache_fingerprint") != manifest["cache_fingerprint"]
        or receipt.get("checkpoint_identity") != manifest["checkpoint"]
        or receipt.get("selection_identity") != manifest["selection"]["identity"]
        or receipt.get("fit_role") != _stage_roles(str(manifest["selection"]["stage"]))[0]
        or receipt.get("transform_only_role")
        != _stage_roles(str(manifest["selection"]["stage"]))[1]
        or any(
            receipt.get(field) is not True
            for field in ("forensic_only", "unsafe_actor_checkpoint", "not_deployable")
        )
        or receipt.get("mechanics_only") is not (
            manifest["selection"]["stage"] == ForensicExperimentStage.MECHANICS_ONLY.value
        )
        or receipt.get("not_heldout") is not (
            manifest["selection"]["stage"] == ForensicExperimentStage.MECHANICS_ONLY.value
        )
        or receipt.get("formal38_calibration_80_aggregation_reproduced") is not False
        or receipt.get("deep_sight_exact_colorization") is not False
    ):
        raise ValueError("forensic basis receipt identity/watermark mismatch")


def fit_forensic_shared_feature_basis(
    forensic_cache: str | Path,
    *,
    interpolation: str,
    output_path: str | Path,
    dino_device: torch.device,
    dino_dtype: torch.dtype,
    dino_batch_size: int,
) -> SharedFeatureBasis:
    """Fit one shared PCA/global scale from the stage's strict training role only."""

    manifest, records, _provenance = _strict_forensic_records(forensic_cache)
    output = Path(output_path)
    receipt = _basis_receipt_path(output)
    if output.exists() or output.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise FileExistsError("forensic basis or receipt output already exists")
    fit_role, _transform_role, _role_counts = _stage_roles(
        str(manifest["selection"]["stage"])
    )
    train = records[fit_role]
    teacher = _build_pinned_dino_teacher(
        device=dino_device, dtype=dino_dtype, batch_size=dino_batch_size
    )
    targets = extract_dino_feature_records(
        train, teacher=teacher, device=dino_device, batch_size=dino_batch_size
    )
    basis = _fit_shared_feature_basis_from_records(
        train,
        targets,
        interpolation=interpolation,
        output_path=output,
        fit_split=fit_role,
        expected_selection_role=fit_role,
    )
    try:
        _write_basis_receipt(output, basis=basis, manifest=manifest)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return basis


def _report_extension(manifest: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    mechanics_only = (
        manifest["selection"]["stage"] == ForensicExperimentStage.MECHANICS_ONLY.value
    )
    return {
        "report_schema": FORENSIC_FEATURE_REPORT_SCHEMA,
        "cache_schema": FORENSIC_QUERY_STATE_CACHE_SCHEMA,
        "cache_owner_role": FORENSIC_QUERY_STATE_OWNER_ROLE,
        "cache_fingerprint": manifest["cache_fingerprint"],
        "checkpoint_identity": manifest["checkpoint"],
        "selection_identity": manifest["selection"]["identity"],
        "selection_role": role,
        "forensic_only": True,
        "unsafe_actor_checkpoint": True,
        "not_deployable": True,
        "mechanics_only": mechanics_only,
        "not_heldout": mechanics_only,
        "formal38_calibration_80_aggregation_reproduced": False,
        "evidence_limit": (
            "Stage A train-derived mechanics diagnostics only; this does not reproduce "
            "Formal38 calibration-80 aggregation or DeepSight exact colorization."
            if mechanics_only
            else "Stage B full all_train/external_validation direct diagnostics; actor failure "
            "remains controlling and this is not DeepSight exact colorization."
        ),
    }


def render_forensic_query_state_feature_reports(
    *,
    forensic_cache: str | Path,
    basis_path: str | Path,
    output_dir: str | Path,
    interpolation: str,
    normalization: str,
    shuffle_seed: int,
    dino_device: torch.device,
    dino_dtype: torch.dtype,
    dino_batch_size: int,
) -> dict[str, Any]:
    """Compute both stage roles without validation refit; bound Stage B visuals."""

    manifest, records, provenance = _strict_forensic_records(forensic_cache)
    teacher = _build_pinned_dino_teacher(
        device=dino_device, dtype=dino_dtype, batch_size=dino_batch_size
    )
    fit_role, transform_role, _role_counts = _stage_roles(
        str(manifest["selection"]["stage"])
    )
    expected = _expected_basis_identity(
        records[fit_role],
        fit_role=fit_role,
        dino_identity=dino_feature_identity(teacher),
        interpolation=interpolation,
    )
    basis = load_shared_feature_basis(basis_path, expected_identity=expected)
    _validate_basis_receipt(basis_path, basis=basis, manifest=manifest)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"forensic feature report output already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    reports: dict[str, Any] = {}
    try:
        for role in (fit_role, transform_role):
            targets = extract_dino_feature_records(
                records[role],
                teacher=teacher,
                device=dino_device,
                batch_size=dino_batch_size,
            )
            stage_b = manifest["selection"]["stage"] == ForensicExperimentStage.STAGE_B_DIAGNOSTIC.value
            visual_selection: list[int] | None = None
            if stage_b and role == fit_role:
                role_dir = temporary / role
                role_dir.mkdir()
                report = {
                    "metadata": _report_extension(manifest, role=role),
                    "metrics": aggregate_direct_feature_metrics(
                        torch.stack([record.state for record in records[role]]),
                        torch.stack([target.features for target in targets]),
                        shuffle_seed=shuffle_seed,
                    ),
                    "rows": [],
                    "contact_sheet": None,
                    "full_statistical_count": len(records[role]),
                    "visual_selection": None,
                }
                (role_dir / "report.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
            else:
                render_records = records[role]
                render_targets = targets
                if stage_b:
                    visual_selection = sorted(
                        range(len(render_records)),
                        key=lambda index: hashlib.sha256(
                            f"{_STAGE_B_VISUAL_SEED}:{render_records[index].row_identity}".encode()
                        ).hexdigest(),
                    )[:_STAGE_B_VISUAL_COUNT]
                    render_records = [render_records[index] for index in visual_selection]
                    render_targets = [targets[index] for index in visual_selection]
                report = _render_query_state_feature_report_from_records(
                    render_records,
                    render_targets,
                    basis=basis,
                    output_dir=temporary / role,
                    interpolation=interpolation,
                    normalization=normalization,
                    shuffle_seed=shuffle_seed,
                    authoritative_provenance=False,
                    expected_selection_role=role,
                    metadata_extension=_report_extension(manifest, role=role),
                    row_metadata_extension={
                        row.row_identity: provenance[role][row.row_identity]
                        for row in render_records
                    },
                )
                if visual_selection is not None:
                    report["metrics"] = aggregate_direct_feature_metrics(
                        torch.stack([record.state for record in records[role]]),
                        torch.stack([target.features for target in targets]),
                        shuffle_seed=shuffle_seed,
                    )
                    report["full_statistical_count"] = len(records[role])
                    report["visual_selection"] = {
                        "algorithm": "sha256_row_identity_sample_v1",
                        "seed": _STAGE_B_VISUAL_SEED,
                        "indices": visual_selection,
                        "row_identities": [records[role][index].row_identity for index in visual_selection],
                    }
                    (temporary / role / "report.json").write_text(
                        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
            reports[role] = report
        mechanics_only = (
            manifest["selection"]["stage"] == ForensicExperimentStage.MECHANICS_ONLY.value
        )
        summary = {
            "schema": FORENSIC_FEATURE_REPORT_SCHEMA,
            "experiment_stage": manifest["selection"]["stage"],
            "watermarks": {
                "forensic_only": True,
                "unsafe_actor_checkpoint": True,
                "not_deployable": True,
                "mechanics_only": mechanics_only,
                "not_heldout": mechanics_only,
            },
            "cache_fingerprint": manifest["cache_fingerprint"],
            "basis_sha256": basis.artifact_sha256,
            "global_scale_sha256": basis.global_scale_sha256,
            "roles": {
                role: {
                    "count": len(records[role]),
                    "visual_count": len(reports[role]["rows"]),
                    "visual_selection": reports[role].get("visual_selection"),
                    "metrics": reports[role]["metrics"],
                    "report": f"{role}/report.json",
                    "contact_sheet": (
                        f"{role}/{reports[role]['contact_sheet']}"
                        if reports[role]["contact_sheet"] is not None else None
                    ),
                }
                for role in reports
            },
            "mechanics_validation_controls_pass_or_checkpoint_selection": False,
            "external_validation_controls_checkpoint_selection": False,
            "formal38_calibration_80_aggregation_reproduced": False,
            "deep_sight_exact_colorization": False,
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--forensic-cache", type=Path, required=True)
    common.add_argument("--dino-device", choices=("cpu", "cuda"), required=True)
    common.add_argument("--dino-dtype", choices=("float32", "float16", "bfloat16"), required=True)
    common.add_argument("--dino-batch-size", type=int, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit-basis", parents=[common])
    fit.add_argument("--interpolation", choices=("nearest", "bilinear", "bicubic"), required=True)
    fit.add_argument("--output", type=Path, required=True)
    render = commands.add_parser("render-report", parents=[common])
    render.add_argument("--basis", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--interpolation", choices=("nearest", "bilinear", "bicubic"), required=True)
    render.add_argument("--normalization", choices=("shared_global",), required=True)
    render.add_argument("--shuffle-seed", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    common = {
        "forensic_cache": args.forensic_cache,
        "dino_device": torch.device(args.dino_device),
        "dino_dtype": _dtype(args.dino_dtype),
        "dino_batch_size": args.dino_batch_size,
    }
    if args.command == "fit-basis":
        basis = fit_forensic_shared_feature_basis(
            **common,
            interpolation=args.interpolation,
            output_path=args.output,
        )
        result = {"basis_sha256": basis.artifact_sha256, "forensic_only": True}
    else:
        result = render_forensic_query_state_feature_reports(
            **common,
            basis_path=args.basis,
            output_dir=args.output_dir,
            interpolation=args.interpolation,
            normalization=args.normalization,
            shuffle_seed=args.shuffle_seed,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "FORENSIC_FEATURE_REPORT_SCHEMA",
    "FORMAL38_FORENSIC_SOURCE_COMMIT",
    "FORMAL38_UNSAFE_UPDATE1605_CONTROL_SHA256",
    "fit_forensic_shared_feature_basis",
    "render_forensic_query_state_feature_reports",
]
