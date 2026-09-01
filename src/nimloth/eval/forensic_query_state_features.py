"""Strict direct-feature diagnostics for Formal38 unsafe update1605.

This entry point is deliberately separate from the deployable Query-State feature
CLI.  It accepts only the Stage A forensic cache owner and preserves the actor
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
    dino_feature_identity,
    extract_dino_feature_records,
    load_shared_feature_basis,
)
from nimloth.training.reconstruction.forensic_query_state_cache import (
    FORENSIC_QUERY_STATE_CACHE_SCHEMA,
    FORENSIC_QUERY_STATE_OWNER_ROLE,
    FORENSIC_SELECTION_MECHANICS_TRAIN,
    FORENSIC_SELECTION_MECHANICS_VALIDATION,
    ForensicQueryStateCacheDataset,
)

FORMAL38_FORENSIC_SOURCE_COMMIT = "4838e5fdb469dffb78909e307cf11a808cb2d29e"
FORMAL38_UNSAFE_UPDATE1605_CONTROL_SHA256 = (
    "414daefe2b501a22805691aa101d76fcc0f5b28447a1332d81b19b3434e838af"
)
FORENSIC_FEATURE_REPORT_SCHEMA = "nimloth_forensic_query_state_feature_report_v1"
_FORENSIC_BASIS_RECEIPT_SCHEMA = "nimloth_forensic_query_state_feature_basis_receipt_v1"
_STAGE = "mechanics_only"
_ROLE_COUNTS = {
    FORENSIC_SELECTION_MECHANICS_TRAIN: 48,
    FORENSIC_SELECTION_MECHANICS_VALIDATION: 16,
}
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
        or selection.get("stage") != _STAGE
        or selection.get("roles") != _ROLE_COUNTS
    ):
        raise ValueError(
            "forensic feature adapter requires exact Formal38 unsafe update1605 "
            "checkpoint/failure evidence and Stage A mechanics roles"
        )
    cache_fingerprint = str(manifest.get("cache_fingerprint"))
    source_manifest_identity = str(manifest["source_jsonl"]["source_manifest_identity"])
    source_sha256 = str(manifest["source_jsonl"]["train"]["sha256"])
    selection_identity = str(selection["identity"])
    if any(
        not _is_sha256(value)
        for value in (
            cache_fingerprint,
            source_manifest_identity,
            source_sha256,
            selection_identity,
        )
    ):
        raise ValueError("forensic feature cache/source/selection identity is invalid")

    raw_by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in _ROLE_COUNTS}
    state_by_identity: dict[str, torch.Tensor] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        state = item.pop("state")
        role = str(item.get("selection_role"))
        if role not in raw_by_role:
            raise ValueError("forensic feature adapter rejects non-Stage-A selection role")
        row = dict(item)
        raw_by_role[role].append(row)
        state_by_identity[str(row["row_identity"])] = state
    if any(len(raw_by_role[role]) != count for role, count in _ROLE_COUNTS.items()):
        raise ValueError("forensic mechanics_train/mechanics_validation count mismatch")
    train_images = {
        row["original_image_sha256"]
        for row in raw_by_role[FORENSIC_SELECTION_MECHANICS_TRAIN]
    }
    validation_images = {
        row["original_image_sha256"]
        for row in raw_by_role[FORENSIC_SELECTION_MECHANICS_VALIDATION]
    }
    if train_images & validation_images:
        raise ValueError("forensic Stage A mechanics roles must be exact-image disjoint")

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
                    source_jsonl_sha256=source_sha256,
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
    dino_identity: str,
    interpolation: str,
) -> SharedFeatureBasisIdentity:
    first = train_records[0]
    return SharedFeatureBasisIdentity(
        method=NIMLOTH_SHARED_BASIS_METHOD,
        bundle_fingerprint=first.bundle_fingerprint,
        source_jsonl_sha256=first.source_jsonl_sha256,
        source_manifest_identity=first.source_manifest_identity,
        fit_split=FORENSIC_SELECTION_MECHANICS_TRAIN,
        fit_split_identity=_feature_split_identity(
            train_records,
            source_manifest_identity=first.source_manifest_identity,
            expected_selection_role=FORENSIC_SELECTION_MECHANICS_TRAIN,
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
        "fit_role": FORENSIC_SELECTION_MECHANICS_TRAIN,
        "transform_only_role": FORENSIC_SELECTION_MECHANICS_VALIDATION,
        "forensic_only": True,
        "unsafe_actor_checkpoint": True,
        "not_deployable": True,
        "mechanics_only": True,
        "not_heldout": True,
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
        or receipt.get("fit_role") != FORENSIC_SELECTION_MECHANICS_TRAIN
        or receipt.get("transform_only_role")
        != FORENSIC_SELECTION_MECHANICS_VALIDATION
        or any(
            receipt.get(field) is not True
            for field in (
                "forensic_only", "unsafe_actor_checkpoint", "not_deployable",
                "mechanics_only", "not_heldout",
            )
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
    """Fit one shared PCA/global scale from mechanics_train only."""

    manifest, records, _provenance = _strict_forensic_records(forensic_cache)
    output = Path(output_path)
    receipt = _basis_receipt_path(output)
    if output.exists() or output.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise FileExistsError("forensic basis or receipt output already exists")
    train = records[FORENSIC_SELECTION_MECHANICS_TRAIN]
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
        fit_split=FORENSIC_SELECTION_MECHANICS_TRAIN,
        expected_selection_role=FORENSIC_SELECTION_MECHANICS_TRAIN,
    )
    try:
        _write_basis_receipt(output, basis=basis, manifest=manifest)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return basis


def _report_extension(manifest: Mapping[str, Any], *, role: str) -> dict[str, Any]:
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
        "mechanics_only": True,
        "not_heldout": True,
        "formal38_calibration_80_aggregation_reproduced": False,
        "evidence_limit": (
            "Stage A train-derived mechanics diagnostics only; this does not reproduce "
            "Formal38 calibration-80 aggregation or DeepSight exact colorization."
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
    """Render direct metrics/maps for both Stage A roles without validation refit."""

    manifest, records, provenance = _strict_forensic_records(forensic_cache)
    teacher = _build_pinned_dino_teacher(
        device=dino_device, dtype=dino_dtype, batch_size=dino_batch_size
    )
    expected = _expected_basis_identity(
        records[FORENSIC_SELECTION_MECHANICS_TRAIN],
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
        for role in (
            FORENSIC_SELECTION_MECHANICS_TRAIN,
            FORENSIC_SELECTION_MECHANICS_VALIDATION,
        ):
            targets = extract_dino_feature_records(
                records[role],
                teacher=teacher,
                device=dino_device,
                batch_size=dino_batch_size,
            )
            reports[role] = _render_query_state_feature_report_from_records(
                records[role],
                targets,
                basis=basis,
                output_dir=temporary / role,
                interpolation=interpolation,
                normalization=normalization,
                shuffle_seed=shuffle_seed,
                authoritative_provenance=False,
                expected_selection_role=role,
                metadata_extension=_report_extension(manifest, role=role),
                row_metadata_extension=provenance[role],
            )
        summary = {
            "schema": FORENSIC_FEATURE_REPORT_SCHEMA,
            "watermarks": {
                "forensic_only": True,
                "unsafe_actor_checkpoint": True,
                "not_deployable": True,
                "mechanics_only": True,
                "not_heldout": True,
            },
            "cache_fingerprint": manifest["cache_fingerprint"],
            "basis_sha256": basis.artifact_sha256,
            "global_scale_sha256": basis.global_scale_sha256,
            "roles": {
                role: {
                    "count": len(reports[role]["rows"]),
                    "metrics": reports[role]["metrics"],
                    "report": f"{role}/report.json",
                    "contact_sheet": f"{role}/{reports[role]['contact_sheet']}",
                }
                for role in reports
            },
            "mechanics_validation_controls_pass_or_checkpoint_selection": False,
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
