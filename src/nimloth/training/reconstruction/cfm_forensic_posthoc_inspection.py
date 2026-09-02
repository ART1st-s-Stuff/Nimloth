"""Strict non-publication RGB inspection for the gate-failed Stage B decoder.

This owner is intentionally read-only: it binds the exact failed Job543457
artifacts, loads only the decoder model, and samples one fixed correct-condition
product.  It cannot train, resume, override the gate, or create a W&B run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from nimloth.recon.cfm import CFMConfig, TokenConditionedFlowUNet, sample_euler
from nimloth.training.reconstruction.cfm_forensic_query_state import (
    FORENSIC_CFM_CHECKPOINT_SCHEMA,
    FORENSIC_CFM_RGB_ARTIFACT_SCHEMA,
    FORENSIC_STAGE_B_EXPERIMENT_STAGE,
    FORENSIC_STAGE_B_FINAL_STEP,
    FORMAL38_SOURCE_COMMIT,
    _is_sha256,
    _sha256_file,
    _sha256_mapping,
    _strip,
    _tensor_image,
    _uint8_image,
    _validate_invariants,
    build_forensic_stage_b_sample_plan,
    load_forensic_image_splits,
    validate_forensic_manifest,
)
from nimloth.training.reconstruction.cfm_query_state import (
    _load_image_uint8,
    flatten_query_state_condition,
)

FORENSIC_POSTHOC_RGB_INSPECTION_SCHEMA = (
    "nimloth_query_state_forensic_cfm_posthoc_rgb_inspection_v1"
)
STAGE_B_RUN_SOURCE_COMMIT = "cd1c002358b6b78e4607a1c7e5ecad6dad3b0e86"
FINAL_CHECKPOINT_SHA256 = "52bf18e22aba3dd5055a51b07c94c4488ded9a5134df87c48b0818cb31798929"
SUMMARY_SHA256 = "07f8cfe806af7f6b1b35b08b1142df86f98961eeb5844cae9691ffae1d81f565"
FAILED_GATE_METADATA_SHA256 = "8b7a2252d0f9f46229a9314fc8ca6faf4bbd9e661df0a2111467c4888e5e691a"
STAGE_B_CACHE_MANIFEST_SHA256 = "76b36e10edd666136ccaa115f14b2fa36156a0bcd56e94b9d1425f6f4dc1d083"
STAGE_B_CACHE_FINGERPRINT = "9bd942267140aede839087b09bb0f755bd023ec69d216a8ad77ee845fc120899"
STAGE_B_SELECTION_IDENTITY = "4227fe24dd88b0948ca08e7c45de77fcd8d611540587e9bbb2549bc083ceebf2"
STAGE_B_SAMPLE_INDICES = (
    424, 1245, 240, 761, 1360, 214, 191, 389,
    84, 3, 182, 45, 246, 1255, 1060, 257,
)
STAGE_B_SAMPLE_INDICES_SHA256 = "55257d76ab8f2dfb12aeb0bf9722fb2fa326be545832d9d119df36dc06015bef"
STAGE_B_INITIAL_NOISE_SHA256 = "7390e403b9d92922fa94fd53b0e6b1fd09df3da002c397a89bca03e68d718575"
POSTHOC_WATERMARKS = (
    "posthoc_human_inspection",
    "publication_gate_failed",
    "not_publication",
    "unsafe_actor_checkpoint",
    "not_deployable",
)
_MANIFEST_FIELDS = {
    "schema", "status", "watermarks", "stage_b_run_source_commit",
    "formal38_source_commit", "decoder_checkpoint_sha256",
    "decoder_checkpoint_step", "summary_sha256", "failed_gate_metadata_sha256",
    "cache_manifest_sha256", "cache_fingerprint", "selection_identity",
    "publication_gate", "correct_condition_only", "optimizer_constructed_or_loaded",
    "training_or_resume_performed", "wandb_enabled", "color_space", "channels",
    "image_size", "ode_steps", "sample_chunk_size", "sample_seed",
    "sample_selection", "source_identity", "checkpoint_invariants_sha256",
    "original_job543457_snapshot", "rows", "contact_sheet_path",
    "contact_sheet_png_sha256", "artifact_identity",
}
_ROW_FIELDS = {
    "ordinal", "selection_index", "row_identity", "source_original_image_path",
    "source_original_image_sha256", "original_path", "original_png_sha256",
    "reconstruction_path", "reconstruction_png_sha256", "strip_path",
    "strip_png_sha256",
}


class ForensicPosthocInspectionDurabilityError(RuntimeError):
    """The inspection manifest committed but final durability is unconfirmed."""


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_png_fsynced(image: Image.Image, path: Path) -> None:
    with path.open("xb") as stream:
        image.save(stream, format="PNG")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_fsynced(value: Mapping[str, Any], path: Path) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _snapshot_regular_files(
    root: Path,
) -> tuple[dict[str, Mapping[str, str]], str]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("original Job543457 output root is invalid")
    result: dict[str, Mapping[str, str]] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            result[relative] = {"type": "symlink", "target": os.readlink(path)}
        elif path.is_file():
            result[relative] = {"type": "file", "sha256": _sha256_file(path)}
        elif not path.is_dir():
            raise ValueError("original Job543457 output contains an unsupported entry")
    if not result:
        raise ValueError("original Job543457 output is empty")
    return result, _sha256_mapping({"entries": result})


def _publish_inspection_noreplace(source: Path, destination: Path) -> None:
    """Publish by durable claim and manifest-last commit without overwrite."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"post-hoc inspection output already exists: {destination}")
    manifest = source / "manifest.json"
    if source.is_symlink() or not source.is_dir() or not manifest.is_file() or manifest.is_symlink():
        raise ValueError("post-hoc inspection staging requires a regular manifest.json")
    payloads = sorted(path for path in source.iterdir() if path.name != "manifest.json")
    if not payloads or any(path.is_symlink() for path in payloads):
        raise ValueError("post-hoc inspection staging payload is invalid")
    for payload in payloads:
        entries = (payload, *payload.rglob("*")) if payload.is_dir() else (payload,)
        if any(entry.is_symlink() or (not entry.is_dir() and not entry.is_file()) for entry in entries):
            raise ValueError("post-hoc inspection staging contains an invalid payload entry")
    try:
        destination.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"post-hoc inspection output already exists: {destination}") from error
    _fsync_directory(destination.parent)
    committed = False
    try:
        for payload in payloads:
            os.rename(payload, destination / payload.name)
        _fsync_directory(destination)
        os.rename(manifest, destination / "manifest.json")
        committed = True
        _fsync_directory(destination)
        source.rmdir()
        _fsync_directory(destination.parent)
    except BaseException as error:
        if committed:
            raise ForensicPosthocInspectionDurabilityError(
                "post-hoc inspection manifest committed but publication durability was not confirmed"
            ) from error
        raise


def _validate_exact_inputs(
    *,
    decoder_checkpoint: Path,
    cache_dir: Path,
    summary_path: Path,
    failed_gate_metadata_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    manifest_path = cache_dir / "manifest.json"
    paths = (decoder_checkpoint, summary_path, failed_gate_metadata_path, manifest_path)
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("post-hoc inspection inputs must be regular exact files")
    if _sha256_file(decoder_checkpoint) != FINAL_CHECKPOINT_SHA256:
        raise ValueError("post-hoc inspection final step4000 checkpoint SHA256 mismatch")
    if _sha256_file(summary_path) != SUMMARY_SHA256:
        raise ValueError("post-hoc inspection Job543457 summary SHA256 mismatch")
    if _sha256_file(failed_gate_metadata_path) != FAILED_GATE_METADATA_SHA256:
        raise ValueError("post-hoc inspection failed-gate metadata SHA256 mismatch")
    if _sha256_file(manifest_path) != STAGE_B_CACHE_MANIFEST_SHA256:
        raise ValueError("post-hoc inspection Stage B cache manifest SHA256 mismatch")

    summary = _read_json_mapping(summary_path, label="Job543457 summary")
    failed = _read_json_mapping(failed_gate_metadata_path, label="Job543457 failed-gate metadata")
    cache_manifest = _read_json_mapping(manifest_path, label="Stage B cache manifest")
    validate_forensic_manifest(
        cache_manifest, experiment_stage=FORENSIC_STAGE_B_EXPERIMENT_STAGE
    )
    selection = cache_manifest.get("selection")
    if (
        cache_manifest.get("cache_fingerprint") != STAGE_B_CACHE_FINGERPRINT
        or not isinstance(selection, Mapping)
        or selection.get("identity") != STAGE_B_SELECTION_IDENTITY
    ):
        raise ValueError("post-hoc inspection cache fingerprint/selection identity mismatch")
    failed_identity = failed.get("artifact_identity")
    if (
        failed.get("schema") != FORENSIC_CFM_RGB_ARTIFACT_SCHEMA
        or failed.get("status") != "publication_gate_failed"
        or failed.get("experiment_stage") != FORENSIC_STAGE_B_EXPERIMENT_STAGE
        or Path(str(failed.get("decoder_checkpoint", ""))).resolve() != decoder_checkpoint.resolve()
        or failed.get("decoder_checkpoint_sha256") != FINAL_CHECKPOINT_SHA256
        or failed.get("decoder_checkpoint_step") != FORENSIC_STAGE_B_FINAL_STEP
        or failed.get("cache_fingerprint") != STAGE_B_CACHE_FINGERPRINT
        or failed.get("rgb_artifacts_published") is not False
        or not isinstance(failed.get("publication_gate"), Mapping)
        or failed["publication_gate"].get("passed") is not False
        or not _is_sha256(failed_identity)
        or _sha256_mapping({key: value for key, value in failed.items() if key != "artifact_identity"}) != failed_identity
    ):
        raise ValueError("post-hoc inspection requires exact valid gate-failed Stage B metadata")
    rgb_artifact = summary.get("rgb_artifact")
    if (
        summary.get("status") != "publication_gate_failed"
        or summary.get("additional_steps_or_checkpoint_fallback_allowed") is not False
        or Path(str(summary.get("final_checkpoint", ""))).resolve() != decoder_checkpoint.resolve()
        or not isinstance(rgb_artifact, Mapping)
        or Path(str(rgb_artifact.get("metadata_path", ""))).resolve() != failed_gate_metadata_path.resolve()
        or rgb_artifact.get("publication_gate") != failed.get("publication_gate")
    ):
        raise ValueError("post-hoc inspection Job543457 summary/failure binding mismatch")
    cfm_root = summary_path.resolve().parent
    if (
        cfm_root.name != "cfm"
        or cfm_root != decoder_checkpoint.resolve().parent
        or failed_gate_metadata_path.resolve().parent != cfm_root / "rgb_samples"
    ):
        raise ValueError("post-hoc inspection inputs are not owned by one exact Job543457 output")
    launch_metadata = _read_json_mapping(
        cfm_root.parent / "launch_metadata.json", label="Job543457 launch metadata"
    )
    if (
        launch_metadata.get("schema") != "nimloth_formal38_forensic_stage_b_cfm_run_v1"
        or launch_metadata.get("source_commit") != STAGE_B_RUN_SOURCE_COMMIT
        or launch_metadata.get("cache_fingerprint") != STAGE_B_CACHE_FINGERPRINT
    ):
        raise ValueError("post-hoc inspection Job543457 source/cache launch identity mismatch")

    try:
        payload = torch.load(decoder_checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("post-hoc inspection decoder checkpoint is unreadable") from error
    required = {
        "schema", "model", "optimizer", "step", "invariants",
        "torch_rng_state", "cuda_rng_state_all",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload.get("schema") != FORENSIC_CFM_CHECKPOINT_SCHEMA
        or payload.get("step") != FORENSIC_STAGE_B_FINAL_STEP
        or not isinstance(payload.get("model"), Mapping)
        or not isinstance(payload.get("optimizer"), Mapping)
        or not isinstance(payload.get("invariants"), Mapping)
        or not isinstance(payload.get("torch_rng_state"), torch.Tensor)
        or payload["torch_rng_state"].ndim != 1
        or payload["torch_rng_state"].dtype != torch.uint8
        or not (
            payload.get("cuda_rng_state_all") is None
            or (
                isinstance(payload.get("cuda_rng_state_all"), list)
                and all(
                    isinstance(state, torch.Tensor)
                    and state.ndim == 1
                    and state.dtype == torch.uint8
                    for state in payload["cuda_rng_state_all"]
                )
            )
        )
    ):
        raise ValueError("post-hoc inspection requires the exact final forensic decoder checkpoint")
    invariants = payload["invariants"]
    config_value = invariants.get("cfm_config")
    try:
        config = CFMConfig(**dict(config_value)) if isinstance(config_value, Mapping) else None
    except (TypeError, ValueError) as error:
        raise ValueError("post-hoc inspection checkpoint CFM config is invalid") from error
    if config is None:
        raise ValueError("post-hoc inspection checkpoint CFM config is missing")
    _validate_invariants(invariants, config)
    optimizer_state = payload["optimizer"]
    param_groups = optimizer_state.get("param_groups") if isinstance(optimizer_state, Mapping) else None
    if (
        set(optimizer_state) != {"state", "param_groups"}
        or not isinstance(optimizer_state.get("state"), Mapping)
        or not isinstance(param_groups, list)
        or len(param_groups) != 1
        or not isinstance(param_groups[0], Mapping)
        or float(param_groups[0].get("lr", float("nan"))) != float(invariants["learning_rate"])
        or float(param_groups[0].get("weight_decay", float("nan"))) != float(invariants["weight_decay"])
        or not isinstance(param_groups[0].get("params"), list)
        or not param_groups[0]["params"]
        or len(set(param_groups[0]["params"])) != len(param_groups[0]["params"])
        or not set(optimizer_state["state"]).issubset(set(param_groups[0]["params"]))
    ):
        raise ValueError("post-hoc inspection checkpoint optimizer evidence is invalid")
    expected = {
        "experiment_stage": FORENSIC_STAGE_B_EXPERIMENT_STAGE,
        "cache_fingerprint": STAGE_B_CACHE_FINGERPRINT,
        "validation_items": 1_413,
        "validation_role": "external_validation",
        "max_steps": FORENSIC_STAGE_B_FINAL_STEP,
        "sample_items": 16,
        "sample_ode_steps": 50,
        "sample_noise_seed": 20260921,
        "sample_batch_size": 8,
    }
    if (
        any(invariants.get(key) != value for key, value in expected.items())
        or failed.get("source_identity") != invariants.get("source_identity")
    ):
        raise ValueError("post-hoc inspection checkpoint sampling/cache/source invariants mismatch")
    return payload, invariants, summary, failed


def sample_forensic_stage_b_posthoc_rgb_inspection(
    *,
    output_dir: str | Path,
    decoder_checkpoint: str | Path,
    cache_dir: str | Path,
    summary_path: str | Path,
    failed_gate_metadata_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Sample the exact correct condition without touching the failed run."""

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"post-hoc inspection output already exists: {destination}")
    checkpoint_path = Path(decoder_checkpoint)
    cache_root = Path(cache_dir)
    summary_file = Path(summary_path)
    failed_file = Path(failed_gate_metadata_path)
    payload, invariants, _, _ = _validate_exact_inputs(
        decoder_checkpoint=checkpoint_path,
        cache_dir=cache_root,
        summary_path=summary_file,
        failed_gate_metadata_path=failed_file,
    )
    original_root = summary_file.resolve().parent.parent
    try:
        destination.resolve().relative_to(original_root)
    except ValueError:
        pass
    else:
        raise ValueError("post-hoc inspection output must be outside original Job543457 output")
    original_files, original_snapshot = _snapshot_regular_files(original_root)

    _, validation, source_identity = load_forensic_image_splits(
        cache_root,
        image_size=128,
        experiment_stage=FORENSIC_STAGE_B_EXPERIMENT_STAGE,
    )
    expected_loaded = {
        "cache_fingerprint": validation.cache_fingerprint,
        "validation_split_identity": validation.split_identity,
        "validation_row_set_identity": validation.row_set_identity,
        "validation_items": len(validation),
        "image_preprocessing": dict(validation.image_preprocessing),
        "source_identity": dict(source_identity),
    }
    if any(invariants.get(key) != value for key, value in expected_loaded.items()):
        raise ValueError("post-hoc inspection live cache/checkpoint identity mismatch")
    plan = build_forensic_stage_b_sample_plan(validation)
    if (
        plan.indices != STAGE_B_SAMPLE_INDICES
        or plan.indices_sha256 != STAGE_B_SAMPLE_INDICES_SHA256
        or plan.initial_noise_sha256 != STAGE_B_INITIAL_NOISE_SHA256
    ):
        raise ValueError("post-hoc inspection deterministic Stage B sample contract drifted")
    selected = torch.tensor(plan.indices, dtype=torch.long)
    states = validation.states[selected].detach().cpu().float().contiguous()

    config = CFMConfig(**dict(invariants["cfm_config"]))
    decoder = TokenConditionedFlowUNet(config).to(device)
    serialized_parameters = payload["optimizer"]["param_groups"][0]["params"]
    if len(serialized_parameters) != len(tuple(decoder.parameters())):
        raise ValueError("post-hoc inspection serialized optimizer/model ownership mismatch")
    try:
        decoder.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError, ValueError, KeyError) as error:
        raise ValueError("post-hoc inspection decoder model state is invalid") from error
    decoder.eval().requires_grad_(False)
    with torch.inference_mode():
        reconstructions = sample_euler(
            decoder,
            flatten_query_state_condition(states),
            plan.initial_noise,
            steps=50,
            device=device,
            chunk_size=8,
        )
    if (
        not isinstance(reconstructions, torch.Tensor)
        or reconstructions.shape != (16, 3, 128, 128)
        or not torch.isfinite(reconstructions).all()
    ):
        raise ValueError("post-hoc inspection reconstruction output is invalid")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent))
    strips: list[Image.Image] = []
    try:
        originals_dir = staging / "originals"
        reconstructions_dir = staging / "reconstructions"
        strips_dir = staging / "strips"
        for directory in (originals_dir, reconstructions_dir, strips_dir):
            directory.mkdir()
        rows: list[dict[str, Any]] = []
        for ordinal, (selection_index, row) in enumerate(zip(plan.indices, plan.rows, strict=True)):
            source_path = Path(str(row.get("original_image_path", "")))
            source_sha = row.get("original_image_sha256")
            if (
                not source_path.is_absolute()
                or not source_path.is_file()
                or source_path.is_symlink()
                or not _is_sha256(source_sha)
                or _sha256_file(source_path) != source_sha
            ):
                raise ValueError("post-hoc inspection original image identity mismatch")
            original_tensor = _load_image_uint8(source_path, 128)
            original = _uint8_image(original_tensor)
            reconstruction = _tensor_image(reconstructions[ordinal])
            row_identity = str(row["row_identity"])
            strip = _strip(original, reconstruction, row_identity)
            stem = f"row_{ordinal:05d}_{hashlib.sha256(row_identity.encode()).hexdigest()[:12]}"
            relative_original = Path("originals") / f"{stem}.png"
            relative_reconstruction = Path("reconstructions") / f"{stem}.png"
            relative_strip = Path("strips") / f"{stem}_strip.png"
            _save_png_fsynced(original, staging / relative_original)
            _save_png_fsynced(reconstruction, staging / relative_reconstruction)
            _save_png_fsynced(strip, staging / relative_strip)
            strips.append(strip)
            rows.append({
                "ordinal": ordinal,
                "selection_index": selection_index,
                "row_identity": row_identity,
                "source_original_image_path": str(source_path),
                "source_original_image_sha256": source_sha,
                "original_path": str(relative_original),
                "original_png_sha256": _sha256_file(staging / relative_original),
                "reconstruction_path": str(relative_reconstruction),
                "reconstruction_png_sha256": _sha256_file(staging / relative_reconstruction),
                "strip_path": str(relative_strip),
                "strip_png_sha256": _sha256_file(staging / relative_strip),
            })
            original.close()
            reconstruction.close()
        contact = Image.new("RGB", (256, sum(strip.height for strip in strips)), "white")
        offset = 0
        for strip in strips:
            contact.paste(strip, (0, offset))
            offset += strip.height
        contact_path = staging / "contact_sheet.png"
        _save_png_fsynced(contact, contact_path)
        contact.close()
        for directory in (originals_dir, reconstructions_dir, strips_dir, staging):
            _fsync_directory(directory)

        manifest: dict[str, Any] = {
            "schema": FORENSIC_POSTHOC_RGB_INSPECTION_SCHEMA,
            "status": "completed_posthoc_human_inspection",
            "watermarks": list(POSTHOC_WATERMARKS),
            "stage_b_run_source_commit": STAGE_B_RUN_SOURCE_COMMIT,
            "formal38_source_commit": FORMAL38_SOURCE_COMMIT,
            "decoder_checkpoint_sha256": FINAL_CHECKPOINT_SHA256,
            "decoder_checkpoint_step": FORENSIC_STAGE_B_FINAL_STEP,
            "summary_sha256": SUMMARY_SHA256,
            "failed_gate_metadata_sha256": FAILED_GATE_METADATA_SHA256,
            "cache_manifest_sha256": STAGE_B_CACHE_MANIFEST_SHA256,
            "cache_fingerprint": STAGE_B_CACHE_FINGERPRINT,
            "selection_identity": STAGE_B_SELECTION_IDENTITY,
            "publication_gate": {"passed": False, "verdict_unchanged": True},
            "correct_condition_only": True,
            "optimizer_constructed_or_loaded": False,
            "training_or_resume_performed": False,
            "wandb_enabled": False,
            "color_space": "sRGB",
            "channels": 3,
            "image_size": 128,
            "ode_steps": 50,
            "sample_chunk_size": 8,
            "sample_seed": 20260921,
            "sample_selection": {
                "algorithm": "torch_randperm_cpu_v1",
                "indices": list(plan.indices),
                "indices_sha256": plan.indices_sha256,
                "row_identities_sha256": plan.row_identities_sha256,
                "original_image_sha256_identity": plan.original_image_sha256_identity,
                "initial_noise_sha256": plan.initial_noise_sha256,
            },
            "source_identity": dict(source_identity),
            "checkpoint_invariants_sha256": _sha256_mapping(dict(invariants)),
            "original_job543457_snapshot": {
                "file_count": len(original_files),
                "identity": original_snapshot,
                "unchanged_after_sampling": True,
            },
            "rows": rows,
            "contact_sheet_path": "contact_sheet.png",
            "contact_sheet_png_sha256": _sha256_file(contact_path),
        }
        manifest["artifact_identity"] = _sha256_mapping(manifest)
        _write_json_fsynced(manifest, staging / "manifest.json")
        _fsync_directory(staging)
        after_files, after_snapshot = _snapshot_regular_files(original_root)
        if after_files != original_files or after_snapshot != original_snapshot:
            raise RuntimeError("post-hoc inspection modified original Job543457 output")
        _publish_inspection_noreplace(staging, destination)
        validated = load_forensic_posthoc_rgb_inspection(destination)
        final_files, final_snapshot = _snapshot_regular_files(original_root)
        if final_files != original_files or final_snapshot != original_snapshot:
            raise RuntimeError("post-hoc inspection modified original Job543457 output")
        return {
            "manifest_path": str(destination / "manifest.json"),
            "artifact_identity": validated["artifact_identity"],
            "contact_sheet_path": str(destination / "contact_sheet.png"),
            "strip_paths": [str(destination / row["strip_path"]) for row in validated["rows"]],
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        for strip in strips:
            strip.close()


def _validated_relative_file(root: Path, value: object, expected_sha: object) -> Path:
    if not isinstance(value, str) or not value or not _is_sha256(expected_sha):
        raise ValueError("post-hoc inspection payload path/hash is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("post-hoc inspection payload paths must stay within the artifact")
    path = root / relative
    if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected_sha:
        raise ValueError("post-hoc inspection payload hash mismatch")
    return path


def load_forensic_posthoc_rgb_inspection(path: str | Path) -> Mapping[str, Any]:
    """Strictly validate a committed non-publication inspection artifact."""

    root = Path(path)
    manifest = _read_json_mapping(root / "manifest.json", label="post-hoc inspection manifest")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("post-hoc inspection artifact root is invalid")
    if (
        set(manifest) != _MANIFEST_FIELDS
        or manifest.get("schema") != FORENSIC_POSTHOC_RGB_INSPECTION_SCHEMA
        or manifest.get("watermarks") != list(POSTHOC_WATERMARKS)
        or manifest.get("status") != "completed_posthoc_human_inspection"
        or manifest.get("stage_b_run_source_commit") != STAGE_B_RUN_SOURCE_COMMIT
        or manifest.get("formal38_source_commit") != FORMAL38_SOURCE_COMMIT
        or manifest.get("decoder_checkpoint_step") != FORENSIC_STAGE_B_FINAL_STEP
        or manifest.get("publication_gate") != {"passed": False, "verdict_unchanged": True}
        or manifest.get("correct_condition_only") is not True
        or manifest.get("optimizer_constructed_or_loaded") is not False
        or manifest.get("training_or_resume_performed") is not False
        or manifest.get("wandb_enabled") is not False
        or manifest.get("decoder_checkpoint_sha256") != FINAL_CHECKPOINT_SHA256
        or manifest.get("summary_sha256") != SUMMARY_SHA256
        or manifest.get("failed_gate_metadata_sha256") != FAILED_GATE_METADATA_SHA256
        or manifest.get("cache_manifest_sha256") != STAGE_B_CACHE_MANIFEST_SHA256
        or manifest.get("cache_fingerprint") != STAGE_B_CACHE_FINGERPRINT
        or manifest.get("selection_identity") != STAGE_B_SELECTION_IDENTITY
        or manifest.get("color_space") != "sRGB"
        or manifest.get("channels") != 3
        or manifest.get("image_size") != 128
        or manifest.get("ode_steps") != 50
        or manifest.get("sample_chunk_size") != 8
        or manifest.get("sample_seed") != 20260921
        or not _is_sha256(manifest.get("checkpoint_invariants_sha256"))
        or not isinstance(manifest.get("source_identity"), Mapping)
        or not isinstance(manifest.get("original_job543457_snapshot"), Mapping)
        or manifest["original_job543457_snapshot"].get("unchanged_after_sampling") is not True
        or not isinstance(manifest["original_job543457_snapshot"].get("file_count"), int)
        or manifest["original_job543457_snapshot"]["file_count"] < 1
        or not _is_sha256(manifest["original_job543457_snapshot"].get("identity"))
        or not isinstance(manifest.get("rows"), list)
        or len(manifest["rows"]) != 16
    ):
        raise ValueError("post-hoc inspection manifest schema/identity is invalid")
    selection = manifest.get("sample_selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("algorithm") != "torch_randperm_cpu_v1"
        or selection.get("indices") != list(STAGE_B_SAMPLE_INDICES)
        or selection.get("indices_sha256") != STAGE_B_SAMPLE_INDICES_SHA256
        or selection.get("initial_noise_sha256") != STAGE_B_INITIAL_NOISE_SHA256
        or not _is_sha256(selection.get("row_identities_sha256"))
        or not _is_sha256(selection.get("original_image_sha256_identity"))
    ):
        raise ValueError("post-hoc inspection deterministic sample identity is invalid")
    row_identities = [row.get("row_identity") for row in manifest["rows"] if isinstance(row, Mapping)]
    image_sha256 = [row.get("source_original_image_sha256") for row in manifest["rows"] if isinstance(row, Mapping)]
    if (
        len(row_identities) != 16
        or len(image_sha256) != 16
        or _sha256_mapping({"row_identities": row_identities}) != selection["row_identities_sha256"]
        or _sha256_mapping({"original_image_sha256": image_sha256}) != selection["original_image_sha256_identity"]
    ):
        raise ValueError("post-hoc inspection ordered row/image identity mismatch")
    artifact_identity = manifest.get("artifact_identity")
    if (
        not _is_sha256(artifact_identity)
        or _sha256_mapping({key: value for key, value in manifest.items() if key != "artifact_identity"}) != artifact_identity
    ):
        raise ValueError("post-hoc inspection artifact identity mismatch")
    contact = _validated_relative_file(
        root, manifest.get("contact_sheet_path"), manifest.get("contact_sheet_png_sha256")
    )
    files: list[tuple[Path, tuple[int, int]]] = [(contact, (256, 16 * 146))]
    for ordinal, row in enumerate(manifest["rows"]):
        if (
            not isinstance(row, Mapping)
            or set(row) != _ROW_FIELDS
            or any("shuffled" in key for key in row)
            or row.get("ordinal") != ordinal
            or row.get("selection_index") != STAGE_B_SAMPLE_INDICES[ordinal]
            or not isinstance(row.get("row_identity"), str)
            or not row["row_identity"]
            or not _is_sha256(row.get("source_original_image_sha256"))
        ):
            raise ValueError("post-hoc inspection ordered row metadata is invalid")
        source_path = Path(str(row.get("source_original_image_path", "")))
        if (
            not source_path.is_absolute()
            or not source_path.is_file()
            or source_path.is_symlink()
            or _sha256_file(source_path) != row["source_original_image_sha256"]
        ):
            raise ValueError("post-hoc inspection source image identity mismatch")
        files.extend([
            (_validated_relative_file(root, row.get("original_path"), row.get("original_png_sha256")), (128, 128)),
            (_validated_relative_file(root, row.get("reconstruction_path"), row.get("reconstruction_png_sha256")), (128, 128)),
            (_validated_relative_file(root, row.get("strip_path"), row.get("strip_png_sha256")), (256, 146)),
        ])
    for image_path, expected_size in files:
        try:
            with Image.open(image_path) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != expected_size:
                    raise ValueError("post-hoc inspection RGB payload shape/format mismatch")
        except OSError as error:
            raise ValueError("post-hoc inspection RGB payload is unreadable") from error
    expected_files = {root / "manifest.json", *(path for path, _ in files)}
    actual_files = {path for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("post-hoc inspection contains undeclared or symlink payload")
    return manifest


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the exact non-publication Job543457 Stage B RGB inspection (separate launch approval required)"
    )
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--failed-gate-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sample_forensic_stage_b_posthoc_rgb_inspection(
        output_dir=args.output_dir,
        decoder_checkpoint=args.decoder_checkpoint,
        cache_dir=args.cache,
        summary_path=args.summary,
        failed_gate_metadata_path=args.failed_gate_metadata,
        device=device,
    )
    return 0


__all__ = [
    "FAILED_GATE_METADATA_SHA256",
    "FINAL_CHECKPOINT_SHA256",
    "FORENSIC_POSTHOC_RGB_INSPECTION_SCHEMA",
    "STAGE_B_CACHE_FINGERPRINT",
    "STAGE_B_CACHE_MANIFEST_SHA256",
    "STAGE_B_INITIAL_NOISE_SHA256",
    "STAGE_B_RUN_SOURCE_COMMIT",
    "STAGE_B_SAMPLE_INDICES",
    "STAGE_B_SAMPLE_INDICES_SHA256",
    "STAGE_B_SELECTION_IDENTITY",
    "SUMMARY_SHA256",
    "ForensicPosthocInspectionDurabilityError",
    "build_cli_parser",
    "load_forensic_posthoc_rgb_inspection",
    "sample_forensic_stage_b_posthoc_rgb_inspection",
]


if __name__ == "__main__":
    raise SystemExit(main())
