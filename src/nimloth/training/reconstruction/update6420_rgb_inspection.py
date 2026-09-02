"""Final-step correct-condition RGB inspection for the unsafe update6420 CFM."""

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

from nimloth.recon.cfm import sample_euler
from nimloth.training.reconstruction.cfm_forensic_posthoc_inspection import (
    _publish_inspection_noreplace,
    _save_png_fsynced,
    _strip,
    _tensor_image,
    _uint8_image,
)
from nimloth.training.reconstruction.cfm_query_state import (
    _load_image_uint8,
    flatten_query_state_condition,
)
from nimloth.training.reconstruction.update6420_cfm import (
    FINAL_STEP,
    UPDATE6420_CFM_EVALUATION_SCHEMA,
    load_update6420_final_decoder,
)
from nimloth.training.reconstruction.update6420_forensic_comparison import (
    LOCKED_INITIAL_NOISE_SHA256,
    LOCKED_SAMPLE_INDICES,
    LOCKED_SAMPLE_INDICES_SHA256,
    UPDATE6420_INSPECTION_SCHEMA,
    _validate_update_evaluation,
    canonical_identity,
)
from nimloth.training.reconstruction.update6420_query_state_cache import (
    Update6420QueryStateCacheDataset,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes(order="C")).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a mapping")
    return value


def _sample_plan() -> tuple[tuple[int, ...], str, torch.Tensor]:
    selection_generator = torch.Generator(device="cpu").manual_seed(20260921)
    selected = torch.randperm(1_413, generator=selection_generator)[:16]
    noise_generator = torch.Generator(device="cpu").manual_seed(20260921)
    noise = torch.randn((16, 3, 128, 128), generator=noise_generator).contiguous()
    return tuple(int(value) for value in selected.tolist()), _tensor_sha256(selected), noise


def _publish_manifest_last(staging: Path, output: Path) -> None:
    _publish_inspection_noreplace(staging, output)


def sample_update6420_rgb_inspection(*, decoder_checkpoint: Path, cache_dir: Path, evaluation_path: Path, output_dir: Path, device: torch.device) -> Mapping[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("update6420 RGB inspection output must be fresh")
    evaluation = _validate_update_evaluation(
        _read_json(evaluation_path, label="update6420 final evaluation")
    )
    if (
        evaluation.get("schema") != UPDATE6420_CFM_EVALUATION_SCHEMA
        or evaluation.get("checkpoint_step") != FINAL_STEP
        or evaluation.get("checkpoint_sha256") != _sha256_file(decoder_checkpoint)
        or evaluation.get("final_only") is not True
        or evaluation.get("actor_unsafe") is not True
        or evaluation.get("deployable") is not False
        or not isinstance(evaluation.get("gate"), Mapping)
        or not isinstance(evaluation["gate"].get("passed"), bool)
    ):
        raise ValueError("update6420 inspection requires the actual final evaluation/gate")
    decoder, invariants, _ = load_update6420_final_decoder(decoder_checkpoint, device=device)
    cache = Update6420QueryStateCacheDataset(cache_dir)
    if cache.cache_fingerprint != evaluation.get("cache_fingerprint") or cache.cache_fingerprint != invariants.get("cache_fingerprint"):
        raise ValueError("update6420 inspection cache/evaluation/checkpoint mismatch")
    external = [cache[index] for index in range(len(cache)) if cache[index]["selection_role"] == "external_validation"]
    if len(external) != 1_413 or max(LOCKED_SAMPLE_INDICES) >= len(external):
        raise ValueError("update6420 inspection external row set mismatch")
    indices, indices_sha256, noise = _sample_plan()
    if indices != LOCKED_SAMPLE_INDICES or indices_sha256 != LOCKED_SAMPLE_INDICES_SHA256 or _tensor_sha256(noise) != LOCKED_INITIAL_NOISE_SHA256:
        raise ValueError("update6420 inspection fixed selection/noise contract drift")
    selected = [external[index] for index in indices]
    states = torch.stack([item["state"] for item in selected]).float().contiguous()
    with torch.inference_mode():
        reconstructions = sample_euler(decoder, flatten_query_state_condition(states), noise, steps=50, device=device, chunk_size=8)
    if reconstructions.shape != (16, 3, 128, 128) or not torch.isfinite(reconstructions).all():
        raise ValueError("update6420 inspection reconstruction output mismatch")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent))
    strips: list[Image.Image] = []
    try:
        for name in ("originals", "reconstructions", "strips"):
            (staging / name).mkdir()
        rows: list[dict[str, Any]] = []
        for ordinal, (selection_index, item) in enumerate(zip(LOCKED_SAMPLE_INDICES, selected, strict=True)):
            source = Path(str(item["original_image_path"]))
            if source.is_symlink() or not source.is_file() or _sha256_file(source) != item["original_image_sha256"]:
                raise ValueError("update6420 inspection original image hash mismatch")
            original = _uint8_image(_load_image_uint8(source, 128))
            reconstruction = _tensor_image(reconstructions[ordinal])
            strip = _strip(original, reconstruction, str(item["row_identity"])); strips.append(strip)
            stem = f"row_{ordinal:05d}"
            paths = {
                "original": Path("originals") / f"{stem}.png",
                "reconstruction": Path("reconstructions") / f"{stem}.png",
                "strip": Path("strips") / f"{stem}.png",
            }
            for image, relative in ((original, paths["original"]), (reconstruction, paths["reconstruction"]), (strip, paths["strip"])):
                _save_png_fsynced(image, staging / relative)
            original.close(); reconstruction.close()
            rows.append({
                "ordinal": ordinal, "selection_index": selection_index,
                "row_identity": item["row_identity"], "original_image_sha256": item["original_image_sha256"],
                **{f"{name}_path": str(path) for name, path in paths.items()},
                **{f"{name}_sha256": _sha256_file(staging / path) for name, path in paths.items()},
            })
        contact = Image.new("RGB", (256, sum(strip.height for strip in strips)), "white")
        offset = 0
        for strip in strips:
            contact.paste(strip, (0, offset)); offset += strip.height
        _save_png_fsynced(contact, staging / "contact_sheet.png"); contact.close()
        watermarks = ["posthoc_human_inspection", "not_publication", "unsafe_actor_checkpoint", "not_deployable"]
        if evaluation["gate"]["passed"] is False:
            watermarks.append("publication_gate_failed")
        manifest: dict[str, Any] = {
            "schema": UPDATE6420_INSPECTION_SCHEMA, "status": "completed_posthoc_human_inspection",
            "watermarks": watermarks, "actual_comparison_gate": dict(evaluation["gate"]),
            "decoder_checkpoint_sha256": _sha256_file(decoder_checkpoint), "decoder_checkpoint_step": FINAL_STEP,
            "evaluation_sha256": _sha256_file(evaluation_path), "cache_manifest_sha256": _sha256_file(cache_dir / "manifest.json"),
            "cache_fingerprint": cache.cache_fingerprint, "correct_condition_only": True,
            "shuffled_condition_generated": False, "sample_indices": list(LOCKED_SAMPLE_INDICES),
            "sample_indices_sha256": LOCKED_SAMPLE_INDICES_SHA256, "initial_noise_sha256": LOCKED_INITIAL_NOISE_SHA256,
            "sample_seed": 20260921, "ode_steps": 50, "sample_batch_size": 8,
            "actor_unsafe": True, "deployable": False, "rows": rows,
            "contact_sheet_path": "contact_sheet.png", "contact_sheet_sha256": _sha256_file(staging / "contact_sheet.png"),
        }
        manifest["artifact_identity"] = canonical_identity(manifest)
        with (staging / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        _publish_manifest_last(staging, output_dir)
        return load_update6420_rgb_inspection(output_dir)
    except BaseException:
        if staging.exists(): shutil.rmtree(staging)
        raise
    finally:
        for strip in strips: strip.close()


def load_update6420_rgb_inspection(root: str | Path) -> Mapping[str, Any]:
    supplied = Path(root)
    if supplied.is_symlink():
        raise ValueError("update6420 RGB inspection root must not be a symlink")
    manifest = _read_json(supplied / "manifest.json", label="update6420 RGB inspection manifest")
    identity = manifest.pop("artifact_identity", None)
    required = {
        "schema", "status", "watermarks", "actual_comparison_gate",
        "decoder_checkpoint_sha256", "decoder_checkpoint_step", "evaluation_sha256",
        "cache_manifest_sha256", "cache_fingerprint", "correct_condition_only",
        "shuffled_condition_generated", "sample_indices", "sample_indices_sha256",
        "initial_noise_sha256", "sample_seed", "ode_steps", "sample_batch_size",
        "actor_unsafe", "deployable", "rows", "contact_sheet_path",
        "contact_sheet_sha256",
    }
    gate = manifest.get("actual_comparison_gate")
    expected_watermarks = [
        "posthoc_human_inspection", "not_publication",
        "unsafe_actor_checkpoint", "not_deployable",
    ]
    if isinstance(gate, Mapping) and gate.get("passed") is False:
        expected_watermarks.append("publication_gate_failed")
    if (
        set(manifest) != required
        or manifest.get("schema") != UPDATE6420_INSPECTION_SCHEMA
        or manifest.get("status") != "completed_posthoc_human_inspection"
        or not isinstance(gate, Mapping)
        or not isinstance(gate.get("passed"), bool)
        or manifest.get("watermarks") != expected_watermarks
        or manifest.get("actor_unsafe") is not True
        or manifest.get("deployable") is not False
        or manifest.get("correct_condition_only") is not True
        or manifest.get("shuffled_condition_generated") is not False
        or manifest.get("decoder_checkpoint_step") != FINAL_STEP
        or manifest.get("sample_indices") != list(LOCKED_SAMPLE_INDICES)
        or manifest.get("sample_indices_sha256") != LOCKED_SAMPLE_INDICES_SHA256
        or manifest.get("initial_noise_sha256") != LOCKED_INITIAL_NOISE_SHA256
        or manifest.get("sample_seed") != 20260921
        or manifest.get("ode_steps") != 50
        or manifest.get("sample_batch_size") != 8
        or any(not isinstance(manifest.get(name), str) or len(manifest[name]) != 64 for name in (
            "decoder_checkpoint_sha256", "evaluation_sha256", "cache_manifest_sha256",
            "cache_fingerprint", "contact_sheet_sha256",
        ))
        or canonical_identity(manifest) != identity
    ):
        raise ValueError("update6420 RGB inspection schema/classification/identity mismatch")
    manifest["artifact_identity"] = identity
    rows = manifest.get("rows")
    row_fields = {
        "ordinal", "selection_index", "row_identity", "original_image_sha256",
        "original_path", "original_sha256", "reconstruction_path",
        "reconstruction_sha256", "strip_path", "strip_sha256",
    }
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("update6420 RGB inspection row count mismatch")
    for ordinal, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != row_fields
            or row.get("ordinal") != ordinal
            or row.get("selection_index") != LOCKED_SAMPLE_INDICES[ordinal]
        ):
            raise ValueError("update6420 RGB inspection row identity mismatch")
        for name in ("original", "reconstruction", "strip"):
            relative = Path(str(row.get(f"{name}_path", "")))
            path = supplied / relative
            if relative.is_absolute() or ".." in relative.parts or path.is_symlink() or not path.is_file() or _sha256_file(path) != row.get(f"{name}_sha256"):
                raise ValueError("update6420 RGB inspection payload hash mismatch")
    if manifest.get("contact_sheet_path") != "contact_sheet.png":
        raise ValueError("update6420 RGB inspection contact-sheet path mismatch")
    contact = supplied / "contact_sheet.png"
    if contact.is_symlink() or not contact.is_file() or _sha256_file(contact) != manifest.get("contact_sheet_sha256"):
        raise ValueError("update6420 RGB inspection contact-sheet hash mismatch")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample final-step correct-condition update6420 RGB for post-hoc human inspection only")
    parser.add_argument("--decoder-checkpoint", type=Path, required=True); parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sample_update6420_rgb_inspection(decoder_checkpoint=args.decoder_checkpoint, cache_dir=args.cache, evaluation_path=args.evaluation, output_dir=args.output_dir, device=torch.device(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
