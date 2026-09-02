"""Fresh matched CFM trainer/evaluator for the unsafe update6420 cache only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from nimloth.recon.cfm import conditional_flow_matching_loss
from nimloth.training.reconstruction.cfm_query_state import (
    LoadedQueryStateImageSplit,
    _load_image_uint8,
    build_decoder_optimizer,
    build_query_state_cfm_model,
    evaluate_query_state_multi_noise_sensitivity,
    flatten_query_state_condition,
)
from nimloth.training.reconstruction.update6420_forensic_comparison import (
    LOCKED_NOISE_SEEDS,
    UPDATE6420_CFM_INVARIANTS_SCHEMA,
    build_matched_cfm_invariants,
    canonical_identity,
    validate_matched_cfm_invariants,
)
from nimloth.training.reconstruction.update6420_query_state_cache import (
    Update6420QueryStateCacheDataset,
)

UPDATE6420_CFM_CHECKPOINT_SCHEMA = "nimloth_update6420_matched_cfm_checkpoint_v1"
UPDATE6420_CFM_EVALUATION_SCHEMA = "nimloth_update6420_matched_cfm_final_evaluation_v1"
FINAL_STEP = 4_000
EVALUATION_INTERVAL = 1_000
SAVE_INTERVAL = 1_000
SEED = 20260921


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_noreplace(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_update6420_image_splits(cache_dir: str | Path) -> tuple[LoadedQueryStateImageSplit, LoadedQueryStateImageSplit, Mapping[str, Any]]:
    dataset = Update6420QueryStateCacheDataset(cache_dir)
    items = [dataset[index] for index in range(len(dataset))]
    preprocessing = {"size": 128, "resample": "bicubic", "range": [-1, 1], "color_space": "sRGB"}
    splits: list[LoadedQueryStateImageSplit] = []
    for role, count in (("all_train", 12_836), ("external_validation", 1_413)):
        selected = [item for item in items if item["selection_role"] == role]
        if len(selected) != count:
            raise ValueError("update6420 CFM cache role/count boundary mismatch")
        states = torch.stack([item["state"] for item in selected]).float().contiguous()
        images = torch.stack([_load_image_uint8(item["original_image_path"], 128) for item in selected])
        rows = tuple({key: value for key, value in item.items() if key != "state"} for item in selected)
        row_identity = canonical_identity(rows)
        splits.append(LoadedQueryStateImageSplit(
            states=states, images_uint8=images, rows=rows,
            cache_schema=str(dataset.manifest["schema"]), cache_fingerprint=dataset.cache_fingerprint,
            bundle_fingerprint=str(dataset.manifest["checkpoint"]["run_identity"]),
            source_manifest_identity=str(dataset.manifest["ordered_identity_digests"]["full"]),
            template_identity=str(rows[0]["template_identity"]),
            checkpoint_identity=str(dataset.manifest["checkpoint"]["checkpoint_identity"]),
            split_name=role,
            split_identity=canonical_identity({"cache_fingerprint": dataset.cache_fingerprint, "role": role, "rows": row_identity}),
            row_set_identity=row_identity, image_preprocessing=preprocessing,
        ))
    train_images = {row["original_image_sha256"] for row in splits[0].rows}
    external_images = {row["original_image_sha256"] for row in splits[1].rows}
    if train_images & external_images:
        raise ValueError("update6420 CFM train/external exact-image overlap is forbidden")
    return splits[0], splits[1], dataset.manifest


def build_update6420_cfm_invariants(*, train: LoadedQueryStateImageSplit, external: LoadedQueryStateImageSplit, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if train.cache_fingerprint != external.cache_fingerprint or train.checkpoint_identity != external.checkpoint_identity:
        raise ValueError("update6420 CFM roles must share one cache/checkpoint owner")
    base = build_matched_cfm_invariants(
        {
            "state_shape": [16, 1024], "image_size": 128, "input_channels": 3, "output_channels": 3,
            "base_channels": 64, "condition_dim": 256, "time_dim": 512, "batch_size": 32,
            "learning_rate": 1e-4, "weight_decay": 1e-4, "gradient_clip": 1.0,
            "max_steps": 4000, "evaluation_interval": 1000, "save_interval": 1000,
            "seed": 20260921, "noise_seeds": list(LOCKED_NOISE_SEEDS), "sample_items": 16,
            "sample_ode_steps": 50, "sample_noise_seed": 20260921, "sample_batch_size": 8,
            "shuffle_algorithm": "global_cyclic_shift_v1", "correct_and_shuffled_share_noise_and_time": True,
            "metric_unit": "mean conditional-flow velocity MSE per normalized [-1,1] RGB element",
            "checkpoint_selection": "final_step4000_only", "pass_min_delta": 0.01,
            "pass_min_aggregate_ratio": 1.05,
            "image_preprocessing": {"color_space": "sRGB", "resample": "bicubic", "range": [-1, 1]},
        },
        cache_fingerprint=train.cache_fingerprint,
        checkpoint_identity=train.checkpoint_identity,
        row_identity_digest=str(manifest["ordered_identity_digests"]["row"]),
    )
    return {
        **base,
        "cache_manifest_sha256": _sha256_file(Path(str(manifest["_manifest_path"]))) if "_manifest_path" in manifest else None,
        "train_split_identity": train.split_identity,
        "train_row_set_identity": train.row_set_identity,
        "external_split_identity": external.split_identity,
        "external_row_set_identity": external.row_set_identity,
        "evaluation_protocol": {
            "rows": 1_413, "seeds": list(LOCKED_NOISE_SEEDS),
            "shuffle": "global_cyclic_shift_v1", "shared_noise_and_time": True,
            "statistical_unit": "normalized [-1,1] RGB element",
            "aggregation": "full-row/RGB mean per seed then mean across seeds",
        },
    }


def _checkpoint_payload(*, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, invariants: Mapping[str, Any]) -> dict[str, Any]:
    model_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if model_ids != optimizer_ids:
        raise ValueError("update6420 CFM optimizer must own decoder parameters only")
    return {
        "schema": UPDATE6420_CFM_CHECKPOINT_SCHEMA,
        "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "invariants": dict(invariants), "decoder_only": True, "actor_unsafe": True,
        "deployable": False, "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_update6420_cfm_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, invariants: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink() or not 0 < step <= FINAL_STEP:
        raise ValueError("update6420 CFM checkpoint path/step is invalid")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(_checkpoint_payload(model=model, optimizer=optimizer, step=step, invariants=invariants), stream)
            stream.flush(); os.fsync(stream.fileno())
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_update6420_final_decoder(path: str | Path, *, device: torch.device) -> tuple[torch.nn.Module, Mapping[str, Any], Mapping[str, Any]]:
    supplied = Path(path)
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError("update6420 final decoder checkpoint must be a regular file")
    try:
        payload = torch.load(supplied, map_location=device, weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("update6420 final decoder checkpoint is unreadable") from error
    required = {"schema", "step", "model", "optimizer", "invariants", "decoder_only", "actor_unsafe", "deployable", "torch_rng_state", "cuda_rng_state_all"}
    invariants = payload.get("invariants") if isinstance(payload, Mapping) else None
    if not isinstance(payload, Mapping) or set(payload) != required or payload.get("schema") != UPDATE6420_CFM_CHECKPOINT_SCHEMA or payload.get("step") != FINAL_STEP or payload.get("decoder_only") is not True or payload.get("actor_unsafe") is not True or payload.get("deployable") is not False or not isinstance(invariants, Mapping):
        raise ValueError("update6420 final decoder schema/step/invariants mismatch")
    validate_matched_cfm_invariants(invariants)
    model = build_query_state_cfm_model(image_size=128, base_channels=64, condition_dim=256, time_dim=512).to(device)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("update6420 final decoder model state mismatch") from error
    model.eval().requires_grad_(False)
    return model, invariants, payload


def _metric_file(*, checkpoint_sha256: str, report: Mapping[str, Any], gate: Mapping[str, Any], train_curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_seed = [{"seed": item["noise_time_seed"], "correct": item["correct_flow_mse"], "shuffled": item["shuffled_flow_mse"]} for item in report["per_seed"]]
    value: dict[str, Any] = {
        "schema": UPDATE6420_CFM_EVALUATION_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256, "checkpoint_step": FINAL_STEP,
        "cache_fingerprint": report["cache_fingerprint"], "per_seed": per_seed,
        "full_report": dict(report), "gate": dict(gate), "train_curve": [dict(item) for item in train_curve],
        "final_only": True, "actor_unsafe": True, "deployable": False,
    }
    value["artifact_identity"] = canonical_identity(value)
    return value


def train_update6420_cfm(*, cache_dir: Path, output_dir: Path, device: torch.device, tracking: Mapping[str, Any]) -> int:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("update6420 CFM output must be fresh")
    torch.manual_seed(SEED)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.manual_seed_all(SEED)
    train, external, manifest = load_update6420_image_splits(cache_dir)
    manifest = dict(manifest); manifest["_manifest_path"] = str(cache_dir / "manifest.json")
    invariants = build_update6420_cfm_invariants(train=train, external=external, manifest=manifest)
    output_dir.mkdir(parents=True)
    _write_json_noreplace(output_dir / "metadata.json", {
        "schema": "nimloth_update6420_matched_cfm_run_v1", "invariants": invariants,
        "trainable_owner": "TokenConditionedFlowUNet_only", "actor_unsafe": True,
        "deployable": False, "fresh_decoder": True, "tracking": dict(tracking),
    })
    model = build_query_state_cfm_model(image_size=128, base_channels=64, condition_dim=256, time_dim=512).to(device)
    optimizer = build_decoder_optimizer(model, learning_rate=1e-4, weight_decay=1e-4)
    wandb_run = None
    if tracking.get("enabled") is True:
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            import wandb
            wandb_run = wandb.init(project=tracking["project"], id=tracking["run_id"], name=tracking["run_name"], config=invariants, dir=str(output_dir), resume="never")
        except Exception as error:
            raise RuntimeError("update6420 CFM W&B initialization failed") from error
        finally:
            torch.set_rng_state(cpu_rng)
            if torch.cuda.is_available() and cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
    curve: list[dict[str, Any]] = []
    log_path = output_dir / "train_step_log.csv"
    with log_path.open("x", newline="") as stream:
        csv.writer(stream).writerow(["time", "step", "train_flow_mse"])
    final_report: Mapping[str, Any] | None = None
    for step in range(1, FINAL_STEP + 1):
        indices = torch.randint(len(train), (32,))
        state = train.states[indices].to(device).float()
        target = train.images_uint8[indices].to(device).float().div(127.5).sub(1)
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = conditional_flow_matching_loss(model, target, flatten_query_state_condition(state))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % EVALUATION_INTERVAL == 0:
            train_report = evaluate_query_state_multi_noise_sensitivity(
                model, train.states, train.images_uint8, device,
                batch_size=32, seeds=LOCKED_NOISE_SEEDS,
            )
            external_report = evaluate_query_state_multi_noise_sensitivity(
                model, external.states, external.images_uint8, device,
                batch_size=32, seeds=LOCKED_NOISE_SEEDS,
            )
            final_report = {**external_report, "cache_fingerprint": train.cache_fingerprint}
            point = {
                "step": step,
                "train_flow_mse": float(loss.detach().cpu()),
                "train_report_identity": train_report["identity"],
                "external_report_identity": final_report["identity"],
            }
            curve.append(point)
            with log_path.open("a", newline="") as stream:
                csv.writer(stream).writerow([time.time(), step, point["train_flow_mse"]])
            if wandb_run is not None:
                wandb_run.log({
                    "cfm/train_flow_mse": point["train_flow_mse"],
                    "cfm/all_train_correct": train_report["aggregate"]["correct_flow_mse"]["mean"],
                    "cfm/all_train_shuffled": train_report["aggregate"]["shuffled_flow_mse"]["mean"],
                    "cfm/external_correct": final_report["aggregate"]["correct_flow_mse"]["mean"],
                    "cfm/external_shuffled": final_report["aggregate"]["shuffled_flow_mse"]["mean"],
                }, step=step)
        if step % SAVE_INTERVAL == 0:
            save_update6420_cfm_checkpoint(output_dir / f"checkpoint_{step:09d}.pt", model=model, optimizer=optimizer, step=step, invariants=invariants)
    if final_report is None:
        raise RuntimeError("update6420 CFM final evaluation was not executed")
    final_path = output_dir / f"checkpoint_{FINAL_STEP:09d}.pt"
    if not final_path.exists():
        save_update6420_cfm_checkpoint(final_path, model=model, optimizer=optimizer, step=FINAL_STEP, invariants=invariants)
    deltas = [float(item["shuffled_minus_correct"]) for item in final_report["per_seed"]]
    ratio = float(final_report["aggregate"]["shuffled_flow_mse"]["mean"]) / max(float(final_report["aggregate"]["correct_flow_mse"]["mean"]), 1e-12)
    gate = {"passed": all(delta >= 0.01 for delta in deltas) and ratio >= 1.05, "each_delta_minimum": 0.01, "aggregate_ratio_minimum": 1.05, "per_seed_delta": deltas, "aggregate_ratio": ratio}
    _write_json_noreplace(output_dir / "final_evaluation.json", _metric_file(checkpoint_sha256=_sha256_file(final_path), report=final_report, gate=gate, train_curve=curve))
    if wandb_run is not None:
        wandb_run.log({"cfm/final_gate_passed": int(gate["passed"])}, step=FINAL_STEP + 1); wandb_run.finish()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fresh exact matched CFM from the unsafe update6420 cache")
    parser.add_argument("--cache", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--wandb-project", default="nimloth-recon"); parser.add_argument("--wandb-run-id"); parser.add_argument("--wandb-run-name"); parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.no_wandb and not all(isinstance(value, str) and value.strip() for value in (args.wandb_project, args.wandb_run_id, args.wandb_run_name)):
        raise ValueError("tracked update6420 CFM requires explicit W&B project/run-id/run-name")
    tracking = {"enabled": not args.no_wandb, "project": args.wandb_project, "run_id": args.wandb_run_id, "run_name": args.wandb_run_name}
    return train_update6420_cfm(cache_dir=args.cache, output_dir=args.output_dir, device=torch.device(args.device), tracking=tracking)


if __name__ == "__main__":
    raise SystemExit(main())
