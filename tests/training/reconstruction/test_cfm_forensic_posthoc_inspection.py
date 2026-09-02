from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

import nimloth.training.reconstruction.cfm_forensic_posthoc_inspection as posthoc
import nimloth.training.reconstruction.cfm_forensic_query_state as forensic_cfm
from nimloth.recon.cfm import CFMConfig
from nimloth.training.reconstruction.cfm_query_state import LoadedQueryStateImageSplit
from nimloth.training.reconstruction.forensic_query_state_cache import (
    FORENSIC_QUERY_STATE_CACHE_SCHEMA,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _external_split(tmp_path: Path) -> LoadedQueryStateImageSplit:
    source = tmp_path / "source.png"
    Image.new("RGB", (128, 128), (12, 34, 56)).save(source)
    source_sha = _sha256(source)
    rows = tuple(
        {
            "row_identity": f"external-{index}",
            "selection_role": "external_validation",
            "original_image_path": str(source.resolve()),
            "original_image_sha256": hashlib.sha256(
                f"image-{index}".encode()
            ).hexdigest(),
        }
        for index in range(1_413)
    )
    # The deterministic helper needs shape/order metadata, not materialized copies.
    states = torch.zeros(1, 16, 1024).expand(1_413, -1, -1)
    images = torch.zeros(1, 3, 128, 128, dtype=torch.uint8).expand(1_413, -1, -1, -1)
    # Sampling tests use the exact preprocessed source bytes and therefore bind the
    # selected rows to the one real fixture image after checking helper hashes.
    sampling_rows = tuple({**row, "original_image_sha256": source_sha} for row in rows)
    return LoadedQueryStateImageSplit(
        states=states,
        images_uint8=images,
        rows=sampling_rows,
        cache_schema=FORENSIC_QUERY_STATE_CACHE_SCHEMA,
        cache_fingerprint=posthoc.STAGE_B_CACHE_FINGERPRINT,
        bundle_fingerprint="b" * 64,
        source_manifest_identity="c" * 64,
        template_identity="d" * 64,
        checkpoint_identity=forensic_cfm.FORMAL38_UNSAFE_CONTROL_SHA256,
        split_name="external_validation",
        split_identity="1" * 64,
        row_set_identity="2" * 64,
        image_preprocessing={
            "size": 128,
            "resample": "bicubic",
            "range": [-1, 1],
            "color_space": "sRGB",
        },
    )


def test_shared_stage_b_plan_locks_indices_rows_images_and_noise() -> None:
    rows = tuple(
        {
            "row_identity": f"external-{index}",
            "original_image_sha256": hashlib.sha256(f"image-{index}".encode()).hexdigest(),
        }
        for index in range(1_413)
    )
    split = LoadedQueryStateImageSplit(
        states=torch.zeros(1, 16, 1024).expand(1_413, -1, -1),
        images_uint8=torch.zeros(1, 3, 128, 128, dtype=torch.uint8).expand(1_413, -1, -1, -1),
        rows=rows,
        cache_schema=FORENSIC_QUERY_STATE_CACHE_SCHEMA,
        cache_fingerprint=posthoc.STAGE_B_CACHE_FINGERPRINT,
        bundle_fingerprint="b" * 64,
        source_manifest_identity="c" * 64,
        template_identity="d" * 64,
        checkpoint_identity=forensic_cfm.FORMAL38_UNSAFE_CONTROL_SHA256,
        split_name="external_validation",
        split_identity="1" * 64,
        row_set_identity="2" * 64,
        image_preprocessing={"size": 128, "color_space": "sRGB"},
    )
    plan = forensic_cfm.build_forensic_stage_b_sample_plan(split)
    assert plan.indices == (424, 1245, 240, 761, 1360, 214, 191, 389, 84, 3, 182, 45, 246, 1255, 1060, 257)
    assert plan.indices_sha256 == "55257d76ab8f2dfb12aeb0bf9722fb2fa326be545832d9d119df36dc06015bef"
    assert plan.row_identities_sha256 == "6f750e269c47c094c30dab0340fe2c67e1a743b3a11b9a963acd51770375b5db"
    assert plan.original_image_sha256_identity == "5e48f771351db60fbd67d7e9e223d8c2ac4e6f3a890c7923eea6558309a0c3ea"
    assert plan.initial_noise_sha256 == "7390e403b9d92922fa94fd53b0e6b1fd09df3da002c397a89bca03e68d718575"
    assert plan.initial_noise.shape == (16, 3, 128, 128)


def test_original_run_snapshot_binds_symlink_paths_and_targets_without_following(
    tmp_path: Path,
) -> None:
    root = tmp_path / "job543457"
    root.mkdir()
    (root / "payload.json").write_text("{}\n")
    (root / "wandb").mkdir()
    (root / "wandb" / "latest-run").symlink_to("run-exact")

    entries, identity = posthoc._snapshot_regular_files(root)
    assert entries["payload.json"] == {
        "type": "file",
        "sha256": _sha256(root / "payload.json"),
    }
    assert entries["wandb/latest-run"] == {
        "type": "symlink",
        "target": "run-exact",
    }
    assert len(identity) == 64

    (root / "wandb" / "latest-run").unlink()
    (root / "wandb" / "latest-run").symlink_to("run-changed")
    changed_entries, changed_identity = posthoc._snapshot_regular_files(root)
    assert changed_entries["wandb/latest-run"]["target"] == "run-changed"
    assert changed_identity != identity


def test_posthoc_cli_surface_has_no_training_resume_gate_or_wandb_controls() -> None:
    parser = posthoc.build_cli_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert options == {
        "-h", "--help", "--decoder-checkpoint", "--cache", "--summary",
        "--failed-gate-metadata", "--output-dir", "--device",
    }
    assert options.isdisjoint({
        "--train", "--resume", "--resume-checkpoint", "--gate-override",
        "--allow-failed-gate", "--wandb-project", "--wandb-run-id", "--seed",
        "--rows", "--sample-steps", "--ode-steps", "--optimizer",
    })


def test_posthoc_api_accepts_no_sampling_or_gate_overrides() -> None:
    parameters = set(inspect.signature(posthoc.sample_forensic_stage_b_posthoc_rgb_inspection).parameters)
    assert parameters == {
        "output_dir", "decoder_checkpoint", "cache_dir", "summary_path",
        "failed_gate_metadata_path", "device",
    }


class _Decoder(nn.Module):
    def __init__(self, _config: CFMConfig) -> None:
        super().__init__()
        self.config = _config
        self.weight = nn.Parameter(torch.zeros(()))

    def load_state_dict(self, state_dict: Any, strict: bool = True) -> Any:
        assert strict is True and set(state_dict) == {"weight"}
        self.weight.data.copy_(state_dict["weight"])
        return None


def _stage_b_invariants() -> dict[str, Any]:
    config = CFMConfig(
        image_size=128, token_count=16, token_dim=1024, base_channels=64,
        condition_dim=256, time_dim=512, input_channels=3, output_channels=3,
    )
    return forensic_cfm.build_forensic_checkpoint_invariants(
        config=config,
        cache_fingerprint=posthoc.STAGE_B_CACHE_FINGERPRINT,
        experiment_stage=forensic_cfm.FORENSIC_STAGE_B_EXPERIMENT_STAGE,
        train_split_identity="3" * 64,
        train_row_set_identity="4" * 64,
        validation_split_identity="1" * 64,
        validation_row_set_identity="2" * 64,
        source_identity={
            "source_commit": forensic_cfm.FORMAL38_SOURCE_COMMIT,
            "control_sha256": forensic_cfm.FORMAL38_UNSAFE_CONTROL_SHA256,
            "failure_manifest_sha256": forensic_cfm.FORMAL38_FAILURE_MANIFEST_SHA256,
            "config_identity": forensic_cfm.FORMAL38_RUN_IDENTITY,
            "run_identity": forensic_cfm.FORMAL38_RUN_IDENTITY,
        },
        train_items=12_836,
        validation_items=1_413,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-4,
        gradient_clip=1.0,
        evaluation_interval=1_000,
        save_interval=1_000,
        seed=20260921,
        noise_seeds=forensic_cfm.FORENSIC_STAGE_B_NOISE_SEEDS,
        sample_items=16,
        sample_ode_steps=50,
        sample_noise_seed=20260921,
        sample_batch_size=8,
        image_preprocessing={
            "size": 128, "resample": "bicubic", "range": [-1, 1],
            "color_space": "sRGB",
        },
    )


def _artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path, Path]:
    run = tmp_path / "job543457"
    cfm = run / "cfm"
    rgb = cfm / "rgb_samples"
    rgb.mkdir(parents=True)
    (run / "launch_metadata.json").write_text(json.dumps({
        "schema": "nimloth_formal38_forensic_stage_b_cfm_run_v1",
        "source_commit": posthoc.STAGE_B_RUN_SOURCE_COMMIT,
        "cache_fingerprint": posthoc.STAGE_B_CACHE_FINGERPRINT,
    }, sort_keys=True) + "\n")
    (run / "controller.log").write_text("immutable original run evidence\n")
    checkpoint = cfm / "checkpoint_000004000.pt"
    invariants = _stage_b_invariants()
    torch.save(
        {
            "schema": forensic_cfm.FORENSIC_CFM_CHECKPOINT_SCHEMA,
            "model": {"weight": torch.tensor(0.0)},
            "optimizer": {"state": {}, "param_groups": [{"lr": 1e-4, "weight_decay": 1e-4, "params": [0]}]},
            "step": 4_000,
            "invariants": invariants,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": None,
        },
        checkpoint,
    )
    failed = {
        "schema": forensic_cfm.FORENSIC_CFM_RGB_ARTIFACT_SCHEMA,
        "status": "publication_gate_failed",
        "experiment_stage": forensic_cfm.FORENSIC_STAGE_B_EXPERIMENT_STAGE,
        "decoder_checkpoint": str(checkpoint.resolve()),
        "decoder_checkpoint_sha256": _sha256(checkpoint),
        "decoder_checkpoint_step": 4_000,
        "cache_fingerprint": posthoc.STAGE_B_CACHE_FINGERPRINT,
        "source_identity": invariants["source_identity"],
        "publication_gate": {"passed": False, "error": "seed 20260931 failed"},
        "rgb_artifacts_published": False,
    }
    failed["artifact_identity"] = forensic_cfm._sha256_mapping(failed)
    failed_path = rgb / "metadata.json"
    failed_path.write_text(json.dumps(failed, sort_keys=True) + "\n")
    summary = {
        "status": "publication_gate_failed",
        "watermarks": invariants["watermarks"],
        "final_checkpoint": str(checkpoint),
        "rgb_artifact": {
            "metadata_path": str(failed_path),
            "publication_gate": failed["publication_gate"],
        },
        "additional_steps_or_checkpoint_fallback_allowed": False,
    }
    summary_path = cfm / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    manifest = {
        "schema": FORENSIC_QUERY_STATE_CACHE_SCHEMA,
        "owner_role": "unsafe_forensic_query_state",
        "forensic_only": True,
        "authoritative": False,
        "terminal_primary": False,
        "deployable": False,
        "sft2_ready": False,
        "cache_fingerprint": posthoc.STAGE_B_CACHE_FINGERPRINT,
        "selection": {
            "stage": "stage_b_diagnostic",
            "roles": {"all_train": 12_836, "external_validation": 1_413},
            "identity": posthoc.STAGE_B_SELECTION_IDENTITY,
        },
        "checkpoint": {
            "source_commit": forensic_cfm.FORMAL38_SOURCE_COMMIT,
            "control_sha256": forensic_cfm.FORMAL38_UNSAFE_CONTROL_SHA256,
            "failure_manifest_sha256": forensic_cfm.FORMAL38_FAILURE_MANIFEST_SHA256,
            "config_identity": forensic_cfm.FORMAL38_RUN_IDENTITY,
            "run_identity": forensic_cfm.FORMAL38_RUN_IDENTITY,
            "world_size": 8,
            "checkpoint_path": "/run/forensics/unsafe_update_00001605",
            "actor_failure": {"passed": False},
        },
    }
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    monkeypatch.setattr(posthoc, "FINAL_CHECKPOINT_SHA256", _sha256(checkpoint))
    monkeypatch.setattr(posthoc, "SUMMARY_SHA256", _sha256(summary_path))
    monkeypatch.setattr(posthoc, "FAILED_GATE_METADATA_SHA256", _sha256(failed_path))
    monkeypatch.setattr(posthoc, "STAGE_B_CACHE_MANIFEST_SHA256", _sha256(manifest_path))
    return run, checkpoint, summary_path, failed_path, cache


def test_posthoc_rejects_wrong_job_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, checkpoint, summary, failed, cache = _artifacts(tmp_path, monkeypatch)
    launch_path = run / "launch_metadata.json"
    launch = json.loads(launch_path.read_text())
    launch["source_commit"] = "0" * 40
    launch_path.write_text(json.dumps(launch, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="source/cache launch identity"):
        posthoc._validate_exact_inputs(
            decoder_checkpoint=checkpoint,
            cache_dir=cache,
            summary_path=summary,
            failed_gate_metadata_path=failed,
        )


def test_posthoc_rejects_wrong_exact_identity_and_gate_passed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run, checkpoint, summary, failed, cache = _artifacts(tmp_path, monkeypatch)
    monkeypatch.setattr(posthoc, "FINAL_CHECKPOINT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        posthoc._validate_exact_inputs(
            decoder_checkpoint=checkpoint,
            cache_dir=cache,
            summary_path=summary,
            failed_gate_metadata_path=failed,
        )

    monkeypatch.setattr(posthoc, "FINAL_CHECKPOINT_SHA256", _sha256(checkpoint))
    failed_value = json.loads(failed.read_text())
    failed_value["publication_gate"] = {"passed": True}
    failed_value["artifact_identity"] = forensic_cfm._sha256_mapping(
        {key: value for key, value in failed_value.items() if key != "artifact_identity"}
    )
    failed.write_text(json.dumps(failed_value, sort_keys=True) + "\n")
    monkeypatch.setattr(posthoc, "FAILED_GATE_METADATA_SHA256", _sha256(failed))
    with pytest.raises(ValueError, match="gate-failed"):
        posthoc._validate_exact_inputs(
            decoder_checkpoint=checkpoint,
            cache_dir=cache,
            summary_path=summary,
            failed_gate_metadata_path=failed,
        )


def test_posthoc_sampling_is_model_only_frozen_inference_and_preserves_original_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, checkpoint, summary, failed, cache = _artifacts(tmp_path, monkeypatch)
    split = _external_split(tmp_path)
    monkeypatch.setattr(posthoc, "load_forensic_image_splits", lambda *_args, **_kwargs: (None, split, _stage_b_invariants()["source_identity"]))
    monkeypatch.setattr(posthoc, "TokenConditionedFlowUNet", _Decoder)

    def optimizer_forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("post-hoc inspection must not construct a live optimizer")

    monkeypatch.setattr(forensic_cfm, "build_decoder_optimizer", optimizer_forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", optimizer_forbidden)
    seen: dict[str, Any] = {}

    def sample(model: nn.Module, condition: torch.Tensor, noise: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        seen.update(model=model, condition=condition, noise=noise.clone(), kwargs=kwargs)
        assert not torch.is_grad_enabled()
        return torch.zeros_like(noise)

    monkeypatch.setattr(posthoc, "sample_euler", sample)
    before = {str(path.relative_to(run)): _sha256(path) for path in run.rglob("*") if path.is_file()}
    output = tmp_path / "inspection"
    result = posthoc.sample_forensic_stage_b_posthoc_rgb_inspection(
        output_dir=output,
        decoder_checkpoint=checkpoint,
        cache_dir=cache,
        summary_path=summary,
        failed_gate_metadata_path=failed,
        device=torch.device("cpu"),
    )
    after = {str(path.relative_to(run)): _sha256(path) for path in run.rglob("*") if path.is_file()}
    assert after == before
    assert not seen["model"].training
    assert all(not parameter.requires_grad for parameter in seen["model"].parameters())
    assert seen["condition"].shape == (16, 16 * 1024)
    assert seen["kwargs"]["steps"] == 50
    assert seen["kwargs"]["chunk_size"] == 8
    manifest = posthoc.load_forensic_posthoc_rgb_inspection(output)
    assert manifest["schema"] == posthoc.FORENSIC_POSTHOC_RGB_INSPECTION_SCHEMA
    assert manifest["watermarks"] == [
        "posthoc_human_inspection", "publication_gate_failed", "not_publication",
        "unsafe_actor_checkpoint", "not_deployable",
    ]
    assert manifest["failed_gate_metadata_sha256"] == _sha256(failed)
    assert len(manifest["rows"]) == 16
    assert Path(result["manifest_path"]) == output / "manifest.json"
    assert "shuffled" not in json.dumps(manifest).lower()

    durability_output = tmp_path / "inspection-durability-unknown"
    real_fsync = posthoc._fsync_directory
    destination_fsyncs = 0

    def fail_after_committed_manifest(path: Path) -> None:
        nonlocal destination_fsyncs
        if Path(path) == durability_output:
            destination_fsyncs += 1
            if destination_fsyncs == 2:
                raise OSError("injected post-commit durability failure")
        real_fsync(Path(path))

    monkeypatch.setattr(posthoc, "_fsync_directory", fail_after_committed_manifest)
    with pytest.raises(posthoc.ForensicPosthocInspectionDurabilityError, match="durability"):
        posthoc.sample_forensic_stage_b_posthoc_rgb_inspection(
            output_dir=durability_output,
            decoder_checkpoint=checkpoint,
            cache_dir=cache,
            summary_path=summary,
            failed_gate_metadata_path=failed,
            device=torch.device("cpu"),
        )
    assert posthoc.load_forensic_posthoc_rgb_inspection(durability_output)[
        "artifact_identity"
    ]
    with pytest.raises(FileExistsError):
        posthoc.sample_forensic_stage_b_posthoc_rgb_inspection(
            output_dir=durability_output,
            decoder_checkpoint=checkpoint,
            cache_dir=cache,
            summary_path=summary,
            failed_gate_metadata_path=failed,
            device=torch.device("cpu"),
        )


def test_posthoc_publish_order_is_claim_payload_then_manifest_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "staging"
    payload_dir = source / "payloads"
    payload_dir.mkdir(parents=True)
    (payload_dir / "payload.bin").write_bytes(b"payload")
    (source / "manifest.json").write_text("{}")
    destination = tmp_path / "inspection"
    events: list[tuple[str, str]] = []
    real_rename = posthoc.os.rename
    real_fsync = posthoc._fsync_directory

    def record_rename(source_path: str | Path, destination_path: str | Path) -> None:
        events.append(("rename", Path(source_path).name))
        real_rename(source_path, destination_path)

    def record_fsync(path: Path) -> None:
        events.append(("fsync", "parent" if Path(path) == destination.parent else "destination"))
        real_fsync(Path(path))

    monkeypatch.setattr(posthoc.os, "rename", record_rename)
    monkeypatch.setattr(posthoc, "_fsync_directory", record_fsync)
    posthoc._publish_inspection_noreplace(source, destination)
    assert events == [
        ("fsync", "parent"),
        ("rename", "payloads"),
        ("fsync", "destination"),
        ("rename", "manifest.json"),
        ("fsync", "destination"),
        ("fsync", "parent"),
    ]


def test_posthoc_premanifest_failure_burns_claim_and_cannot_be_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "staging"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"payload")
    (source / "manifest.json").write_text("{}")
    destination = tmp_path / "inspection"
    real_rename = posthoc.os.rename

    def fail_manifest(source_path: str | Path, destination_path: str | Path) -> None:
        if Path(source_path).name == "manifest.json":
            raise OSError("injected pre-manifest failure")
        real_rename(source_path, destination_path)

    monkeypatch.setattr(posthoc.os, "rename", fail_manifest)
    with pytest.raises(OSError, match="pre-manifest"):
        posthoc._publish_inspection_noreplace(source, destination)
    assert destination.is_dir()
    assert not (destination / "manifest.json").exists()
    with pytest.raises(ValueError, match="manifest"):
        posthoc.load_forensic_posthoc_rgb_inspection(destination)
    with pytest.raises(FileExistsError):
        posthoc._publish_inspection_noreplace(source, destination)


def test_posthoc_existing_sentinel_and_postcommit_durability_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "staging"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"payload")
    (source / "manifest.json").write_text("{}")
    destination = tmp_path / "inspection"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        posthoc._publish_inspection_noreplace(source, destination)
    assert sentinel.read_bytes() == b"keep"

    fresh_source = tmp_path / "staging-fresh"
    fresh_source.mkdir()
    (fresh_source / "payload.bin").write_bytes(b"payload")
    (fresh_source / "manifest.json").write_text("{}")
    fresh_destination = tmp_path / "inspection-fresh"
    real_fsync = posthoc._fsync_directory
    destination_fsyncs = 0

    def fail_after_manifest(path: Path) -> None:
        nonlocal destination_fsyncs
        if Path(path) == fresh_destination:
            destination_fsyncs += 1
            if destination_fsyncs == 2:
                raise OSError("post-commit fsync failure")
        real_fsync(Path(path))

    monkeypatch.setattr(posthoc, "_fsync_directory", fail_after_manifest)
    with pytest.raises(posthoc.ForensicPosthocInspectionDurabilityError, match="durability"):
        posthoc._publish_inspection_noreplace(fresh_source, fresh_destination)
    assert (fresh_destination / "manifest.json").is_file()
    with pytest.raises(FileExistsError):
        posthoc._publish_inspection_noreplace(fresh_source, fresh_destination)


def test_inspection_and_publication_schemas_are_bidirectionally_rejected(tmp_path: Path) -> None:
    inspection = tmp_path / "inspection"
    inspection.mkdir()
    (inspection / "manifest.json").write_text(json.dumps({"schema": forensic_cfm.FORENSIC_CFM_RGB_ARTIFACT_SCHEMA}))
    with pytest.raises(ValueError, match="schema|inspection"):
        posthoc.load_forensic_posthoc_rgb_inspection(inspection)
    with pytest.raises(ValueError, match="schema|publication"):
        forensic_cfm.validate_forensic_rgb_publication_metadata(
            {"schema": posthoc.FORENSIC_POSTHOC_RGB_INSPECTION_SCHEMA}
        )
