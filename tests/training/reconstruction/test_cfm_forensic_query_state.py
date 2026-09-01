from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

import nimloth.training.reconstruction.cfm_forensic_query_state as forensic_cfm
from nimloth.training.reconstruction.cfm_query_state import (
    LoadedQueryStateImageSplit,
    build_query_state_cfm_model,
    evaluate_query_state_multi_noise_sensitivity,
)
from nimloth.training.reconstruction.forensic_query_state_cache import (
    FORENSIC_QUERY_STATE_CACHE_SCHEMA,
    FORENSIC_QUERY_STATE_OWNER_ROLE,
    FORENSIC_SELECTION_MECHANICS_TRAIN,
    FORENSIC_SELECTION_MECHANICS_VALIDATION,
)
from nimloth.training.reconstruction.query_state_cache import (
    QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA,
)


def _split(role: str, *, marker: str, image_marker: str) -> LoadedQueryStateImageSplit:
    return LoadedQueryStateImageSplit(
        states=torch.zeros(3, 16, 1024),
        images_uint8=torch.zeros(3, 3, 8, 8, dtype=torch.uint8),
        rows=tuple(
            {
                "row_identity": f"{role}-{index}",
                "original_image_sha256": image_marker * 64,
                "selection_role": role,
            }
            for index in range(3)
        ),
        cache_schema=FORENSIC_QUERY_STATE_CACHE_SCHEMA,
        cache_fingerprint="a" * 64,
        bundle_fingerprint="b" * 64,
        source_manifest_identity="c" * 64,
        template_identity="d" * 64,
        checkpoint_identity=forensic_cfm.FORMAL38_UNSAFE_CONTROL_SHA256,
        split_name=role,
        split_identity=marker * 64,
        row_set_identity=("e" if role == FORENSIC_SELECTION_MECHANICS_TRAIN else "f") * 64,
        image_preprocessing={"size": 8, "resample": "bicubic", "range": [-1, 1], "color_space": "sRGB"},
    )


def _invariants() -> dict[str, Any]:
    model = build_query_state_cfm_model(image_size=8, base_channels=4, condition_dim=8, time_dim=16)
    return forensic_cfm.build_forensic_checkpoint_invariants(
        config=model.config,
        cache_fingerprint="a" * 64,
        train_split_identity="1" * 64,
        train_row_set_identity="2" * 64,
        validation_split_identity="3" * 64,
        validation_row_set_identity="4" * 64,
        source_identity={
            "source_commit": forensic_cfm.FORMAL38_SOURCE_COMMIT,
            "control_sha256": forensic_cfm.FORMAL38_UNSAFE_CONTROL_SHA256,
            "failure_manifest_sha256": forensic_cfm.FORMAL38_FAILURE_MANIFEST_SHA256,
            "config_identity": forensic_cfm.FORMAL38_RUN_IDENTITY,
            "run_identity": forensic_cfm.FORMAL38_RUN_IDENTITY,
        },
        train_items=48,
        validation_items=16,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-4,
        gradient_clip=1.0,
        evaluation_interval=500,
        save_interval=2000,
        seed=20260901,
        noise_seeds=(11, 29, 47),
        sample_items=8,
        sample_ode_steps=4,
        sample_noise_seed=71,
        sample_batch_size=4,
        image_preprocessing={"size": 8, "resample": "bicubic", "range": [-1, 1], "color_space": "sRGB"},
    )


def _evidence(deltas: tuple[float, ...], *, ratios: tuple[float, ...] | None = None) -> dict[str, Any]:
    states = torch.zeros(48, 16, 1024)
    states[:, 0, 0] = torch.arange(48) + 1
    images = torch.zeros(48, 3, 8, 8, dtype=torch.uint8)
    evidence = evaluate_query_state_multi_noise_sensitivity(
        _RecordingVelocity(), states, images, torch.device("cpu"), batch_size=2, seeds=(11, 29, 47)
    )
    for index, delta in enumerate(deltas):
        correct = 1.0
        shuffled = correct + delta
        if ratios is not None:
            shuffled = ratios[index]
            delta = shuffled - correct
        item = evidence["per_seed"][index]
        item["correct_flow_mse"] = correct
        item["shuffled_flow_mse"] = shuffled
        item["shuffled_minus_correct"] = delta
        item["shuffled_over_correct"] = shuffled / correct
    names = ("correct_flow_mse", "shuffled_flow_mse", "shuffled_minus_correct", "shuffled_over_correct")
    evidence["aggregate"] = {
        name: {
            "mean": sum(float(item[name]) for item in evidence["per_seed"]) / 3,
            "min": min(float(item[name]) for item in evidence["per_seed"]),
            "max": max(float(item[name]) for item in evidence["per_seed"]),
        }
        for name in names
    }
    payload = {key: value for key, value in evidence.items() if key != "identity"}
    evidence["identity"] = forensic_cfm._sha256_mapping(payload)
    return evidence


class _RecordingVelocity(nn.Module):
    def forward(self, image: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(image) + condition[:, :1, None, None] / 100.0


def test_cli_is_forensic_stage_a_only_and_has_no_unsafe_override() -> None:
    parser = forensic_cfm.build_cli_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--experiment-stage" in options
    assert "--allow-unsafe" not in options
    assert {"--wm-checkpoint", "--state-proj-checkpoint", "--source-checkpoint"}.isdisjoint(options)
    parsed = parser.parse_args([
        "--experiment-stage", "mechanics_only", "--cache", "cache", "--output-dir", "out",
        "--noise-seeds", "11", "29", "47", "--no-wandb",
    ])
    assert parsed.max_steps == 10000
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--experiment-stage", "stage_b", "--cache", "cache", "--output-dir", "out",
            "--noise-seeds", "11", "29", "47",
        ])


def test_manifest_gate_rejects_deployable_wrong_owner_source_and_stage() -> None:
    valid = {
        "schema": FORENSIC_QUERY_STATE_CACHE_SCHEMA,
        "owner_role": FORENSIC_QUERY_STATE_OWNER_ROLE,
        "forensic_only": True,
        "authoritative": False,
        "terminal_primary": False,
        "deployable": False,
        "sft2_ready": False,
        "cache_fingerprint": "a" * 64,
        "selection": {"stage": "mechanics_only", "roles": {"mechanics_train": 48, "mechanics_validation": 16}},
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
    forensic_cfm.validate_forensic_stage_a_manifest(valid)
    mutations = (
        {**valid, "schema": QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA},
        {**valid, "owner_role": "deployable_query_state"},
        {**valid, "selection": {"stage": "stage_b", "roles": valid["selection"]["roles"]}},
        {**valid, "checkpoint": {**valid["checkpoint"], "source_commit": "0" * 40}},
    )
    for value in mutations:
        with pytest.raises(ValueError, match="forensic|owner|stage|Formal38|source|schema"):
            forensic_cfm.validate_forensic_stage_a_manifest(value)


def test_stage_a_pair_requires_exact_roles_same_cache_and_image_disjoint() -> None:
    train = _split(FORENSIC_SELECTION_MECHANICS_TRAIN, marker="1", image_marker="7")
    validation = _split(FORENSIC_SELECTION_MECHANICS_VALIDATION, marker="3", image_marker="8")
    forensic_cfm.validate_forensic_split_pair(train, validation)
    with pytest.raises(ValueError, match="cache"):
        forensic_cfm.validate_forensic_split_pair(
            train, LoadedQueryStateImageSplit(**{**validation.__dict__, "cache_fingerprint": "9" * 64})
        )
    with pytest.raises(ValueError, match="image overlap"):
        forensic_cfm.validate_forensic_split_pair(
            train, LoadedQueryStateImageSplit(**{**validation.__dict__, "rows": ({"row_identity": "v0", "original_image_sha256": "7" * 64, "selection_role": "mechanics_validation"}, *validation.rows[1:])})
        )


def test_invariants_are_distinct_stage_bound_and_require_three_seeds() -> None:
    invariants = _invariants()
    assert invariants["experiment_stage"] == "mechanics_only"
    assert invariants["cache_schema"] == FORENSIC_QUERY_STATE_CACHE_SCHEMA
    assert invariants["train_role"] == "mechanics_train"
    assert invariants["validation_role"] == "mechanics_validation"
    assert invariants["max_steps"] == 10000
    assert invariants["pass_min_delta"] == pytest.approx(1e-4)
    assert invariants["pass_min_aggregate_ratio"] == pytest.approx(1.25)
    with pytest.raises(ValueError, match="three|seeds"):
        forensic_cfm.build_forensic_checkpoint_invariants(**{
            **{key: value for key, value in _invariant_kwargs().items()}, "noise_seeds": (11, 29)
        })


def _invariant_kwargs() -> dict[str, Any]:
    model = build_query_state_cfm_model(image_size=8, base_channels=4, condition_dim=8, time_dim=16)
    return {
        "config": model.config, "cache_fingerprint": "a" * 64,
        "train_split_identity": "1" * 64, "train_row_set_identity": "2" * 64,
        "validation_split_identity": "3" * 64, "validation_row_set_identity": "4" * 64,
        "source_identity": {"source_commit": forensic_cfm.FORMAL38_SOURCE_COMMIT, "control_sha256": forensic_cfm.FORMAL38_UNSAFE_CONTROL_SHA256, "failure_manifest_sha256": forensic_cfm.FORMAL38_FAILURE_MANIFEST_SHA256, "config_identity": forensic_cfm.FORMAL38_RUN_IDENTITY, "run_identity": forensic_cfm.FORMAL38_RUN_IDENTITY},
        "train_items": 48, "validation_items": 16, "batch_size": 32,
        "learning_rate": 1e-4, "weight_decay": 1e-4, "gradient_clip": 1.0,
        "evaluation_interval": 500, "save_interval": 2000, "seed": 20260901,
        "noise_seeds": (11, 29, 47), "sample_items": 8, "sample_ode_steps": 4,
        "sample_noise_seed": 71, "sample_batch_size": 4,
        "image_preprocessing": {"size": 8, "resample": "bicubic", "range": [-1, 1], "color_space": "sRGB"},
    }


def test_checkpoint_is_decoder_only_and_rejects_cross_cache_resume(tmp_path: Path) -> None:
    model = build_query_state_cfm_model(image_size=8, base_channels=4, condition_dim=8, time_dim=16)
    optimizer = forensic_cfm.build_decoder_optimizer(model, learning_rate=1e-4, weight_decay=1e-4)
    checkpoint = tmp_path / "final.pt"
    forensic_cfm.save_forensic_cfm_checkpoint(
        checkpoint, model=model, optimizer=optimizer, step=10000, invariants=_invariants()
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["schema"] == forensic_cfm.FORENSIC_CFM_CHECKPOINT_SCHEMA
    assert set(payload) == {"schema", "model", "optimizer", "step", "invariants", "torch_rng_state", "cuda_rng_state_all"}
    with pytest.raises(ValueError, match="cache|invariants|resume"):
        forensic_cfm.load_forensic_cfm_checkpoint(
            checkpoint, model=model, optimizer=optimizer,
            expected_invariants={**_invariants(), "cache_fingerprint": "f" * 64},
            device=torch.device("cpu"),
        )

    corrupt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    corrupt["step"] = True
    corrupt_path = tmp_path / "corrupt-step.pt"
    torch.save(corrupt, corrupt_path)
    with pytest.raises(ValueError, match="step|RNG"):
        forensic_cfm.load_forensic_cfm_checkpoint(
            corrupt_path,
            model=model,
            optimizer=optimizer,
            expected_invariants=_invariants(),
            device=torch.device("cpu"),
        )


def test_stage_a_gate_uses_complete_48_row_evidence_every_seed_and_aggregate_ratio() -> None:
    passed = forensic_cfm.evaluate_stage_a_pass(_evidence((0.3, 0.3, 0.3)))
    assert passed["passed"] is True
    assert passed["metric_unit"] == "mean conditional-flow velocity MSE per normalized [-1,1] RGB image element"
    with pytest.raises(ValueError, match="seed|shuffled_minus_correct|delta"):
        forensic_cfm.evaluate_stage_a_pass(_evidence((0.3, 0.00001, 0.3)))
    with pytest.raises(ValueError, match="aggregate.*ratio"):
        forensic_cfm.evaluate_stage_a_pass(_evidence((0.1, 0.1, 0.1)))
    incomplete = _evidence((0.3, 0.3, 0.3))
    incomplete["num_items"] = 47
    with pytest.raises(ValueError, match="complete|evidence"):
        forensic_cfm.evaluate_stage_a_pass(incomplete)


def test_final_publication_requires_step10000_train_gate_and_watermarks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = build_query_state_cfm_model(image_size=8, base_channels=4, condition_dim=8, time_dim=16)
    optimizer = forensic_cfm.build_decoder_optimizer(model, learning_rate=1e-4, weight_decay=1e-4)
    checkpoint = tmp_path / "final.pt"
    forensic_cfm.save_forensic_cfm_checkpoint(checkpoint, model=model, optimizer=optimizer, step=10000, invariants=_invariants())
    train = _split("mechanics_train", marker="1", image_marker="7")
    validation = _split("mechanics_validation", marker="3", image_marker="8")
    train = LoadedQueryStateImageSplit(**{**train.__dict__, "states": torch.zeros(48, 16, 1024), "images_uint8": torch.zeros(48, 3, 8, 8, dtype=torch.uint8), "rows": tuple({"row_identity": f"t{i}", "original_image_sha256": hashlib.sha256(f"t{i}".encode()).hexdigest(), "selection_role": "mechanics_train"} for i in range(48)), "row_set_identity": "2" * 64})
    validation = LoadedQueryStateImageSplit(**{**validation.__dict__, "states": torch.zeros(16, 16, 1024), "images_uint8": torch.zeros(16, 3, 8, 8, dtype=torch.uint8), "rows": tuple({"row_identity": f"v{i}", "original_image_sha256": hashlib.sha256(f"v{i}".encode()).hexdigest(), "selection_role": "mechanics_validation"} for i in range(16)), "row_set_identity": "4" * 64})
    reports = {"mechanics_train": _evidence((0.3, 0.3, 0.3)), "mechanics_validation": _evidence((-0.2, -0.2, -0.2))}
    monkeypatch.setattr(forensic_cfm, "load_forensic_image_splits", lambda *args, **kwargs: (train, validation, _invariants()["source_identity"]))
    monkeypatch.setattr(forensic_cfm, "evaluate_query_state_multi_noise_sensitivity", lambda _model, states, *_args, **_kwargs: reports["mechanics_train" if len(states) == 48 else "mechanics_validation"])
    monkeypatch.setattr(forensic_cfm, "sample_euler", lambda _model, condition, noise, **_kwargs: torch.zeros_like(noise))
    artifact = forensic_cfm.publish_forensic_rgb_artifacts(
        output_dir=tmp_path / "rgb", decoder_checkpoint=checkpoint, cache_dir=tmp_path / "cache", device=torch.device("cpu")
    )
    metadata = json.loads(Path(artifact["metadata_path"]).read_text())
    assert metadata["watermarks"] == ["mechanics_only", "unsafe_actor_checkpoint", "not_heldout", "not_deployable"]
    assert metadata["pass_source_role"] == "mechanics_train"
    assert metadata["mechanics_validation_controls_pass"] is False
    assert set(metadata["split_reports"]) == {"mechanics_train", "mechanics_validation"}
    assert metadata["decoder_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metadata["cache_fingerprint"] == "a" * 64
    artifact_images = [
        *artifact["strip_paths"],
        artifact["contact_sheet_path"],
        *(row["original_path"] for row in metadata["rows"]),
        *(row["reconstruction_path"] for row in metadata["rows"]),
    ]
    opened = [Image.open(path) for path in artifact_images]
    try:
        assert all(image.mode == "RGB" and image.format == "PNG" for image in opened)
    finally:
        for image in opened:
            image.close()
    assert all(len(row["original_png_sha256"]) == 64 for row in metadata["rows"])
    assert all(len(row["reconstruction_png_sha256"]) == 64 for row in metadata["rows"])

    bad = tmp_path / "intermediate.pt"
    forensic_cfm.save_forensic_cfm_checkpoint(bad, model=model, optimizer=optimizer, step=2000, invariants=_invariants())
    with pytest.raises(ValueError, match="final step10000"):
        forensic_cfm.publish_forensic_rgb_artifacts(
            output_dir=tmp_path / "bad-rgb", decoder_checkpoint=bad, cache_dir=tmp_path / "cache", device=torch.device("cpu")
        )
