from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import nimloth.training.reconstruction.cfm_forensic_oracle_ladder as ladder
from nimloth.eval import query_state_oracle_ladder as oracle_eval
from nimloth.recon.cfm import CFMConfig


def _split(
    *,
    condition_family: str,
    role: str,
    marker: str,
    count: int = 4,
) -> ladder.LoadedOracleLadderSplit:
    return ladder.LoadedOracleLadderSplit(
        conditions=torch.arange(count * 16 * 12, dtype=torch.float32).reshape(
            count, 16, 12
        ),
        images_uint8=torch.arange(count * 3 * 16 * 16, dtype=torch.int64)
        .remainder(256)
        .byte()
        .reshape(count, 3, 16, 16),
        rows=tuple(
            {
                "selection_ordinal": index,
                "selection_role": role,
                "row_identity": f"row-{role}-{index}",
                "original_image_path": f"/images/{role}-{index}.png",
                "original_image_sha256": marker * 64,
            }
            for index in range(count)
        ),
        condition_family=condition_family,
        cache_schema=(
            ladder.FORENSIC_QUERY_STATE_CACHE_SCHEMA
            if condition_family == "state"
            else ladder.FORENSIC_DINO_ORACLE_CACHE_SCHEMA
        ),
        cache_fingerprint=marker * 64,
        source_state_cache_fingerprint="a" * 64,
        selection_identity="b" * 64,
        split_name=role,
        row_set_identity=("c" if role == "all_train" else "d") * 64,
        image_preprocessing={
            "size": 16,
            "resample": "bicubic",
            "range": [-1, 1],
            "color_space": "sRGB",
        },
    )


def _config() -> CFMConfig:
    return CFMConfig(
        image_size=16,
        token_count=16,
        token_dim=12,
        base_channels=4,
        condition_dim=8,
        time_dim=16,
    )


def _install_tiny_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ladder, "IMAGE_SIZE", 16)
    monkeypatch.setattr(ladder, "TOKEN_DIM", 12)
    monkeypatch.setattr(ladder, "BASE_CHANNELS", 4)
    monkeypatch.setattr(ladder, "CONDITION_DIM", 8)
    monkeypatch.setattr(ladder, "TIME_DIM", 16)
    monkeypatch.setattr(ladder, "TRAIN_ITEMS", 4)
    monkeypatch.setattr(ladder, "EXTERNAL_ITEMS", 4)
    monkeypatch.setattr(ladder, "BATCH_SIZE", 2)
    monkeypatch.setattr(ladder, "FINAL_STEP", 4)
    monkeypatch.setattr(ladder, "EVAL_INTERVAL", 2)
    monkeypatch.setattr(ladder, "SAVE_INTERVAL", 2)
    monkeypatch.setattr(ladder, "SAMPLE_ROWS", 3)


def _invariants(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cell: str = "token_oracle",
    output_dir: Path = Path("/tmp/oracle-ladder-test"),
) -> dict[str, Any]:
    _install_tiny_contract(monkeypatch)
    condition_family = "state" if cell == "spatial_state" else "dino_oracle"
    train = _split(condition_family=condition_family, role="all_train", marker="1")
    validation = _split(
        condition_family=condition_family,
        role="external_validation",
        marker="2",
    )
    return ladder.build_oracle_ladder_invariants(
        cell=cell,
        config=_config(),
        train=train,
        validation=validation,
        batch_size=2,
        learning_rate=1e-4,
        weight_decay=1e-4,
        gradient_clip=1.0,
        final_step=4,
        evaluation_interval=2,
        save_interval=2,
        seed=20260921,
        noise_seeds=(20260931, 20260932, 20260933),
        fixed_times=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95),
        sample_rows=3,
        sample_ode_steps=50,
        sample_selection_identity="e" * 64,
        source_commit="d" * 40,
        state_cache_path="/tmp/state-cache",
        oracle_cache_path="/tmp/oracle-cache",
        output_dir=str(output_dir.resolve()),
        wandb_project="nimloth-recon",
        wandb_mode="online",
        wandb_run_id="test-run-id",
        wandb_run_name="test-run-name",
    )


def test_ladder_cells_are_exact_and_build_separate_decoder_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_contract(monkeypatch)
    assert ladder.CELLS == {
        "token_state": {
            "decoder_family": "token_set_v1",
            "condition_family": "state",
            "train": False,
        },
        "token_oracle": {
            "decoder_family": "token_set_v1",
            "condition_family": "dino_oracle",
            "train": True,
        },
        "spatial_state": {
            "decoder_family": "spatial_grid_v1",
            "condition_family": "state",
            "train": True,
        },
        "spatial_oracle": {
            "decoder_family": "spatial_grid_v1",
            "condition_family": "dino_oracle",
            "train": True,
        },
    }
    token = ladder.build_oracle_ladder_model(
        decoder_family="token_set_v1", config=_config()
    )
    spatial = ladder.build_oracle_ladder_model(
        decoder_family="spatial_grid_v1", config=_config()
    )
    assert token.decoder_family == "token_set_v1"
    assert spatial.decoder_family == "spatial_grid_v1"
    assert sum(parameter.numel() for parameter in token.parameters()) > 0
    assert sum(parameter.numel() for parameter in spatial.parameters()) > 0
    with pytest.raises(ValueError, match="decoder family"):
        ladder.build_oracle_ladder_model(decoder_family="unknown", config=_config())


def test_ladder_invariants_lock_cell_condition_decoder_budget_and_sample_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invariants = _invariants(monkeypatch)
    assert invariants["cell"] == "token_oracle"
    assert invariants["decoder_family"] == "token_set_v1"
    assert invariants["condition_family"] == "dino_oracle"
    assert invariants["trainable_owner"] == "decoder_only"
    assert invariants["final_step"] == 4
    assert invariants["fixed_times"] == [0.0, 0.05, 0.25, 0.5, 0.75, 0.95]
    assert invariants["sample_rows"] == 3
    assert invariants["sample_ode_steps"] == 50
    assert invariants["source_state_cache_fingerprint"] == "a" * 64

    train = _split(
        condition_family="dino_oracle", role="all_train", marker="1"
    )
    validation = _split(
        condition_family="dino_oracle",
        role="external_validation",
        marker="2",
    )
    base = {
        "cell": "token_oracle",
        "config": _config(),
        "train": train,
        "validation": validation,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
        "final_step": 4,
        "evaluation_interval": 2,
        "save_interval": 2,
        "seed": 20260921,
        "noise_seeds": (20260931, 20260932, 20260933),
        "fixed_times": (0.0, 0.05, 0.25, 0.5, 0.75, 0.95),
        "sample_rows": 3,
        "sample_ode_steps": 50,
        "sample_selection_identity": "e" * 64,
        "source_commit": "d" * 40,
        "state_cache_path": "/tmp/state-cache",
        "oracle_cache_path": "/tmp/oracle-cache",
        "output_dir": "/tmp/oracle-ladder-test",
        "wandb_project": "nimloth-recon",
        "wandb_mode": "online",
        "wandb_run_id": "test-run-id",
        "wandb_run_name": "test-run-name",
    }
    for name, value in (
        ("condition", replace(train, condition_family="state")),
        ("steps", 5),
        ("noise", (1, 2, 3)),
        ("times", (0.0, 0.5, 0.95)),
    ):
        kwargs = dict(base)
        if name == "condition":
            kwargs["train"] = value
        elif name == "steps":
            kwargs["final_step"] = value
        elif name == "noise":
            kwargs["noise_seeds"] = value
        else:
            kwargs["fixed_times"] = value
        with pytest.raises(ValueError, match="cell|condition|matched|contract"):
            ladder.build_oracle_ladder_invariants(**kwargs)


class _ConditionVelocity(nn.Module):
    def forward(
        self,
        image: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        signal = condition[:, :1, None, None] / 100.0
        return image * 0.1 + signal + time[:, None, None, None]


def test_fixed_time_sensitivity_uses_exact_times_full_rows_and_matched_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_contract(monkeypatch)
    conditions = torch.zeros(4, 16, 12)
    conditions[:, 0, 0] = torch.arange(4, dtype=torch.float32)
    images = torch.arange(4 * 3 * 16 * 16, dtype=torch.int64)
    images = images.remainder(256).byte().reshape(4, 3, 16, 16)
    report = ladder.evaluate_fixed_time_sensitivity(
        _ConditionVelocity(),
        conditions,
        images,
        torch.device("cpu"),
        batch_size=2,
        seeds=(11, 29, 47),
        fixed_times=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95),
    )

    assert report["num_items"] == 4
    assert report["seeds"] == [11, 29, 47]
    assert report["fixed_times"] == [0.0, 0.05, 0.25, 0.5, 0.75, 0.95]
    assert len(report["per_seed_time"]) == 18
    assert all(item["matched_noise"] is True for item in report["per_seed_time"])
    assert {
        item["target_rgb_fraction_in_model_input"]
        for item in report["per_seed_time"]
    } == {0.0, 0.05, 0.25, 0.5, 0.75, 0.95}
    assert all(item["num_items"] == 4 for item in report["per_seed_time"])
    assert len(report["identity"]) == 64


def test_checkpoint_rejects_cross_cell_or_decoder_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invariants = _invariants(monkeypatch, output_dir=tmp_path)
    model = ladder.build_oracle_ladder_model(
        decoder_family="token_set_v1", config=_config()
    )
    optimizer = ladder.build_decoder_optimizer(
        model, learning_rate=1e-4, weight_decay=1e-4
    )
    checkpoint = tmp_path / "token-oracle.pt"
    ladder.save_oracle_ladder_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        step=4,
        invariants=invariants,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["schema"] == ladder.ORACLE_LADDER_CFM_CHECKPOINT_SCHEMA
    assert payload["invariants"]["decoder_family"] == "token_set_v1"

    spatial_invariants = _invariants(
        monkeypatch,
        cell="spatial_oracle",
        output_dir=tmp_path,
    )
    spatial = ladder.build_oracle_ladder_model(
        decoder_family="spatial_grid_v1", config=_config()
    )
    spatial_optimizer = ladder.build_decoder_optimizer(
        spatial, learning_rate=1e-4, weight_decay=1e-4
    )
    with pytest.raises(ValueError, match="cell|decoder|invariants|resume"):
        ladder.load_oracle_ladder_checkpoint(
            checkpoint,
            model=spatial,
            optimizer=spatial_optimizer,
            expected_invariants=spatial_invariants,
            device=torch.device("cpu"),
        )

    corrupt = json.loads(json.dumps(invariants))
    corrupt["cell"] = "spatial_oracle"
    with pytest.raises(ValueError, match="cell|decoder|invariants"):
        ladder.validate_oracle_ladder_invariants(corrupt, model.config)


def test_checkpoint_resume_restores_exact_rng_and_decoder_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invariants = _invariants(monkeypatch, output_dir=tmp_path)
    model = ladder.build_oracle_ladder_model(
        decoder_family="token_set_v1", config=_config()
    )
    optimizer = ladder.build_decoder_optimizer(
        model, learning_rate=1e-4, weight_decay=1e-4
    )
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().mean() for parameter in model.parameters()).backward()
    optimizer.step()
    torch.manual_seed(101)
    checkpoint = tmp_path / "checkpoint_000000002.pt"
    ladder.save_oracle_ladder_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        step=2,
        invariants=invariants,
    )
    expected_random = torch.rand(8)
    saved_parameter = next(model.parameters()).detach().clone()
    checkpoint_bytes = checkpoint.read_bytes()
    with pytest.raises(FileExistsError, match="exists"):
        ladder.save_oracle_ladder_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=2,
            invariants=invariants,
        )
    assert checkpoint.read_bytes() == checkpoint_bytes
    linked_checkpoint = tmp_path / "linked-checkpoint.pt"
    linked_checkpoint.symlink_to(checkpoint.name)
    with pytest.raises(ValueError, match="symlink|regular"):
        ladder.load_oracle_ladder_checkpoint(
            linked_checkpoint,
            model=model,
            optimizer=optimizer,
            expected_invariants=invariants,
            device=torch.device("cpu"),
        )
    with torch.no_grad():
        next(model.parameters()).zero_()
    torch.manual_seed(999)

    step = ladder.load_oracle_ladder_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected_invariants=invariants,
        device=torch.device("cpu"),
    )

    assert step == 2
    torch.testing.assert_close(torch.rand(8), expected_random)
    torch.testing.assert_close(next(model.parameters()), saved_parameter)
    copied_dir = tmp_path / "copied-output"
    copied_dir.mkdir()
    copied = copied_dir / checkpoint.name
    copied.write_bytes(checkpoint.read_bytes())
    with pytest.raises(ValueError, match="output|W&B"):
        ladder.load_oracle_ladder_checkpoint(
            copied,
            model=model,
            optimizer=optimizer,
            expected_invariants=invariants,
            device=torch.device("cpu"),
        )
    resumed_model = ladder.build_oracle_ladder_model(
        decoder_family="token_set_v1", config=_config()
    )
    resumed_optimizer = ladder.build_decoder_optimizer(
        resumed_model, learning_rate=1e-4, weight_decay=1e-4
    )
    ladder.load_oracle_ladder_checkpoint(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        expected_invariants=invariants,
        device=torch.device("cpu"),
    )
    optimizer.zero_grad(set_to_none=True)
    resumed_optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().mean() for parameter in model.parameters()).backward()
    sum(
        parameter.square().mean() for parameter in resumed_model.parameters()
    ).backward()
    optimizer.step()
    resumed_optimizer.step()
    for uninterrupted, resumed in zip(
        model.parameters(), resumed_model.parameters(), strict=True
    ):
        torch.testing.assert_close(resumed, uninterrupted, rtol=0, atol=0)


def test_evaluator_rejects_final_checkpoint_without_completed_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invariants = _invariants(monkeypatch, output_dir=tmp_path)
    monkeypatch.setattr(oracle_eval, "FINAL_STEP", 4)
    model = ladder.build_oracle_ladder_model(
        decoder_family="token_set_v1", config=_config()
    )
    optimizer = ladder.build_decoder_optimizer(
        model, learning_rate=1e-4, weight_decay=1e-4
    )
    checkpoint = (tmp_path / "checkpoint_000000004.pt").resolve()
    ladder.save_oracle_ladder_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        step=4,
        invariants=invariants,
    )

    with pytest.raises(ValueError, match="completed training summary"):
        oracle_eval._load_new_cell_checkpoint(
            checkpoint,
            cell="token_oracle",
            device=torch.device("cpu"),
        )


def test_evaluator_accepts_exact_completed_checkpoint_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invariants = _invariants(monkeypatch, output_dir=tmp_path)
    monkeypatch.setattr(oracle_eval, "FINAL_STEP", 4)
    model = ladder.build_oracle_ladder_model(
        decoder_family="token_set_v1", config=_config()
    )
    optimizer = ladder.build_decoder_optimizer(
        model, learning_rate=1e-4, weight_decay=1e-4
    )
    checkpoint = (tmp_path / "checkpoint_000000004.pt").resolve()
    ladder.save_oracle_ladder_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        step=4,
        invariants=invariants,
    )
    resolved_config = {
        "cell": "token_oracle",
        "state_cache": invariants["state_cache_path"],
        "oracle_cache": invariants["oracle_cache_path"],
        "output": str(tmp_path.resolve()),
        "device": "cpu",
        "wandb_project": "nimloth-recon",
        "wandb_mode": "online",
        "wandb_run_id": "test-run-id",
        "wandb_run_name": "test-run-name",
        "invariants_identity": ladder._identity(invariants),
    }
    producer = {
        "module": "nimloth.training.reconstruction.cfm_forensic_oracle_ladder",
        "source_commit": invariants["source_commit"],
        "resolved_config": resolved_config,
    }
    producer["identity"] = ladder._identity(producer)
    metadata = {
        "schema": ladder.ORACLE_LADDER_METADATA_SCHEMA,
        "cell": "token_oracle",
        "invariants": invariants,
        "state_cache": invariants["state_cache_path"],
        "oracle_cache": invariants["oracle_cache_path"],
        "output": str(tmp_path.resolve()),
        "producer": producer,
        "tracking": {
            "enabled": True,
            "project": "nimloth-recon",
            "mode": "online",
            "run_id": "test-run-id",
            "run_name": "test-run-name",
        },
    }
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    random_plan = {"owner": "test"}
    random_plan["identity"] = ladder._identity(random_plan)
    random_report = {
        "schema": "nimloth_query_state_cfm_multi_noise_sensitivity_v1",
        "seeds": [20260931, 20260932, 20260933],
        "num_items": 4,
        "per_seed": [],
        "aggregate": {},
    }
    random_report["identity"] = ladder._identity(random_report)
    random_report["noise_time_provenance"] = random_plan
    random_report["owner_identity"] = ladder._identity(random_report)
    summary = {
        "schema": ladder.ORACLE_LADDER_SUMMARY_SCHEMA,
        "cell": "token_oracle",
        "status": "completed_final_step4000",
        "final_step": 4,
        "final_checkpoint": str(checkpoint),
        "final_checkpoint_sha256": ladder._sha256_file(checkpoint),
        "random_time_report": random_report,
        "invariants_identity": ladder._identity(invariants),
        "metadata_identity": ladder._identity(metadata),
        "producer_identity": producer["identity"],
        "evaluation_pending": True,
        "additional_steps_or_fallback_allowed": False,
    }
    summary["identity"] = ladder._identity(summary)
    (tmp_path / "summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded, loaded_invariants, checkpoint_sha = (
        oracle_eval._load_new_cell_checkpoint(
            checkpoint,
            cell="token_oracle",
            device=torch.device("cpu"),
        )
    )
    assert loaded.decoder_family == "token_set_v1"
    assert loaded_invariants == invariants
    assert checkpoint_sha == ladder._sha256_file(checkpoint)


def test_output_isolated_and_cli_requires_tracked_wandb_identity(
    tmp_path: Path,
) -> None:
    state_cache = tmp_path / "state-cache"
    oracle_cache = tmp_path / "oracle-cache"
    state_cache.mkdir()
    oracle_cache.mkdir()
    with pytest.raises(ValueError, match="immutable|inside|output"):
        ladder._reject_protected_output(
            state_cache / "forbidden-run",
            (state_cache, oracle_cache),
        )
    ladder._reject_protected_output(
        tmp_path / "fresh-run",
        (state_cache, oracle_cache),
    )
    dangling_parent = tmp_path / "dangling-parent"
    dangling_parent.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        ladder._reject_protected_output(
            dangling_parent / "run",
            (state_cache, oracle_cache),
        )
    resume_dir = tmp_path / "resume-run"
    resume_dir.mkdir()
    committed = resume_dir / "checkpoint_000001000.pt"
    committed.write_bytes(b"committed")
    (resume_dir / "checkpoint_000002000.pt").symlink_to(committed.name)
    with pytest.raises(ValueError, match="automatic resume|symlink"):
        ladder._select_auto_resume_checkpoint(resume_dir)

    parser = ladder.build_cli_parser()
    required = [
        "--cell",
        "token_oracle",
        "--state-cache",
        str(state_cache),
        "--oracle-cache",
        str(oracle_cache),
        "--output-dir",
        str(tmp_path / "fresh-run"),
        "--device",
        "cpu",
        "--wandb-project",
        "nimloth-recon",
        "--wandb-mode",
        "online",
        "--wandb-run-id",
        "exact-id",
        "--wandb-run-name",
        "exact-name",
    ]
    parsed = parser.parse_args(required)
    assert parsed.wandb_project == "nimloth-recon"
    assert parsed.wandb_mode == "online"
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--no-wandb"])
