from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import nimloth.training.reconstruction.cfm_dino_grid_ceiling as ceiling
from nimloth.recon.cfm import CFMConfig


def _split(*, grid_size: int, role: str, marker: str) -> ceiling.LoadedDinoGridCeilingSplit:
    count = 4
    return ceiling.LoadedDinoGridCeilingSplit(
        conditions=torch.arange(
            count * grid_size**2 * 12, dtype=torch.float32
        ).reshape(count, grid_size**2, 12),
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
        grid_size=grid_size,
        cache_schema=ceiling.DINO_GRID_CEILING_CACHE_SCHEMA,
        cache_fingerprint="1" * 64,
        feature_identity=ceiling.DINO_GRID_CEILING_FEATURE_IDENTITIES[grid_size],
        embedded_state_cache_fingerprint="a" * 64,
        source_grid4_cache_fingerprint="b" * 64,
        selection_identity="c" * 64,
        split_name=role,
        row_set_identity=("d" if role == "all_train" else "e") * 64,
        image_preprocessing={
            "size": 16,
            "resample": "bicubic",
            "range": [-1, 1],
            "color_space": "sRGB",
        },
    )


def _install_tiny(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ceiling, "IMAGE_SIZE", 16)
    monkeypatch.setattr(ceiling, "TOKEN_DIM", 12)
    monkeypatch.setattr(ceiling, "BASE_CHANNELS", 4)
    monkeypatch.setattr(ceiling, "CONDITION_DIM", 8)
    monkeypatch.setattr(ceiling, "TIME_DIM", 16)
    monkeypatch.setattr(ceiling, "TRAIN_ITEMS", 4)
    monkeypatch.setattr(ceiling, "EXTERNAL_ITEMS", 4)
    monkeypatch.setattr(ceiling, "BATCH_SIZE", 2)
    monkeypatch.setattr(ceiling, "FINAL_STEP", 4)
    monkeypatch.setattr(ceiling, "EVAL_INTERVAL", 2)
    monkeypatch.setattr(ceiling, "SAVE_INTERVAL", 2)
    monkeypatch.setattr(ceiling, "SAMPLE_ROWS", 3)


def _config(grid_size: int) -> CFMConfig:
    return CFMConfig(
        image_size=16,
        token_count=grid_size**2,
        token_dim=12,
        base_channels=4,
        condition_dim=8,
        time_dim=16,
    )


def _invariants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    cell: str = "spatial_dino8",
) -> dict:
    _install_tiny(monkeypatch)
    grid_size = ceiling.CELLS[cell]["grid_size"]
    return ceiling.build_dino_grid_ceiling_invariants(
        cell=cell,
        config=_config(grid_size),
        train=_split(grid_size=grid_size, role="all_train", marker="1"),
        validation=_split(
            grid_size=grid_size,
            role="external_validation",
            marker="2",
        ),
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
        source_commit="f" * 40,
        multigrid_cache_path="/tmp/multigrid-cache",
        output_dir=str(tmp_path.resolve()),
        wandb_entity=ceiling.WANDB_ENTITY,
        wandb_project="nimloth-recon",
        wandb_mode="online",
        wandb_run_id="test-run",
        wandb_run_name="test-name",
    )


def test_cells_are_direct_dino_only_and_models_have_equal_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny(monkeypatch)
    assert ceiling.CELLS == {
        "spatial_dino8": {
            "decoder_family": "spatial_grid_v1",
            "condition_family": "direct_dino",
            "grid_size": 8,
        },
        "spatial_dino16": {
            "decoder_family": "spatial_grid_v1",
            "condition_family": "direct_dino",
            "grid_size": 16,
        },
    }
    models = [ceiling.build_dino_grid_ceiling_model(_config(grid)) for grid in (8, 16)]
    assert {model.decoder_family for model in models} == {"spatial_grid_v1"}
    assert len({sum(parameter.numel() for parameter in model.parameters()) for model in models}) == 1
    assert all(model.config.token_count in {64, 256} for model in models)


def test_invariants_lock_direct_feature_grid_budget_and_exclude_state_condition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invariants = _invariants(monkeypatch, tmp_path)
    assert invariants["cell"] == "spatial_dino8"
    assert invariants["condition_owner"] == "direct_original_observation_dino"
    assert invariants["grid_size"] == 8
    assert invariants["token_count"] == 64
    assert invariants["trainable_owner"] == "decoder_only"
    assert invariants["wandb_entity"] == ceiling.WANDB_ENTITY
    assert invariants["final_step"] == 4
    assert invariants["source_grid4_cache_fingerprint"] == "b" * 64
    assert "state" not in invariants["condition_owner"]

    for field, value in (
        ("validation_cache_fingerprint", "9" * 64),
        ("feature_identity", "8" * 64),
        ("image_preprocessing", {"size": 16}),
    ):
        mutated = {**invariants, field: value}
        with pytest.raises(ValueError, match="invariant|cache|grid"):
            ceiling.validate_dino_grid_ceiling_invariants(mutated, _config(8))

    grid8 = _split(grid_size=8, role="all_train", marker="1")
    bad = replace(grid8, grid_size=16)
    with pytest.raises(ValueError, match="grid|condition|split"):
        ceiling.validate_dino_grid_ceiling_split_pair(
            bad,
            _split(grid_size=8, role="external_validation", marker="2"),
        )


def test_checkpoint_resume_is_exact_and_rejects_cross_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid8_invariants = _invariants(monkeypatch, tmp_path, cell="spatial_dino8")
    model8 = ceiling.build_dino_grid_ceiling_model(_config(8))
    optimizer8 = ceiling.build_decoder_optimizer(
        model8, learning_rate=1e-4, weight_decay=1e-4
    )
    checkpoint = tmp_path / "checkpoint_000000002.pt"
    torch.manual_seed(101)
    ceiling.save_dino_grid_ceiling_checkpoint(
        checkpoint,
        model=model8,
        optimizer=optimizer8,
        step=2,
        invariants=grid8_invariants,
    )
    expected_random = torch.rand(8)
    with torch.no_grad():
        next(model8.parameters()).zero_()
    torch.manual_seed(999)
    assert (
        ceiling.load_dino_grid_ceiling_checkpoint(
            checkpoint,
            model=model8,
            optimizer=optimizer8,
            expected_invariants=grid8_invariants,
            device=torch.device("cpu"),
        )
        == 2
    )
    torch.testing.assert_close(torch.rand(8), expected_random, rtol=0, atol=0)

    for index, (option, value) in enumerate(
        (("lr", 2e-4), ("foreach", True), ("capturable", True), ("fused", True))
    ):
        corrupt_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        corrupt_payload["optimizer"]["param_groups"][0][option] = value
        corrupt_checkpoint = tmp_path / f"corrupt-{index}.pt"
        torch.save(corrupt_payload, corrupt_checkpoint)
        with pytest.raises(ValueError, match="AdamW|hyperparameter|optimizer"):
            ceiling.load_dino_grid_ceiling_checkpoint(
                corrupt_checkpoint,
                model=model8,
                optimizer=optimizer8,
                expected_invariants=grid8_invariants,
                device=torch.device("cpu"),
            )

    grid16_invariants = _invariants(monkeypatch, tmp_path, cell="spatial_dino16")
    model16 = ceiling.build_dino_grid_ceiling_model(_config(16))
    optimizer16 = ceiling.build_decoder_optimizer(
        model16, learning_rate=1e-4, weight_decay=1e-4
    )
    with pytest.raises(ValueError, match="grid|cell|invariant|resume"):
        ceiling.load_dino_grid_ceiling_checkpoint(
            checkpoint,
            model=model16,
            optimizer=optimizer16,
            expected_invariants=grid16_invariants,
            device=torch.device("cpu"),
        )


def test_output_protection_and_cli_require_explicit_cell_and_wandb(
    tmp_path: Path,
) -> None:
    cache = (tmp_path / "multigrid").resolve()
    cache.mkdir()
    with pytest.raises(ValueError, match="input|inside|output"):
        ceiling._reject_protected_output(cache / "run", (cache,))

    parser = ceiling.build_cli_parser()
    args = parser.parse_args(
        [
            "--cell",
            "spatial_dino8",
            "--multigrid-cache",
            str(cache),
            "--output-dir",
            str((tmp_path / "run").resolve()),
            "--device",
            "cpu",
            "--wandb-entity",
            ceiling.WANDB_ENTITY,
            "--wandb-project",
            "nimloth-recon",
            "--wandb-mode",
            "online",
            "--wandb-run-id",
            "run-id",
            "--wandb-run-name",
            "run-name",
        ]
    )
    assert args.cell == "spatial_dino8"
    assert args.wandb_entity == ceiling.WANDB_ENTITY
    assert args.wandb_run_id == "run-id"
    assert not hasattr(args, "state_cache")


def test_wandb_finish_records_explicit_success_and_failure() -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.exit_codes: list[int] = []

        def finish(self, *, exit_code: int) -> None:
            self.exit_codes.append(exit_code)

    run = FakeRun()
    ceiling._finish_wandb_run(run, succeeded=True)
    ceiling._finish_wandb_run(run, succeeded=False)
    assert run.exit_codes == [0, 1]


def test_training_pins_entity_and_finishes_success_or_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny(monkeypatch)
    monkeypatch.setattr(ceiling, "_current_source_commit", lambda: "f" * 40)
    monkeypatch.setattr(
        ceiling,
        "load_dino_grid_ceiling_splits",
        lambda **_kwargs: (
            _split(grid_size=8, role="all_train", marker="1"),
            _split(grid_size=8, role="external_validation", marker="2"),
        ),
    )

    class FakeRun:
        entity = ceiling.WANDB_ENTITY
        project = "nimloth-recon"
        id = "run-id"
        name = "run-name"

        def __init__(self) -> None:
            self.exit_codes: list[int] = []

        def log(self, *_args, **_kwargs) -> None:
            pass

        def finish(self, *, exit_code: int) -> None:
            self.exit_codes.append(exit_code)

    runs: list[FakeRun] = []
    init_kwargs: list[dict] = []

    def fake_init(**kwargs):
        init_kwargs.append(kwargs)
        run = FakeRun()
        runs.append(run)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))

    def args(output: Path) -> SimpleNamespace:
        return SimpleNamespace(
            cell="spatial_dino8",
            multigrid_cache=(tmp_path / "cache").resolve(),
            output_dir=output.resolve(),
            device=torch.device("cpu"),
            resume=False,
            resume_checkpoint=None,
            wandb_entity=ceiling.WANDB_ENTITY,
            wandb_project="nimloth-recon",
            wandb_mode="online",
            wandb_run_id="run-id",
            wandb_run_name="run-name",
        )

    assert ceiling.train_dino_grid_ceiling_cfm(args(tmp_path / "success")) == 0
    assert init_kwargs[0]["entity"] == ceiling.WANDB_ENTITY
    assert runs[0].exit_codes == [0]

    monkeypatch.setattr(
        ceiling,
        "conditional_flow_matching_loss",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("train failed")),
    )
    with pytest.raises(RuntimeError, match="train failed"):
        ceiling.train_dino_grid_ceiling_cfm(args(tmp_path / "failure"))
    assert runs[1].exit_codes == [1]

    FakeRun.entity = "wrong-entity"
    with pytest.raises(RuntimeError, match="different run identity"):
        ceiling.train_dino_grid_ceiling_cfm(args(tmp_path / "identity-mismatch"))
    assert runs[2].exit_codes == [1]
