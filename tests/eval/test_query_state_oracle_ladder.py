from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from nimloth.eval import query_state_oracle_ladder as oracle_eval


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _rows(count: int = 8) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "row_identity": f"external-{index}",
            "original_image_sha256": _sha(f"image-{index}"),
            "selection_role": "external_validation",
        }
        for index in range(count)
    )


def test_sample_plan_is_identity_only_order_invariant_and_exact() -> None:
    rows = _rows()
    first = oracle_eval.build_sample_plan(rows, count=4)
    second = oracle_eval.build_sample_plan(tuple(reversed(rows)), count=4)
    expected = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"oracle-ladder-v1:{row['row_identity']}".encode()
        ).hexdigest(),
    )[:4]

    assert first.algorithm == "sha256_oracle_ladder_v1"
    assert first.row_identities == tuple(row["row_identity"] for row in expected)
    assert second.row_identities == first.row_identities
    assert len(first.identity) == 64
    assert first.identity == second.identity
    assert all(rows[index]["row_identity"] in first.row_identities for index in first.indices)

    duplicate = (*rows[:-1], dict(rows[0]))
    with pytest.raises(ValueError, match="unique|row identity"):
        oracle_eval.build_sample_plan(duplicate, count=4)
    with pytest.raises(ValueError, match="count"):
        oracle_eval.build_sample_plan(rows, count=9)


def test_paired_row_bootstrap_is_deterministic_and_preserves_effect_direction() -> None:
    differences = torch.tensor([0.1, 0.2, 0.0, 0.3], dtype=torch.float64)
    first = oracle_eval.paired_row_bootstrap(
        differences,
        seed=20260951,
        resamples=2000,
    )
    second = oracle_eval.paired_row_bootstrap(
        differences,
        seed=20260951,
        resamples=2000,
    )

    assert first == second
    assert first["row_count"] == 4
    assert first["mean"] == pytest.approx(0.15)
    assert first["lower_95"] >= 0.0
    assert first["upper_95"] > first["mean"]
    assert first["seed"] == 20260951
    assert first["resamples"] == 2000
    with pytest.raises(ValueError, match="finite|one-dimensional"):
        oracle_eval.paired_row_bootstrap(
            torch.tensor([[float("nan")]]), seed=1, resamples=10
        )


def test_sample_pair_metrics_use_rows_as_units_after_averaging_noise_seeds() -> None:
    source_rgb = torch.zeros(2, 3, 8, 8)
    source_dino = torch.zeros(2, 16, 4)
    correct_rgb = torch.stack(
        [source_rgb + 0.1, source_rgb + 0.2, source_rgb + 0.3], dim=1
    )
    shuffled_rgb = torch.stack(
        [source_rgb + 0.4, source_rgb + 0.5, source_rgb + 0.6], dim=1
    )
    correct_dino = torch.stack(
        [source_dino + 0.1, source_dino + 0.2, source_dino + 0.3], dim=1
    )
    shuffled_dino = torch.stack(
        [source_dino + 0.4, source_dino + 0.5, source_dino + 0.6], dim=1
    )

    report = oracle_eval.sample_pair_metrics(
        rows=_rows(2),
        noise_seeds=(11, 29, 47),
        source_rgb=source_rgb,
        source_dino=source_dino,
        correct_rgb=correct_rgb,
        shuffled_rgb=shuffled_rgb,
        correct_dino=correct_dino,
        shuffled_dino=shuffled_dino,
    )

    assert report["statistical_unit"] == "external_validation_row"
    assert report["seed_aggregation"] == "mean_within_row_before_row_bootstrap"
    assert report["noise_seeds"] == [11, 29, 47]
    assert len(report["rows"]) == 2
    assert report["rows"][0]["correct_rgb_l1"] == pytest.approx(0.2)
    assert report["rows"][0]["shuffled_rgb_l1"] == pytest.approx(0.5)
    assert report["rows"][0]["correct_dino_mse"] == pytest.approx(
        (0.01 + 0.04 + 0.09) / 3
    )
    assert report["rows"][0]["shuffled_dino_mse"] == pytest.approx(
        (0.16 + 0.25 + 0.36) / 3
    )
    assert report["rows"][0]["rgb_l1_shuffled_minus_correct"] > 0
    assert report["rows"][0]["dino_mse_shuffled_minus_correct"] > 0
    assert len(report["identity"]) == 64


def test_strict_sample_reader_rejects_rehashed_bootstrap_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oracle_eval, "SAMPLE_ROWS", 2)
    monkeypatch.setattr(oracle_eval, "NOISE_SEEDS", (11, 29, 47))
    monkeypatch.setattr(oracle_eval, "IMAGE_SIZE", 8)
    rows = _rows(2)
    source_rgb = torch.zeros(2, 3, 8, 8)
    source_dino = torch.zeros(2, 16, 4)
    correct_rgb = torch.stack([source_rgb + value for value in (0.1, 0.2, 0.3)], dim=1)
    shuffled_rgb = torch.stack([source_rgb + value for value in (0.4, 0.5, 0.6)], dim=1)
    correct_dino = torch.stack([source_dino + value for value in (0.1, 0.2, 0.3)], dim=1)
    shuffled_dino = torch.stack([source_dino + value for value in (0.4, 0.5, 0.6)], dim=1)
    metrics = oracle_eval.sample_pair_metrics(
        rows=rows,
        noise_seeds=(11, 29, 47),
        source_rgb=source_rgb,
        source_dino=source_dino,
        correct_rgb=correct_rgb,
        shuffled_rgb=shuffled_rgb,
        correct_dino=correct_dino,
        shuffled_dino=shuffled_dino,
    )
    sampling = []
    for seed in (11, 29, 47):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((2, 3, 8, 8), generator=generator)
        evidence = {
            "seed": seed,
            "initial_noise_sha256": oracle_eval._tensor_sha256(noise),
            "shuffle_algorithm": oracle_eval.QUERY_STATE_SHUFFLE_ALGORITHM,
            "shuffle_mapping": [1, 0],
            "shuffle_donor_row_identities": ["external-1", "external-0"],
            "ode_solver": "midpoint_euler",
            "ode_steps": 50,
        }
        sampling.append(
            {**evidence, "identity": oracle_eval._mapping_identity(evidence)}
        )
    report = {
        **{key: value for key, value in metrics.items() if key != "identity"},
        "sampling_evidence": sampling,
    }
    report["identity"] = oracle_eval._mapping_identity(report)
    oracle_eval._validate_sample_metric_evidence(report)

    mutated = json.loads(json.dumps(report))
    mutated["paired_row_bootstrap"]["rgb_l1_shuffled_minus_correct"] = {
        "mean": 999.0
    }
    mutated["identity"] = oracle_eval._mapping_identity(
        {key: value for key, value in mutated.items() if key != "identity"}
    )
    with pytest.raises(ValueError, match="bootstrap|metric"):
        oracle_eval._validate_sample_metric_evidence(mutated)

    rehashed_aggregate = json.loads(json.dumps(report))
    rehashed_aggregate["rows"][0]["correct_dino_mse"] += 0.25
    rehashed_aggregate["identity"] = oracle_eval._mapping_identity(
        {
            key: value
            for key, value in rehashed_aggregate.items()
            if key != "identity"
        }
    )
    with pytest.raises(ValueError, match="aggregation"):
        oracle_eval._validate_sample_metric_evidence(rehashed_aggregate)

    rehashed_delta = json.loads(json.dumps(report))
    rehashed_delta["rows"][0]["rgb_rmse_shuffled_minus_correct"] += 0.25
    rehashed_delta["identity"] = oracle_eval._mapping_identity(
        {key: value for key, value in rehashed_delta.items() if key != "identity"}
    )
    with pytest.raises(ValueError, match="delta"):
        oracle_eval._validate_sample_metric_evidence(rehashed_delta)


def test_strict_fixed_time_reader_rejects_rehashed_metric_algebra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oracle_eval, "EXTERNAL_ITEMS", 2)
    monkeypatch.setattr(oracle_eval, "NOISE_SEEDS", (11,))
    monkeypatch.setattr(oracle_eval, "FIXED_TIMES", (0.0,))
    mapping = oracle_eval.make_global_shuffle_mapping(item_count=2, seed=11)
    item = {
        "seed": 11,
        "time": 0.0,
        "target_rgb_fraction_in_model_input": 0.0,
        "matched_noise": True,
        "noise_sha256": "d" * 64,
        "time_sha256": "e" * 64,
        "shuffle_algorithm": oracle_eval.QUERY_STATE_SHUFFLE_ALGORITHM,
        "shuffle_identity": oracle_eval._mapping_identity(
            {
                "seed": 11,
                "mapping": mapping.tolist(),
                "algorithm": oracle_eval.QUERY_STATE_SHUFFLE_ALGORITHM,
            }
        ),
        "num_items": 2,
        "correct_flow_mse": 1.0,
        "shuffled_flow_mse": 2.0,
        "shuffled_minus_correct": 1.0,
        "shuffled_over_correct": 2.0,
    }
    report = {
        "schema": "nimloth_query_state_oracle_ladder_fixed_time_v1",
        "metric_unit": "mean velocity MSE per normalized RGB element",
        "num_items": 2,
        "seeds": [11],
        "fixed_times": [0.0],
        "per_seed_time": [item],
    }
    report["identity"] = oracle_eval._mapping_identity(report)
    oracle_eval._validate_fixed_time_evidence(report)

    report["per_seed_time"][0]["shuffled_minus_correct"] = 0.5
    report["identity"] = oracle_eval._mapping_identity(
        {key: value for key, value in report.items() if key != "identity"}
    )
    with pytest.raises(ValueError, match="fixed-time|item"):
        oracle_eval._validate_fixed_time_evidence(report)


def _cell_report(cell: str, offset: float) -> dict:
    decoder, condition = cell.split("_", maxsplit=1)
    decoder_family = "token_set_v1" if decoder == "token" else "spatial_grid_v1"
    condition_family = "state" if condition == "state" else "dino_oracle"
    return {
        "cell": cell,
        "decoder_family": decoder_family,
        "condition_family": condition_family,
        "row_set_identity": "a" * 64,
        "sample_selection_identity": "b" * 64,
        "noise_seeds": [11, 29, 47],
        "rows": [
            {
                "row_identity": row["row_identity"],
                "correct_dino_cosine": 0.2 + offset + index / 100,
                "correct_dino_mse": 1.0 - offset - index / 100,
                "correct_rgb_l1": 0.8 - offset - index / 100,
                "correct_rgb_rmse": 0.9 - offset - index / 100,
                "rgb_l1_shuffled_minus_correct": 0.05 + offset,
                "rgb_rmse_shuffled_minus_correct": 0.06 + offset,
                "dino_mse_shuffled_minus_correct": 0.04 + offset,
                "dino_cosine_correct_minus_shuffled": 0.03 + offset,
            }
            for index, row in enumerate(_rows(4))
        ],
    }


def test_four_cell_comparison_requires_matched_rows_and_reports_factor_effects() -> None:
    reports = {
        "token_state": _cell_report("token_state", 0.00),
        "token_oracle": _cell_report("token_oracle", 0.10),
        "spatial_state": _cell_report("spatial_state", 0.05),
        "spatial_oracle": _cell_report("spatial_oracle", 0.15),
    }
    comparison = oracle_eval.build_four_cell_comparison(
        reports,
        bootstrap_seed=20260951,
        bootstrap_resamples=2000,
    )

    assert comparison["cells"] == [
        "token_state",
        "token_oracle",
        "spatial_state",
        "spatial_oracle",
    ]
    assert comparison["effects"]["teacher_at_token"]["correct_dino_cosine"][
        "mean"
    ] == pytest.approx(0.1)
    assert comparison["effects"]["decoder_at_state"]["correct_dino_cosine"][
        "mean"
    ] == pytest.approx(0.05)
    assert comparison["effects"]["teacher_at_spatial"]["correct_rgb_l1"][
        "mean"
    ] == pytest.approx(0.1)
    assert comparison["effects"]["decoder_at_oracle"]["correct_dino_mse"][
        "mean"
    ] == pytest.approx(0.05)
    assert comparison["claim_scope"] == "forensic_representation_decodability_only"
    assert len(comparison["identity"]) == 64

    missing = dict(reports)
    missing.pop("spatial_oracle")
    with pytest.raises(ValueError, match="four|cell"):
        oracle_eval.build_four_cell_comparison(
            missing, bootstrap_seed=1, bootstrap_resamples=10
        )
    drifted = {key: dict(value) for key, value in reports.items()}
    drifted["spatial_oracle"] = {
        **drifted["spatial_oracle"],
        "row_set_identity": "f" * 64,
    }
    with pytest.raises(ValueError, match="matched|row"):
        oracle_eval.build_four_cell_comparison(
            drifted, bootstrap_seed=1, bootstrap_resamples=10
        )


def test_id198_visual_reference_is_byte_and_pair_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indices = tuple(range(16))
    rows = [
        {
            "row_identity": _sha(f"id198-row-{index}"),
            "image_sha256": _sha(f"id198-image-{index}"),
        }
        for index in indices
    ]
    summary = {
        "schema": "nimloth_forensic_query_state_feature_report_v1",
        "experiment_stage": "stage_b_diagnostic",
        "cache_fingerprint": (
            "9bd942267140aede839087b09bb0f755bd023ec69d216a8ad77ee845fc120899"
        ),
        "roles": {"external_validation": {"visual_count": 16}},
    }
    external = {
        "visual_selection": {
            "indices": list(indices),
            "seed": 20260921,
            "row_identities": [row["row_identity"] for row in rows],
        },
        "rows": rows,
    }
    summary_path = (tmp_path / "summary.json").resolve()
    external_path = (tmp_path / "external.json").resolve()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    external_path.write_text(json.dumps(external), encoding="utf-8")
    pairs = [
        {
            "row_identity": row["row_identity"],
            "original_image_sha256": row["image_sha256"],
        }
        for row in rows
    ]
    monkeypatch.setattr(oracle_eval, "STAGE_B_SAMPLE_INDICES", indices)
    monkeypatch.setattr(
        oracle_eval,
        "ID198_SUMMARY_SHA256",
        hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        oracle_eval,
        "ID198_EXTERNAL_REPORT_SHA256",
        hashlib.sha256(external_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        oracle_eval,
        "ID198_ROW_IMAGE_PAIR_IDENTITY",
        oracle_eval._mapping_identity({"pairs": pairs}),
    )
    reference = oracle_eval._load_id198_visual_reference(
        summary_path=summary_path,
        external_report_path=external_path,
    )
    assert reference["row_image_pairs"] == pairs

    reordered = {
        **external,
        "visual_selection": {
            **external["visual_selection"],
            "row_identities": [row["row_identity"] for row in rows[::-1]],
        },
        "rows": rows[::-1],
    }
    external_path.write_text(json.dumps(reordered), encoding="utf-8")
    monkeypatch.setattr(
        oracle_eval,
        "ID198_EXTERNAL_REPORT_SHA256",
        hashlib.sha256(external_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="pair identity"):
        oracle_eval._load_id198_visual_reference(
            summary_path=summary_path,
            external_report_path=external_path,
        )


def test_contact_sheet_has_original_plus_four_fixed_cell_columns(
    tmp_path: Path,
) -> None:
    rows = _rows(2)
    originals = [Image.new("RGB", (8, 8), (20, 30, 40)) for _ in rows]
    samples = {
        cell: [Image.new("RGB", (8, 8), color) for _ in rows]
        for cell, color in {
            "token_state": (50, 0, 0),
            "token_oracle": (0, 50, 0),
            "spatial_state": (0, 0, 50),
            "spatial_oracle": (50, 50, 0),
        }.items()
    }
    output = tmp_path / "contact.png"
    metadata = oracle_eval.save_four_cell_contact_sheet(
        output,
        rows=rows,
        originals=originals,
        correct_samples=samples,
        sample_seed=20260921,
        sample_indices=(0, 1),
        sample_indices_sha256="d" * 64,
        initial_noise_sha256="e" * 64,
    )
    try:
        with Image.open(output) as image:
            assert image.mode == "RGB"
            assert image.width == 5 * 8
            assert image.height > 2 * 8
        assert metadata["columns"] == [
            "original",
            "token_state",
            "token_oracle",
            "spatial_state",
            "spatial_oracle",
        ]
        assert metadata["row_identities"] == [
            row["row_identity"] for row in rows
        ]
        assert metadata["image_sha256"] == [
            row["original_image_sha256"] for row in rows
        ]
        assert len(metadata["row_image_pair_identity"]) == 64
        assert metadata["human_inspection_only"] is True
        assert metadata["controls_scientific_result"] is False
        assert len(metadata["png_sha256"]) == 64
    finally:
        for image in originals:
            image.close()
        for images in samples.values():
            for image in images:
                image.close()
