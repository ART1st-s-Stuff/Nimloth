from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from nimloth.eval import dino_grid_ceiling_decision as ceiling_decision
from nimloth.eval import dino_grid_reconstruction_ceiling as ceiling_eval


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _rows(count: int) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "row_identity": f"external-{index}",
            "original_image_sha256": _sha(f"image-{index}"),
            "selection_role": "external_validation",
        }
        for index in range(count)
    )


def test_evaluator_cli_has_no_sft1_state_cache_input() -> None:
    help_text = ceiling_eval.build_cli_parser().format_help()
    assert "--state-cache" not in help_text
    assert "--grid4-cache" in help_text
    assert "--multigrid-cache" in help_text


def test_multiscale_metrics_average_noise_inside_rows_and_keep_dino_owners() -> None:
    rows = _rows(2)
    source_rgb = torch.zeros(2, 3, 64, 64)
    correct_rgb = torch.stack(
        [source_rgb + value for value in (0.1, 0.2, 0.3)], dim=1
    )
    shuffled_rgb = torch.stack(
        [source_rgb + value for value in (0.4, 0.5, 0.6)], dim=1
    )
    source_dino = {
        4: torch.ones(2, 16, 4),
        16: torch.ones(2, 256, 4),
    }
    correct_dino = {
        grid: torch.stack([value + offset for offset in (0.1, 0.2, 0.3)], dim=1)
        for grid, value in source_dino.items()
    }
    shuffled_dino = {
        grid: torch.stack([value + offset for offset in (0.4, 0.5, 0.6)], dim=1)
        for grid, value in source_dino.items()
    }

    report = ceiling_eval.sample_multiscale_metrics(
        rows=rows,
        noise_seeds=(11, 29, 47),
        source_rgb=source_rgb,
        correct_rgb=correct_rgb,
        shuffled_rgb=shuffled_rgb,
        source_dino=source_dino,
        correct_dino=correct_dino,
        shuffled_dino=shuffled_dino,
    )

    assert report["statistical_unit"] == "external_validation_row"
    assert report["seed_aggregation"] == "mean_within_row_before_row_bootstrap"
    assert report["source_dino_input_owner"] == "original_archived_observation"
    assert report["generated_dino_input_owner"] == "decoder_generated_128px_srgb"
    assert report["generated_dino_is_canonical_teacher_target"] is False
    first = report["rows"][0]
    assert first["correct_rgb128_l1"] == pytest.approx(0.2)
    assert first["correct_rgb32_l1"] == pytest.approx(0.2)
    assert first["correct_rgb16_l1"] == pytest.approx(0.2)
    assert first["correct_rgb32_gradient_l1"] == pytest.approx(0.0)
    assert first["correct_rgb16_gradient_l1"] == pytest.approx(0.0)
    assert first["correct_dino4_mse"] == pytest.approx((0.01 + 0.04 + 0.09) / 3)
    assert first["correct_dino16_cosine"] == pytest.approx(1.0)
    assert len(first["per_seed"]) == 3


def _cell_report(cell: str, improvement: float) -> dict:
    return {
        "cell": cell,
        "row_set_identity": "a" * 64,
        "sample_selection_identity": "b" * 64,
        "noise_seeds": [11, 29, 47],
        "rows": [
            {
                "row_identity": row["row_identity"],
                "correct_rgb128_l1": 0.50 - improvement,
                "correct_rgb128_rmse": 0.60 - improvement,
                "correct_rgb32_l1": 0.40 - improvement,
                "correct_rgb32_rmse": 0.50 - improvement,
                "correct_rgb32_gradient_l1": 0.30 - improvement,
                "correct_rgb16_l1": 0.35 - improvement,
                "correct_rgb16_rmse": 0.45 - improvement,
                "correct_rgb16_gradient_l1": 0.25 - improvement,
                "correct_dino4_mse": 0.70 - improvement,
                "correct_dino4_cosine": 0.20 + improvement,
                "correct_dino16_mse": 0.80 - improvement,
                "correct_dino16_cosine": 0.10 + improvement,
            }
            for row in _rows(16)
        ],
    }


def test_three_grid_comparison_uses_paired_rows_and_preregistered_statistics() -> None:
    reports = {
        "grid4": _cell_report("grid4", 0.00),
        "grid8": _cell_report("grid8", 0.08),
        "grid16": _cell_report("grid16", 0.12),
    }
    comparison = ceiling_eval.build_three_grid_comparison(
        reports,
        bootstrap_seed=20260951,
        bootstrap_resamples=2000,
    )

    assert comparison["cells"] == ["grid4", "grid8", "grid16"]
    assert comparison["statistical_unit"] == "external_validation_row"
    assert comparison["comparisons"]["grid8_vs_grid4"]["rgb32_l1_improvement"][
        "mean"
    ] == pytest.approx(0.08)
    assert comparison["comparisons"]["grid8_vs_grid4"]["statistical_pass"] is True
    assert comparison["comparisons"]["grid16_vs_grid4"]["statistical_pass"] is True
    assert comparison["decision_contract"] == {
        "human_fixed_rows": 16,
        "human_required_recognizable": 12,
        "statistical_metrics": [
            "rgb32_l1_improvement",
            "rgb32_gradient_l1_improvement",
        ],
        "statistical_rule": "both_paired_95ci_lower_bounds_gt_zero",
        "recommendation": "smallest_grid_passing_human_and_statistical_rules",
    }

    drifted = dict(reports)
    drifted["grid16"] = {**drifted["grid16"], "noise_seeds": [1, 2, 3]}
    with pytest.raises(ValueError, match="matched|seed|row"):
        ceiling_eval.build_three_grid_comparison(
            drifted, bootstrap_seed=1, bootstrap_resamples=10
        )


def test_human_structure_gate_requires_exact_fixed16_and_selects_smallest_pass() -> None:
    comparison = ceiling_eval.build_three_grid_comparison(
        {
            "grid4": _cell_report("grid4", 0.00),
            "grid8": _cell_report("grid8", 0.08),
            "grid16": _cell_report("grid16", 0.12),
        },
        bootstrap_seed=20260951,
        bootstrap_resamples=100,
    )
    decision = ceiling_eval.apply_human_structure_judgments(
        comparison,
        judgments={
            "grid8": (True,) * 12 + (False,) * 4,
            "grid16": (True,) * 16,
        },
    )
    assert decision["grids"]["grid8"]["human_recognizable"] == 12
    assert decision["grids"]["grid8"]["passed"] is True
    assert decision["recommended_grid"] == 8

    with pytest.raises(ValueError, match="16|judgment"):
        ceiling_eval.apply_human_structure_judgments(
            comparison,
            judgments={"grid8": (True,) * 15, "grid16": (True,) * 16},
        )


def test_human_decision_publication_binds_report_contact_and_each_id198_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = ceiling_eval.build_three_grid_comparison(
        {
            "grid4": _cell_report("grid4", 0.00),
            "grid8": _cell_report("grid8", 0.08),
            "grid16": _cell_report("grid16", 0.12),
        },
        bootstrap_seed=20260951,
        bootstrap_resamples=100,
    )
    rows = _rows(16)
    report = {
        "artifact_identity": "f" * 64,
        "comparison": comparison,
        "contact_sheet": {
            "identity": "c" * 64,
            "png_sha256": "d" * 64,
            "row_identities": [row["row_identity"] for row in rows],
            "image_sha256": [row["original_image_sha256"] for row in rows],
        },
    }
    report_root = (tmp_path / "report").resolve()
    report_root.mkdir()
    (report_root / "manifest.json").write_text("{}\n")
    monkeypatch.setattr(
        ceiling_decision,
        "validate_dino_grid_ceiling_report",
        lambda _root: report,
    )
    judgment_rows = [
        {
            "row_identity": row["row_identity"],
            "original_image_sha256": row["original_image_sha256"],
            "grid8_recognizable": index < 12,
            "grid16_recognizable": True,
        }
        for index, row in enumerate(rows)
    ]
    judgments = {
        "schema": ceiling_decision.DINO_GRID_CEILING_JUDGMENTS_SCHEMA,
        "report_artifact_identity": report["artifact_identity"],
        "contact_sheet_identity": report["contact_sheet"]["identity"],
        "contact_sheet_sha256": report["contact_sheet"]["png_sha256"],
        "rows": judgment_rows,
    }
    judgments["identity"] = ceiling_eval._mapping_identity(judgments)
    judgments_path = (tmp_path / "judgments.json").resolve()
    judgments_path.write_text(json.dumps(judgments) + "\n")

    output = (tmp_path / "decision").resolve()
    manifest = ceiling_decision.publish_human_decision(
        output,
        report_root=report_root,
        judgments_path=judgments_path,
    )
    assert manifest["decision"]["recommended_grid"] == 8
    assert manifest["source_report"]["contact_sheet_identity"] == "c" * 64
    assert manifest["human_judgments"]["sha256"] == ceiling_eval._sha256_file(
        judgments_path
    )

    tampered = {**judgments, "rows": list(reversed(judgment_rows))}
    tampered["identity"] = ceiling_eval._mapping_identity(
        {key: value for key, value in tampered.items() if key != "identity"}
    )
    with pytest.raises(ValueError, match="row|order|image"):
        ceiling_decision.validate_human_judgments(tampered, report=report)


def test_contact_sheet_has_fixed_original_grid4_grid8_grid16_columns(
    tmp_path: Path,
) -> None:
    rows = _rows(2)
    originals = [Image.new("RGB", (8, 8), (20, 30, 40)) for _ in rows]
    samples = {
        cell: [Image.new("RGB", (8, 8), color) for _ in rows]
        for cell, color in {
            "grid4": (50, 0, 0),
            "grid8": (0, 50, 0),
            "grid16": (0, 0, 50),
        }.items()
    }
    output = tmp_path / "contact.png"
    metadata = ceiling_eval.save_three_grid_contact_sheet(
        output,
        rows=rows,
        originals=originals,
        correct_samples=samples,
        sample_indices=(0, 1),
        sample_seed=20260921,
        initial_noise_sha256="e" * 64,
    )
    try:
        assert metadata["columns"] == ["original", "grid4", "grid8", "grid16"]
        assert metadata["human_judgment_rule"] == "at_least_12_of_fixed_16"
        assert metadata["controls_scientific_result"] is True
        assert len(metadata["png_sha256"]) == 64
        with Image.open(output) as image:
            assert image.width == 4 * 8
    finally:
        for image in originals:
            image.close()
        for images in samples.values():
            for image in images:
                image.close()
