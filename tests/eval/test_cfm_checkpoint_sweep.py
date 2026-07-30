from __future__ import annotations

import pytest
import torch

from nimloth.eval.cfm_checkpoint_sweep import prepare_output_dir, reconstruction_metrics


def test_prepare_output_dir_allows_only_lifecycle_readme(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "README.md").write_text("running\n", encoding="utf-8")

    prepare_output_dir(output)

    (output / "contract.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="contract.json"):
        prepare_output_dir(output)


def test_reconstruction_metrics_preserve_matched_row_and_horizon_units() -> None:
    gt = torch.zeros(4, 3, 2, 2)
    actual = torch.stack(
        [torch.full((3, 2, 2), value) for value in (0.1, 0.2, 0.3, 0.4)]
    )
    predicted = torch.stack(
        [torch.full((3, 2, 2), value) for value in (0.2, 0.1, 0.5, 0.2)]
    )

    result = reconstruction_metrics(
        actual_images=actual,
        predicted_images=predicted,
        gt=gt,
        horizons=torch.tensor([1, 1, 2, 2]),
    )

    assert result["image_actual_to_gt_l1"] == pytest.approx(0.25)
    assert result["image_predicted_to_gt_l1"] == pytest.approx(0.25)
    assert result["image_predicted_to_actual_output_l1"] == pytest.approx(0.15)
    assert result["image_predicted_better_frame_fraction"] == pytest.approx(0.5)
    assert result["horizons"]["1"]["count"] == 2
    assert result["horizons"]["1"]["image_predicted_to_gt_l1"] == pytest.approx(0.15)
    assert result["horizons"]["2"]["image_predicted_to_gt_l1"] == pytest.approx(0.35)


def test_reconstruction_metrics_reject_misaligned_rows() -> None:
    with pytest.raises(ValueError, match="image shapes must match"):
        reconstruction_metrics(
            actual_images=torch.zeros(2, 3, 2, 2),
            predicted_images=torch.zeros(1, 3, 2, 2),
            gt=torch.zeros(2, 3, 2, 2),
            horizons=torch.tensor([1, 2]),
        )
