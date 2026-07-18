import pytest
import torch

from nimloth.eval.projected_cfm_predicted_trajectory import calculate_metrics


def test_calculate_metrics_reports_horizon_state_and_decoder_gap() -> None:
    rows = [{"horizon": horizon} for horizon in range(1, 6)]
    projected = torch.zeros(5, 4)
    predicted = torch.arange(1, 6).float()[:, None].expand(-1, 4)
    gt = torch.zeros(5, 3, 2, 2)
    images = {
        "qwen": torch.zeros_like(gt),
        "query": torch.ones_like(gt) * 0.1,
        "projected": torch.ones_like(gt) * 0.2,
        "predicted": torch.arange(1, 6).float()[:, None, None, None].expand_as(gt) * 0.1,
    }
    metrics, horizon = calculate_metrics(
        rows, {"projected": projected, "predicted": predicted}, images, gt
    )
    assert metrics["state/predicted_to_actual_mse"] == pytest.approx(11.0)
    assert metrics["image/projected_to_gt_l1"] == pytest.approx(0.2)
    assert horizon["1"]["state_mse"] == pytest.approx(1.0)
    assert horizon["5"]["predicted_image_l1"] == pytest.approx(0.5)
    assert horizon["3"]["actual_predicted_output_l1"] == pytest.approx(0.1)
