import pytest
import torch

from nimloth.eval.query_cfm_teacher_forced import COLUMNS, calculate_metrics


def test_requested_columns_are_exact_and_ordered() -> None:
    assert COLUMNS == [
        "GT",
        "Qwen ViT-token CFM",
        "query-latent CFM",
        "WM pred + Decoder + CFM",
    ]


def test_metrics_report_decoder_and_image_quality() -> None:
    query = torch.zeros(2, 2, 3)
    decoded = torch.ones_like(query)
    gt = torch.zeros(2, 3, 2, 2)
    images = {
        "qwen": torch.full_like(gt, 0.1),
        "query": torch.full_like(gt, 0.2),
        "predicted": torch.full_like(gt, 0.3),
    }
    metrics = calculate_metrics(query, decoded, images, gt)
    assert metrics["decoder/predicted_to_query_mse"] == pytest.approx(1.0)
    assert metrics["image/qwen_to_gt_l1"] == pytest.approx(0.1)
    assert metrics["image/query_to_gt_l1"] == pytest.approx(0.2)
    assert metrics["image/predicted_to_gt_l1"] == pytest.approx(0.3)
    assert metrics["image/query_predicted_output_l1"] == pytest.approx(0.1)
