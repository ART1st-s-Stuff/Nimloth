import pytest
import torch

from nimloth.eval.query_cfm_teacher_forced import (
    COLUMNS,
    calculate_metrics,
    resolve_condition_shapes,
)
from nimloth.training.reconstruction.projected_query_decoder import (
    ProjectedQueryDecoderConfig,
)


def test_requested_columns_are_exact_and_ordered() -> None:
    assert COLUMNS == [
        "GT",
        "Qwen ViT-token CFM",
        "query-latent CFM",
        "WM pred + Decoder + CFM",
    ]


def test_condition_shapes_support_one_query_token() -> None:
    query_shape, projected_shape = resolve_condition_shapes(
        {"representation": "qwen_query_hidden", "state_shape": [1, 2048]},
        {"representation": "projected", "state_shape": [1024]},
        ProjectedQueryDecoderConfig(
            projected_dim=1024,
            hidden_dim=1024,
            query_tokens=1,
            query_dim=2048,
        ),
    )
    assert query_shape == [1, 2048]
    assert projected_shape == [1024]


def test_condition_shapes_reject_decoder_mismatch() -> None:
    with pytest.raises(ValueError, match="decoder/query cache shape mismatch"):
        resolve_condition_shapes(
            {"representation": "qwen_query_hidden", "state_shape": [1, 2048]},
            {"representation": "projected", "state_shape": [1024]},
            ProjectedQueryDecoderConfig(
                projected_dim=1024,
                hidden_dim=1024,
                query_tokens=8,
                query_dim=2048,
            ),
        )


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
