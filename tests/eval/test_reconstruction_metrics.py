import torch

from nimloth.eval.reconstruction import reconstruction_metrics


def test_reconstruction_metrics_prefix_and_values() -> None:
    pred = torch.zeros(1, 3, 4, 4)
    target = torch.ones(1, 3, 4, 4)
    metrics = reconstruction_metrics(pred, target, prefix="pred")
    assert metrics["pred_mse"] == 1.0
    assert metrics["pred_mae"] == 1.0
    assert metrics["pred_psnr"] == 0.0
