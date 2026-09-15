from pathlib import Path
from types import SimpleNamespace

import torch

from nimloth.training.sft.stage3.diagnostics import DINOFeatureWriter


def test_dino_feature_writer_keeps_full_spatial_grids(tmp_path: Path) -> None:
    batch = SimpleNamespace(
        prediction_horizon=2,
        batch_size=2,
        sample_weights=torch.tensor([1.0, 0.0]),
        current_keys=[("kept", 3), ("padding", 0)],
        action_sequences=torch.tensor([[1, 2], [3, 4]]),
    )
    grids = torch.arange(2 * 2 * 4 * 3, dtype=torch.float32).reshape(4, 4, 3)
    output = SimpleNamespace(
        diagnostics={
            "predicted_states": grids,
            "target_states": grids + 1,
            "dino_targets": grids + 2,
            "current_dino_targets": torch.zeros(2, 4, 3),
        }
    )
    writer = DINOFeatureWriter(tmp_path, rank=0)
    writer(batch, output)
    payload = torch.load(writer.paths[0], weights_only=True)
    assert payload["schema"] == "stage3_dino_feature_batch_v1"
    assert payload["keys"] == [("kept", 3)]
    assert payload["predicted"].shape == (1, 2, 4, 3)
    assert payload["direct"].shape == payload["dino"].shape
