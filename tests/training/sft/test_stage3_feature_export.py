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
        next_indices=torch.arange(4),
        current_indices=torch.tensor([0, 2]),
    )
    grids = torch.arange(2 * 2 * 4 * 3, dtype=torch.float32).reshape(4, 4, 3)
    output = SimpleNamespace(
        online_states=grids + 3,
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
    assert torch.equal(payload["online_direct"], (grids + 3).reshape(2, 2, 4, 3)[:1])
    assert torch.equal(payload["online_current"], (grids + 3)[:1])


def test_visualization_selection_covers_trajectories() -> None:
    from experiments.training.sft.stage3.render_dino_feature_comparison import (
        select_page_rows,
    )

    rows = [{"trajectory": name, "window_start": start, "horizon_step": 1}
            for name in ("a", "b", "c") for start in range(10)]
    selected = select_page_rows(rows, 1, limit=3)
    assert [row["trajectory"] for row in selected] == ["a", "b", "c"]


def test_grid_renderer_accepts_other_square_grids() -> None:
    import numpy as np

    from experiments.training.sft.stage3.render_dino_feature_comparison import (
        feature_image,
    )

    rendered = feature_image(torch.zeros(4, 3), np.zeros(3), np.eye(3), np.zeros(3), np.ones(3))
    assert rendered.size == (128, 128)
