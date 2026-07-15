from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nimloth.training.reconstruction.query_bottleneck_probe import (
    QueryBottleneckAdapter,
    initialize_from_baseline,
    load_frozen_baseline,
)
from nimloth.training.reconstruction.state_to_vision_tokens import (
    StateToVisionTokens,
    VisionTokenAdapterConfig,
)


def test_query_bottleneck_emits_exact_token_shape_and_visual_tokens() -> None:
    model = QueryBottleneckAdapter(
        input_tokens=8,
        input_dim=32,
        bottleneck_dim=16,
    )
    query = torch.randn(3, 8, 32)

    encoded = model.encode(query)
    output = model(query)

    assert encoded.shape == (3, 8, 16)
    assert output.shape == (3, 16, 512)
    assert torch.isfinite(encoded).all()
    assert torch.isfinite(output).all()


def test_query_bottleneck_rejects_wrong_cached_shape() -> None:
    model = QueryBottleneckAdapter(
        input_tokens=8,
        input_dim=32,
        bottleneck_dim=16,
    )

    with pytest.raises(ValueError, match="expected query hidden"):
        model(torch.randn(2, 1, 32))


def test_initialize_and_load_frozen_baseline(tmp_path: Path) -> None:
    config = VisionTokenAdapterConfig(input_tokens=8, input_dim=32)
    baseline = StateToVisionTokens(config)
    checkpoint = tmp_path / "baseline.pt"
    torch.save({"query_adapter": baseline.state_dict(), "step": 7500}, checkpoint)

    loaded, payload = load_frozen_baseline(
        checkpoint,
        input_tokens=8,
        input_dim=32,
        device=torch.device("cpu"),
    )
    bottleneck = QueryBottleneckAdapter(
        input_tokens=8,
        input_dim=32,
        bottleneck_dim=16,
    )
    copied = initialize_from_baseline(bottleneck, loaded)

    assert payload["step"] == 7500
    assert copied > 0
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert torch.equal(
        bottleneck.adapter.output_queries,
        loaded.output_queries,
    )
