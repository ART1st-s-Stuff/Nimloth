from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nimloth.backbone.qwen25vl.factory import (
    _load_kwargs,
    _validate_model_parallel_placement,
    model_output_device,
)


def test_pair_load_uses_balanced_placement(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    enabled, kwargs = _load_kwargs(
        torch.device("cuda:2"),
        model_parallel_size=2,
    )

    assert enabled is True
    assert kwargs["device_map"] == "balanced"
    assert kwargs["max_memory"] == {2: "74GiB", 3: "74GiB", "cpu": "64GiB"}


def test_pair_placement_requires_both_local_gpus() -> None:
    model = SimpleNamespace(
        hf_device_map={
            "model.visual": 2,
            "model.language_model.layers.0": 2,
            "model.language_model.norm": 2,
            "lm_head": 2,
        }
    )

    with pytest.raises(RuntimeError, match=r"expected=\[2, 3\], actual=\[2\]"):
        _validate_model_parallel_placement(
            model,
            input_device=torch.device("cuda:2"),
            model_parallel_size=2,
        )


def test_model_output_device_follows_lm_head() -> None:
    model = SimpleNamespace(
        hf_device_map={
            "model.language_model.norm": 2,
            "lm_head": 3,
        }
    )

    assert model_output_device(
        model,
        default=torch.device("cuda:2"),
    ) == torch.device("cuda:3")
