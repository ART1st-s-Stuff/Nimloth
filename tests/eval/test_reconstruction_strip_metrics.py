from __future__ import annotations

import pytest
import torch

from nimloth.eval.reconstruction_strip_metrics import structural_similarity


def test_structural_similarity_is_one_for_identical_images() -> None:
    generator = torch.Generator().manual_seed(7)
    images = torch.rand(3, 3, 32, 32, generator=generator)

    score = structural_similarity(images, images)

    assert score.tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)


def test_structural_similarity_rejects_misaligned_images() -> None:
    with pytest.raises(ValueError, match=r"matching \[N,C,H,W\]"):
        structural_similarity(
            torch.zeros(2, 3, 16, 16),
            torch.zeros(1, 3, 16, 16),
        )
