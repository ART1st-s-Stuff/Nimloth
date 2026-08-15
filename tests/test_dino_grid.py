from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

from nimloth.backbone import DINOIdentity, FrozenDINOGridTargets


class _ImageProcessor:
    def __call__(self, *, images, return_tensors: str):  # type: ignore[no-untyped-def]
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros(len(images), 3, 4, 4)}


class _DINO(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(hidden_size=2, patch_size=1)
        self.calls = 0

    def forward(self, *, pixel_values: torch.Tensor):  # type: ignore[no-untyped-def]
        self.calls += 1
        batch, _channels, height, width = pixel_values.shape
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(height),
                torch.arange(width),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(1, height * width, 2).expand(batch, -1, -1).float()
        cls = torch.full((batch, 1, 2), -1.0)
        return SimpleNamespace(
            last_hidden_state=torch.cat((cls, coordinates), dim=1)
        )


def test_frozen_dino_grid_targets_pool_row_major_and_cache_images(tmp_path) -> None:
    image_path = tmp_path / "observation.png"
    Image.new("RGB", (4, 4)).save(image_path)
    model = _DINO()
    targets = FrozenDINOGridTargets(
        model=model,
        image_processor=_ImageProcessor(),
        identity=DINOIdentity(
            source="fake",
            revision="fake",
            processor_fingerprint="fake",
            hidden_size=2,
        ),
        grid_size=2,
    )

    first = targets.load((image_path,), device=torch.device("cpu"))
    second = targets.load((image_path,), device=torch.device("cpu"))

    expected = torch.tensor(
        [[[0.5, 0.5], [0.5, 2.5], [2.5, 0.5], [2.5, 2.5]]]
    )
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    assert model.calls == 1
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_frozen_dino_grid_targets_encode_in_memory_rollout_images() -> None:
    model = _DINO()
    targets = FrozenDINOGridTargets(
        model=model,
        image_processor=_ImageProcessor(),
        identity=DINOIdentity(
            source="fake",
            revision="fake",
            processor_fingerprint="fake",
            hidden_size=2,
        ),
        grid_size=2,
    )
    images = [Image.new("RGB", (4, 4)), Image.new("L", (4, 4))]
    encoded = targets.load_images(images, device=torch.device("cpu"))
    assert encoded.shape == (2, 4, 2)
    assert encoded.dtype == torch.float32
    assert model.calls == 1
    with torch.no_grad():
        expected = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 2.5], [2.5, 0.5], [2.5, 2.5]],
                [[0.5, 0.5], [0.5, 2.5], [2.5, 0.5], [2.5, 2.5]],
            ]
        )
    torch.testing.assert_close(encoded, expected)
