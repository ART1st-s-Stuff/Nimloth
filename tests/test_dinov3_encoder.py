from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

from nimloth.backbone.dinov3 import FrozenDINOv3Encoder


class FakeImageProcessor:
    def __call__(self, *, images, return_tensors: str):
        assert return_tensors == "pt"
        rows = []
        for image in images:
            value = float(image.getpixel((0, 0))[0]) / 255.0
            rows.append(torch.full((3, 2, 2), value))
        return {"pixel_values": torch.stack(rows)}


class FakeDINOv3(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(hidden_size=4)
        self.training_flags: list[bool] = []

    def train(self, mode: bool = True):
        self.training_flags.append(mode)
        return super().train(mode)

    def forward(self, *, pixel_values: torch.Tensor):
        means = pixel_values.float().mean(dim=(1, 2, 3))
        cls = torch.stack([means, means + 1, means + 2, means + 3], dim=-1)
        registers_and_patches = torch.zeros(pixel_values.shape[0], 3, 4, device=pixel_values.device)
        return SimpleNamespace(last_hidden_state=torch.cat([cls.unsqueeze(1), registers_and_patches], dim=1))


def test_frozen_dinov3_encoder_returns_detached_cls_features(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(first)
    Image.new("RGB", (2, 2), (255, 255, 255)).save(second)
    model = FakeDINOv3()
    encoder = FrozenDINOv3Encoder(model=model, image_processor=FakeImageProcessor(), source="fake")

    encoder.train()
    features = encoder.encode_image_paths([str(first), str(second)], device=torch.device("cpu"))

    assert encoder.hidden_size == 4
    assert encoder.source == "fake"
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert features.requires_grad is False
    torch.testing.assert_close(
        features,
        torch.tensor([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]),
    )
