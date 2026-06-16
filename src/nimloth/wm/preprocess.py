"""Image preprocessing for LeWM-style encoders (separate from Qwen VL processor)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class LeWMImageTransform:
    """Resize + tensor normalize for navigation WM training."""

    img_size: int = 96

    def __call__(self, image: Image.Image | str) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        return self.pipeline(image)

    @property
    def pipeline(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )


def default_image_transform(img_size: int = 96) -> LeWMImageTransform:
    return LeWMImageTransform(img_size=img_size)
