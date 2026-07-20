"""Image helpers for RCDM guided-diffusion tensors."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def image_to_diffusion_tensor(
    path: str | Path,
    *,
    image_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Load an RGB image as a ``[-1, 1]`` tensor with shape ``(3, H, W)``."""

    resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    img = Image.open(path).convert("RGB").resize((image_size, image_size), resample)
    data = torch.tensor(list(img.getdata()), dtype=torch.float32)
    data = data.view(image_size, image_size, 3).permute(2, 0, 1).div(127.5).sub(1.0)
    return data.to(device) if device is not None else data


def diffusion_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert one ``[-1, 1]`` diffusion image tensor to a PIL RGB image."""

    if tensor.ndim != 3:
        raise ValueError(f"expected image tensor with shape (3, H, W), got {tuple(tensor.shape)}")
    arr = tensor.detach().float().add(1.0).mul(127.5).clamp(0, 255).byte().cpu()
    arr = arr.permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


def save_diffusion_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    """Save one ``[-1, 1]`` diffusion image tensor to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    diffusion_tensor_to_pil(tensor).save(path)


def make_horizontal_strip(images: list[Image.Image]) -> Image.Image:
    """Concatenate equally sized images left-to-right."""

    if not images:
        raise ValueError("images must be non-empty")
    widths, heights = zip(*(img.size for img in images), strict=True)
    out = Image.new("RGB", (sum(widths), max(heights)))
    x = 0
    for img in images:
        out.paste(img.convert("RGB"), (x, 0))
        x += img.width
    return out
