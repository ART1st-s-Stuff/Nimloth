"""Encoder 基类定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from torch import nn


@dataclass
class EncoderOutput:
    """统一 encoder 输出。"""

    z: Any
    aux: dict[str, Any]


class WMImageEncoder(nn.Module):
    """WM 图像编码器抽象。"""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def encode_image_path(self, image_path: str) -> EncoderOutput:
        raise NotImplementedError

    def encode_image_paths(self, image_paths: Sequence[str]) -> list[EncoderOutput]:
        raise NotImplementedError