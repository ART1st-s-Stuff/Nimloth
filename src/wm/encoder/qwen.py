"""Qwen 图像编码器。"""

from __future__ import annotations

from typing import Any, Sequence

from src.wm.encoder.base import EncoderOutput, WMImageEncoder
from src.vlm.qwen_adapter import QwenVLMAdapter


class QwenImageEncoder(WMImageEncoder):
    """Qwen encoder 实现，封装 QwenVLMAdapter。"""

    def __init__(
        self,
        latent_dim: int,
        name: str,
        model_name: str = "Qwen/Qwen2.5-VL-8B-Instruct",
        enabled: bool = True,
        fallback_enabled: bool = True,
        num_patches: int | None = None,
        token_strategy: str = "patch_mean",
    ) -> None:
        super().__init__(latent_dim=latent_dim)
        self.name = name
        self.num_patches = num_patches
        self.token_strategy = token_strategy
        self._adapter = QwenVLMAdapter(
            model_name=model_name,
            latent_dim=latent_dim,
            enabled=enabled,
            fallback_enabled=fallback_enabled,
            num_patches=num_patches,
            token_strategy=token_strategy,
        )

    def encode_image_path(self, image_path: str) -> EncoderOutput:
        z = self._adapter.extract_visual_embedding(image_path)
        return EncoderOutput(
            z=z,
            aux={
                "encoder": self.name,
                "image_path": image_path,
                "token_strategy": self.token_strategy,
                "num_patches": self.num_patches,
            },
        )

    def encode_image_paths(self, image_paths: Sequence[str]) -> list[EncoderOutput]:
        return [self.encode_image_path(path) for path in image_paths]