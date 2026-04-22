"""WM encoder 构建与特征提取。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.hub import load as torch_hub_load


def _ensure_torch_serialization_compat() -> None:
    """兼容旧权重中对 torch.utils.serialization 的引用。"""
    import sys
    import types

    if "torch.utils.serialization" in sys.modules:
        return
    serialization_mod = types.ModuleType("torch.utils.serialization")
    serialization_mod.__dict__.update(torch.serialization.__dict__)
    sys.modules["torch.utils.serialization"] = serialization_mod


def _to_3ch_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _preprocess_pil(image: Image.Image, image_size: int) -> torch.Tensor:
    rgb = _to_3ch_rgb(image).resize((image_size, image_size))
    arr = np.asarray(rgb).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


@dataclass
class EncoderOutput:
    """统一 encoder 输出。"""

    z: torch.Tensor
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


class DinoV2MiniEncoder(WMImageEncoder):
    """DINOv2 mini + MLP 投影。"""

    def __init__(
        self,
        latent_dim: int,
        freeze_backbone: bool = True,
        image_size: int = 224,
        model_name: str = "dinov2_vits14",
        token_strategy: str = "patch_mean",
    ) -> None:
        super().__init__(latent_dim=latent_dim)
        self.image_size = image_size
        self.model_name = model_name
        self.token_strategy = token_strategy
        self.freeze_backbone = freeze_backbone
        if not torch.cuda.is_available():
            raise RuntimeError("WM 编码推理要求 CUDA，可用 GPU 不存在或不可用。")
        self.device = torch.device("cuda")
        _ensure_torch_serialization_compat()
        self.backbone = torch_hub_load("facebookresearch/dinov2", model_name)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        embed_dim = int(getattr(self.backbone, "embed_dim", 384))
        self.proj = nn.Sequential(nn.Linear(embed_dim, latent_dim), nn.GELU())
        self.backbone.to(self.device)
        self.proj.to(self.device)

    def _select_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(pixel_values)
            patch_tokens = features.get("x_norm_patchtokens")
            cls_token = features.get("x_norm_clstoken")
            if self.token_strategy == "cls":
                if cls_token is None:
                    raise RuntimeError("DINOv2 未返回 cls token，无法使用 token_strategy=cls。")
                return cls_token
            if self.token_strategy == "patch_mean":
                if patch_tokens is None:
                    raise RuntimeError("DINOv2 未返回 patch tokens，无法使用 patch token 策略。")
                return patch_tokens.mean(dim=1)
            if self.token_strategy == "patch_attention":
                raise NotImplementedError("token_strategy=patch_attention 目前为占位，后续实现可学习注意力池化。")
            raise ValueError(f"不支持的 token_strategy: {self.token_strategy}")
        features = self.backbone(pixel_values)
        if features.dim() == 3:
            if self.token_strategy == "cls":
                return features[:, 0, :]
            return features.mean(dim=1)
        return features

    def encode_image_path(self, image_path: str) -> EncoderOutput:
        return self.encode_image_paths([image_path])[0]

    def encode_image_paths(self, image_paths: Sequence[str]) -> list[EncoderOutput]:
        if not image_paths:
            return []
        tensors: list[torch.Tensor] = []
        for image_path in image_paths:
            image = Image.open(image_path)
            tensors.append(_preprocess_pil(image=image, image_size=self.image_size))
        pixel_values = torch.cat(tensors, dim=0).to(self.device)
        if self.freeze_backbone:
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self._select_tokens(pixel_values)
                z_batch = self.proj(features).detach().cpu()
        else:
            features = self._select_tokens(pixel_values)
            z_batch = self.proj(features).detach().cpu()
        outputs: list[EncoderOutput] = []
        for image_path, z in zip(image_paths, z_batch, strict=True):
            outputs.append(
                EncoderOutput(
                    z=z,
                    aux={"encoder": "dinov2_mini", "image_path": image_path, "token_strategy": self.token_strategy},
                )
            )
        return outputs


class PlaceholderEncoder(WMImageEncoder):
    """未来 encoder 的占位实现。"""

    def __init__(self, latent_dim: int, name: str) -> None:
        super().__init__(latent_dim=latent_dim)
        self.name = name

    def encode_image_path(self, image_path: str) -> EncoderOutput:
        raise NotImplementedError(f"{self.name} 尚未实现，请后续细化该 encoder。")

    def encode_image_paths(self, image_paths: Sequence[str]) -> list[EncoderOutput]:
        raise NotImplementedError(f"{self.name} 尚未实现，请后续细化该 encoder。")


def build_wm_image_encoder(wm_cfg: Any) -> WMImageEncoder | None:
    encoder_cfg = getattr(wm_cfg, "encoder", None)
    if encoder_cfg is None:
        return None
    encoder_name = str(getattr(encoder_cfg, "name", "none")).lower()
    latent_dim = int(getattr(wm_cfg, "latent_dim", 128))
    if encoder_name in {"none", ""}:
        return None
    if encoder_name == "cfm_dinov2m":
        return DinoV2MiniEncoder(
            latent_dim=latent_dim,
            freeze_backbone=bool(getattr(encoder_cfg, "freeze_backbone", True)),
            image_size=int(getattr(encoder_cfg, "image_size", 224)),
            model_name=str(getattr(encoder_cfg, "backbone_name", "dinov2_vits14")),
            token_strategy=str(getattr(encoder_cfg, "token_strategy", "patch_mean")),
        )
    if encoder_name in {"cfm_qwen25vl_8b", "cfm_qwen25vl_8b_frozen", "cfm_dinov2m_qwen25vl_8b"}:
        return PlaceholderEncoder(latent_dim=latent_dim, name=encoder_name)
    raise ValueError(f"未知 WM encoder 配置: {encoder_name}")

