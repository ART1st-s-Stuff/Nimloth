"""DINOv2 图像编码器。

提供冻结和可微调两种模式的 DINOv2 encoder。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.hub import load as torch_hub_load

from src.wm.encoder.base import EncoderOutput, WMImageEncoder
from src.wm.sigreg_modules import SIGReg


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


class DinoV2MiniEncoder(WMImageEncoder):
    """DINOv2 mini + MLP 投影（冻结模式）。

    用于预计算 latents 或推理阶段的固定 encoder。
    不区分来源（CFM/LeWM 训练后的权重通用）。
    """

    def __init__(
        self,
        latent_dim: int,
        freeze_backbone: bool = True,
        image_size: int = 224,
        patch_size: int = 14,
        num_patches: int | None = None,
        model_name: str = "dinov2_vits14",
        token_strategy: str = "patch_mean",
    ) -> None:
        super().__init__(latent_dim=latent_dim)
        self.image_size = image_size
        self.patch_size = max(1, int(patch_size))
        if num_patches is not None and int(num_patches) > 0:
            self.target_num_patches = int(num_patches)
        else:
            grid_size = max(1, int(self.image_size // self.patch_size))
            self.target_num_patches = grid_size * grid_size
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

    def _pool_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.dim() != 3:
            raise ValueError(f"patch_tokens 形状不合法: {tuple(patch_tokens.shape)}")
        token_count = int(patch_tokens.size(1))
        side = int(round(math.sqrt(token_count)))
        if side * side != token_count:
            raise RuntimeError(f"DINO patch token 数不是平方数: {token_count}")
        target_side = int(round(math.sqrt(self.target_num_patches)))
        if target_side * target_side != self.target_num_patches:
            raise RuntimeError(f"目标 patch token 数不是平方数: {self.target_num_patches}")
        if side == target_side:
            return patch_tokens
        token_dim = int(patch_tokens.size(2))
        grid_tokens = patch_tokens.transpose(1, 2).reshape(patch_tokens.size(0), token_dim, side, side)
        pooled = torch.nn.functional.adaptive_avg_pool2d(grid_tokens, output_size=(target_side, target_side))
        return pooled.reshape(patch_tokens.size(0), token_dim, target_side * target_side).transpose(1, 2)

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
            if self.token_strategy == "patch_tokens":
                if patch_tokens is None:
                    raise RuntimeError("DINOv2 未返回 patch tokens，无法使用 token_strategy=patch_tokens。")
                return self._pool_patch_tokens(patch_tokens)
            if self.token_strategy == "patch_attention":
                raise NotImplementedError("token_strategy=patch_attention 目前为占位，后续实现可学习注意力池化。")
            raise ValueError(f"不支持的 token_strategy: {self.token_strategy}")
        features = self.backbone(pixel_values)
        if features.dim() == 3:
            if self.token_strategy == "patch_tokens":
                return self._pool_patch_tokens(features)
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
                if self.token_strategy == "patch_tokens":
                    z_batch = features.detach().cpu()
                else:
                    z_batch = self.proj(features).detach().cpu()
        else:
            features = self._select_tokens(pixel_values)
            if self.token_strategy == "patch_tokens":
                z_batch = features.detach().cpu()
            else:
                z_batch = self.proj(features).detach().cpu()
        outputs: list[EncoderOutput] = []
        for image_path, z in zip(image_paths, z_batch, strict=True):
            if self.token_strategy == "patch_tokens" and int(z.numel()) != int(self.latent_dim):
                raise RuntimeError(
                    f"patch token 展平维度与 latent_dim 不一致: got={int(z.numel())}, expected={int(self.latent_dim)}"
                )
            outputs.append(
                EncoderOutput(
                    z=z,
                    aux={
                        "encoder": "dinov2_mini",
                        "image_path": image_path,
                        "token_strategy": self.token_strategy,
                        "image_size": self.image_size,
                        "patch_size": self.patch_size,
                        "num_patches": self.target_num_patches,
                    },
                )
            )
        return outputs


class TrainableDinoV2Encoder(WMImageEncoder):
    """可微调的 DINOv2 encoder，用于方案2（图像 encoder 微调 + SIGReg）。

    用于 Phase2 训练，需要区分来源以加载正确权重。
    支持 cfm_trainable_dinov2m 和 lewm_trainable_dinov2m。

    与 DinoV2MiniEncoder 的区别：
    - backbone 默认可训练（freeze_backbone=False）
    - 支持 SIGReg 正则化作为输出约束
    - 训练时直接返回梯度，而非 detach
    """

    def __init__(
        self,
        latent_dim: int,
        freeze_backbone: bool = False,
        sigreg_enabled: bool = False,
        sigreg_latent_dim: int | None = None,
        sigreg_num_quadrature_points: int = 16,
        sigreg_num_proj: int = 256,
        sigreg_t_min: float = 0.2,
        sigreg_t_max: float = 4.0,
        sigreg_kernel_sigma: float = 1.0,
        image_size: int = 224,
        patch_size: int = 14,
        num_patches: int | None = None,
        model_name: str = "dinov2_vits14",
        token_strategy: str = "patch_tokens",
        source_prefix: str = "",
    ) -> None:
        super().__init__(latent_dim=latent_dim)
        self.image_size = image_size
        self.patch_size = max(1, int(patch_size))
        if num_patches is not None and int(num_patches) > 0:
            self.target_num_patches = int(num_patches)
        else:
            grid_size = max(1, int(self.image_size // self.patch_size))
            self.target_num_patches = grid_size * grid_size
        self.model_name = model_name
        self.token_strategy = token_strategy
        self.freeze_backbone = freeze_backbone
        self.sigreg_enabled = sigreg_enabled
        self.source_prefix = source_prefix

        if not torch.cuda.is_available():
            raise RuntimeError("WM 编码推理要求 CUDA，可用 GPU 不存在或不可用。")
        self.device = torch.device("cuda")
        _ensure_torch_serialization_compat()
        self.backbone = torch_hub_load("facebookresearch/dinov2", model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        else:
            self.backbone.train()

        embed_dim = int(getattr(self.backbone, "embed_dim", 384))

        # 投影层：backbone 输出维度 -> 目标 latent 维度
        self.proj = nn.Sequential(nn.Linear(embed_dim, latent_dim), nn.GELU())

        # 方案2：SIGReg 模块
        self.sigreg: SIGReg | None = None
        self.sigreg_proj: nn.Module | None = None
        if sigreg_enabled:
            sigreg_dim = sigreg_latent_dim or latent_dim
            if embed_dim != sigreg_dim:
                self.sigreg_proj = nn.Sequential(
                    nn.Linear(embed_dim, max(embed_dim, sigreg_dim)),
                    nn.LayerNorm(max(embed_dim, sigreg_dim)),
                    nn.GELU(),
                    nn.Linear(max(embed_dim, sigreg_dim), sigreg_dim),
                )
            else:
                self.sigreg_proj = nn.Identity()
            self.sigreg = SIGReg(
                num_quadrature_points=sigreg_num_quadrature_points,
                num_proj=sigreg_num_proj,
                t_min=sigreg_t_min,
                t_max=sigreg_t_max,
                kernel_sigma=sigreg_kernel_sigma,
            )

        self.backbone.to(self.device)
        self.proj.to(self.device)

    def _pool_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.dim() != 3:
            raise ValueError(f"patch_tokens 形状不合法: {tuple(patch_tokens.shape)}")
        token_count = int(patch_tokens.size(1))
        side = int(round(math.sqrt(token_count)))
        if side * side != token_count:
            raise RuntimeError(f"DINO patch token 数不是平方数: {token_count}")
        target_side = int(round(math.sqrt(self.target_num_patches)))
        if target_side * target_side != self.target_num_patches:
            raise RuntimeError(f"目标 patch token 数不是平方数: {self.target_num_patches}")
        if side == target_side:
            return patch_tokens
        token_dim = int(patch_tokens.size(2))
        grid_tokens = patch_tokens.transpose(1, 2).reshape(patch_tokens.size(0), token_dim, side, side)
        pooled = torch.nn.functional.adaptive_avg_pool2d(grid_tokens, output_size=(target_side, target_side))
        return pooled.reshape(patch_tokens.size(0), token_dim, target_side * target_side).transpose(1, 2)

    def _select_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(pixel_values)
            patch_tokens = features.get("x_norm_patchtokens")
            if self.token_strategy == "patch_tokens":
                if patch_tokens is None:
                    raise RuntimeError("DINOv2 未返回 patch tokens，无法使用 token_strategy=patch_tokens。")
                return self._pool_patch_tokens(patch_tokens)
            raise ValueError(f"TrainableDinoV2Encoder 仅支持 token_strategy=patch_tokens，当前为 {self.token_strategy}")
        raise RuntimeError("TrainableDinoV2Encoder 需要 DINOv2 backbone 支持 forward_features")

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

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = self._select_tokens(pixel_values)
                z_batch = self.proj(features).detach().cpu()

        outputs: list[EncoderOutput] = []
        for image_path, z in zip(image_paths, z_batch, strict=True):
            outputs.append(
                EncoderOutput(
                    z=z,
                    aux={
                        "encoder": f"{self.source_prefix}trainable_dinov2" if self.source_prefix else "trainable_dinov2",
                        "image_path": image_path,
                        "token_strategy": self.token_strategy,
                        "image_size": self.image_size,
                        "patch_size": self.patch_size,
                        "num_patches": self.target_num_patches,
                        "sigreg_enabled": self.sigreg_enabled,
                    },
                )
            )
        return outputs

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """前向传播，返回投影后的 latent。

        Args:
            pixel_values: [B, C, H, W] 图像张量

        Returns:
            [B, P, D] patch tokens 投影后的 latent
        """
        features = self._select_tokens(pixel_values)
        return self.proj(features)

    def compute_sigreg(self, latents: torch.Tensor) -> torch.Tensor | None:
        """计算 SIGReg 正则损失。

        Args:
            latents: [B, T, D] 或 [B, D] encoder 输出的 latent 序列

        Returns:
            SIGReg 损失值，如果未启用则返回 None
        """
        if not self.sigreg_enabled or self.sigreg is None or self.sigreg_proj is None:
            return None

        if latents.dim() == 2:
            latents = latents.unsqueeze(1)  # [B, 1, D]

        # 投影到 SIGReg space
        z = self.sigreg_proj(latents)  # [B, T, sigreg_dim]

        # 转换为 [T, B, D] 格式
        T, B, D = z.shape[1], z.shape[0], z.shape[2]
        z_perm = z.permute(1, 0, 2)  # [T, B, D]

        return self.sigreg(z_perm)