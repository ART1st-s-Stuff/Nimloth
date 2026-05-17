"""Qwen 图像编码器。"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from src.wm.encoder.base import EncoderOutput, WMImageEncoder
from src.vlm.qwen_adapter import QwenVLMAdapter


class QwenImageEncoder(WMImageEncoder):
    """Qwen encoder 实现，封装 QwenVLMAdapter。"""

    DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(
        self,
        latent_dim: int,
        name: str,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        enabled: bool = True,
        fallback_enabled: bool = True,
        num_patches: int | None = None,
        token_strategy: str = "patch_mean",
        encoder_embed_dim: int | None = None,
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
            encoder_embed_dim=encoder_embed_dim,
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


class TrainableQwenLatentAdapter(nn.Module):
    """Qwen latent 轻量微调模块。

    该模块不改动 Qwen backbone，仅在 latent 空间做小参数适配，
    以便注入物理信息并通过蒸馏约束保持原始语义。
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 1024,
        mode: str = "adapter_only",
        trainable_blocks: int = 0,
        distill_teacher: str = "frozen_qwen",
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = max(1, int(hidden_dim))
        self.mode = str(mode).strip().lower()
        self.trainable_blocks = max(0, int(trainable_blocks))
        self.distill_teacher = str(distill_teacher).strip().lower()
        # 小型残差适配器：z_student = z + f(z)
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        # mode 预留：当前仅支持 adapter_only / lora_topk（同样走 adapter）
        if self.mode not in {"adapter_only", "lora_topk"}:
            raise ValueError(f"不支持的 encoder_finetune.mode={mode}")

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """输入 shape: [..., latent_dim] 或 [..., P, D]，输出同形状。"""
        original_shape = latents.shape
        if latents.size(-1) == self.latent_dim:
            flat = latents
            use_reshape_back = False
        else:
            if latents.dim() >= 2:
                last_numel = int(latents[0, 0].numel()) if latents.dim() >= 3 else int(latents[0].numel())
            else:
                last_numel = int(latents.numel())
            if last_numel != self.latent_dim:
                raise ValueError(
                    f"latent 维度不匹配: got={tuple(original_shape)}, expected_flat={self.latent_dim}"
                )
            if latents.dim() >= 2:
                flat = latents.reshape(*original_shape[:-2], self.latent_dim)
            else:
                flat = latents.reshape(-1, self.latent_dim)
            use_reshape_back = True
        delta = self.adapter(flat)
        adapted = flat + delta
        if use_reshape_back:
            return adapted.reshape(original_shape)
        return adapted

    def teacher_forward(self, latents: torch.Tensor) -> torch.Tensor:
        """teacher 分支（冻结 Qwen）的等价输出。"""
        if self.distill_teacher != "frozen_qwen":
            raise ValueError(f"不支持的 distill.teacher={self.distill_teacher}")
        return latents.detach()

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """暴露参数分组，便于后续扩展不同学习率策略。"""
        return {"adapter": list(self.adapter.parameters())}


class QwenLLMLatentEncoder(WMImageEncoder):
    """使用 Qwen LLM hidden state 作为 latent 的 encoder。

    架构：
        Image → Vision Encoder → vision tokens → LLM backbone (FROZEN) → hidden state

    用于 Phase 2 WM 训练，让 WM 直接在 Qwen LLM 的 embedding space 中学习 dynamics。

    特点：
    - Vision Encoder 可训练
    - LLM backbone 固定（FROZEN），保持语义理解能力
    - Latent 格式：[B, D]，其中 D = Qwen hidden dim (通常 4096)
    - WM 需要适配 [B, 1, 1, D] 的输入格式

    参数：
    - llm_backbone_trainable: 预留接口，True 时 LLM backbone 也参与训练
    """

    DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    _DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"  # 用于默认值

    def __init__(
        self,
        latent_dim: int,
        name: str = "qwen_llm",
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        enabled: bool = True,
        fallback_enabled: bool = True,
        prompt_template: str | None = None,
        qwen_adapter: QwenVLMAdapter | None = None,
        use_fallback: bool = False,
        use_vision_only: bool = False,
        visual_pooling: str = "last",
        visual_num_tokens: int | None = None,
        cache_latents: bool = False,
        llm_backbone_trainable: bool = False,
        latent_anchor_mode: str = "last_token",
    ) -> None:
        super().__init__(latent_dim=latent_dim)
        self.name = name
        self.prompt_template = prompt_template
        self.use_fallback = use_fallback
        self.use_vision_only = use_vision_only
        self.visual_pooling = str(visual_pooling).strip().lower()
        if self.visual_pooling not in {"last", "mean", "tokens"}:
            raise ValueError(f"不支持的 visual_pooling={visual_pooling}")
        self.visual_num_tokens = int(visual_num_tokens) if visual_num_tokens is not None else None
        self.cache_latents = bool(cache_latents)
        self._latent_cache: dict[tuple[str, str | None, str | None, bool, str, int | None, str], torch.Tensor] = {}
        self.llm_backbone_trainable = llm_backbone_trainable
        self.latent_anchor_mode = str(latent_anchor_mode).strip().lower()
        if self.latent_anchor_mode not in {"last_token", "planner_special"}:
            raise ValueError(f"不支持的 latent_anchor_mode={latent_anchor_mode}")
        # 复用已有的 adapter 或创建新的
        if qwen_adapter is not None:
            self._adapter = qwen_adapter
        elif use_fallback:
            # 直接使用 fallback，不尝试加载模型
            self._adapter = QwenVLMAdapter(
                model_name=model_name,
                latent_dim=latent_dim,
                enabled=False,  # 禁用模型加载
                fallback_enabled=True,
            )
        else:
            self._adapter = QwenVLMAdapter(
                model_name=model_name,
                latent_dim=latent_dim,
                enabled=enabled,
                fallback_enabled=fallback_enabled,
            )

    def encode_image_path_with_prompt(
        self,
        image_path: str,
        prompt_override: str | None = None,
        response_override: str | None = None,
    ) -> EncoderOutput:
        """返回 [D] 维 latent（last token hidden state）"""
        prompt = prompt_override if prompt_override is not None else self.prompt_template
        cache_key = (
            str(image_path),
            prompt,
            response_override,
            bool(self.use_vision_only),
            self.visual_pooling,
            self.visual_num_tokens,
            self.latent_anchor_mode,
        )
        if self.cache_latents and cache_key in self._latent_cache:
            z = self._latent_cache[cache_key].clone()
            return EncoderOutput(
                z=z,
                aux={
                    "encoder": self.name,
                    "image_path": image_path,
                    "prompt": prompt,
                    "response_override": response_override,
                    "llm_hidden_state": not self.use_vision_only,
                    "use_vision_only": self.use_vision_only,
                    "visual_pooling": self.visual_pooling,
                    "visual_num_tokens": self.visual_num_tokens,
                    "llm_backbone_trainable": self.llm_backbone_trainable,
                    "latent_anchor_mode": self.latent_anchor_mode,
                    "cache_hit": True,
                },
            )
        if self.latent_anchor_mode == "planner_special":
            planner = self._adapter.get_planner_latent_and_action_prior(
                image_path=image_path,
                prompt=prompt or "",
                response=response_override,
                llm_backbone_trainable=self.llm_backbone_trainable,
            )
            z = planner["latent"]
        else:
            z = self._adapter.get_image_hidden_state(
                image_path=image_path,
                prompt=prompt,
                return_last_token_only=self.visual_pooling != "mean",
                use_vision_only=self.use_vision_only,
                return_visual_tokens=self.visual_pooling == "tokens",
                visual_num_tokens=self.visual_num_tokens,
                llm_backbone_trainable=self.llm_backbone_trainable,
            )
        if self.cache_latents:
            self._latent_cache[cache_key] = z.detach().cpu()
        return EncoderOutput(
            z=z,
            aux={
                "encoder": self.name,
                "image_path": image_path,
                "prompt": prompt,
                "response_override": response_override,
                "llm_hidden_state": not self.use_vision_only,
                "use_vision_only": self.use_vision_only,
                "visual_pooling": self.visual_pooling,
                "visual_num_tokens": self.visual_num_tokens,
                "llm_backbone_trainable": self.llm_backbone_trainable,
                "latent_anchor_mode": self.latent_anchor_mode,
            },
        )

    def encode_image_path(self, image_path: str) -> EncoderOutput:
        return self.encode_image_path_with_prompt(image_path=image_path)

    def encode_image_paths(self, image_paths: Sequence[str]) -> list[EncoderOutput]:
        return [self.encode_image_path(path) for path in image_paths]

    def get_latent_batch(self, image_paths: list[str]) -> torch.Tensor:
        """批量获取 latents，返回 [B, D]"""
        latents = []
        for path in image_paths:
            output = self.encode_image_path(path)
            latents.append(output.z)
        return torch.stack(latents)


class QwenVisualTokenEncoder(QwenLLMLatentEncoder):
    """Explicit Qwen visual-token encoder for WM visual latents.

    This is a naming-compatible wrapper around QwenLLMLatentEncoder configured for
    the existing frozen Qwen visual-token path. It has no parameters of its own, so
    existing checkpoints and state_dict keys remain valid.
    """

    def __init__(
        self,
        latent_dim: int = 57344,
        *,
        name: str = "qwen_visual_tokens",
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        qwen_adapter=None,
        cache_latents: bool = True,
        visual_num_tokens: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            name=name,
            model_name=model_name,
            qwen_adapter=qwen_adapter,
            use_vision_only=True,
            visual_pooling="tokens",
            visual_num_tokens=visual_num_tokens,
            cache_latents=cache_latents,
            **kwargs,
        )

