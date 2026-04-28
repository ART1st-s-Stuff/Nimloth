"""Qwen-VL 适配层。"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None

# Qwen2.5-VL-8B vision encoder output dimension
QWEN_VISION_EMBED_DIM = 1536


class QwenVLMAdapter:
    """统一管理 Qwen 模型加载、视觉向量与 CoT。"""

    def __init__(
        self,
        model_name: str,
        latent_dim: int,
        enabled: bool = True,
        fallback_enabled: bool = True,
        max_new_tokens: int = 128,
        num_patches: int | None = None,
        token_strategy: str = "patch_mean",
        encoder_embed_dim: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.latent_dim = int(latent_dim)
        self.enabled = bool(enabled)
        self.fallback_enabled = bool(fallback_enabled)
        self.max_new_tokens = int(max_new_tokens)
        self.num_patches = num_patches
        self.token_strategy = token_strategy
        self._processor: Any | None = None
        self._model: Any | None = None
        self._init_error: str | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._vision_embed_dim = encoder_embed_dim or 1536

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def _deterministic_vector(self, text: str) -> torch.Tensor:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        for index in range(self.latent_dim):
            byte_value = digest[index % len(digest)]
            values.append((float(byte_value) / 255.0) * 2.0 - 1.0)
        return torch.tensor(values, dtype=torch.float32)

    def _fallback_visual(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB").resize((64, 64))
        arr = np.asarray(image).astype("float32") / 255.0
        pooled = arr.mean(axis=(0, 1))
        token = f"{image_path}|{pooled[0]:.4f}|{pooled[1]:.4f}|{pooled[2]:.4f}|fallback_visual"
        return self._deterministic_vector(token)

    def _fallback_cot(self, task_text: str, env_context: str) -> str:
        return (
            f"任务: {task_text}\n"
            f"环境: {env_context}\n"
            "分析: 当前观察可支持继续执行稳定推进策略。\n"
            "建议: 保持朝向一致，优先避免碰撞并接近目标区域。"
        )

    def _ensure_model(self) -> None:
        if not self.enabled:
            self._init_error = "QwenVLMAdapter disabled by config."
            return
        if self._model is not None and self._processor is not None:
            return
        if self._init_error is not None:
            return
        if AutoProcessor is None or Qwen2_5_VLForConditionalGeneration is None:
            self._init_error = "transformers 未安装 Qwen2.5-VL 依赖。"
            return
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device_map = "auto" if torch.cuda.is_available() else None
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=device_map,
            )
            self._model.eval()
        except Exception as exc:  # pragma: no cover
            self._init_error = str(exc)

    def _pad_or_trim(self, vector: torch.Tensor) -> torch.Tensor:
        vector = vector.float().detach().cpu()
        if vector.numel() == self.latent_dim:
            return vector
        if vector.numel() > self.latent_dim:
            return vector[: self.latent_dim]
        return torch.cat([vector, torch.zeros(self.latent_dim - vector.numel())], dim=0)

    def _pool_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """对 patch tokens 进行池化以匹配目标 num_patches。

        输出形状: [B, target_num_patches, token_dim]
        """
        if patch_tokens.dim() != 3:
            raise ValueError(f"patch_tokens 形状不合法: {tuple(patch_tokens.shape)}")
        token_count = int(patch_tokens.size(1))
        side = int(round(math.sqrt(token_count)))
        if side * side != token_count:
            raise RuntimeError(f"Qwen patch token 数不是平方数: {token_count}")
        target_num_patches = int(self.num_patches) if self.num_patches else 16
        target_side = int(round(math.sqrt(target_num_patches)))
        if target_side * target_side != target_num_patches:
            raise RuntimeError(f"目标 patch token 数不是平方数: {target_num_patches}")
        token_dim = int(patch_tokens.size(2))
        if side == target_side:
            # 形状已匹配，保持 3D 形状 [B, num_patches, token_dim]
            return patch_tokens
        # 需要池化
        grid_tokens = patch_tokens.transpose(1, 2).reshape(patch_tokens.size(0), token_dim, side, side)
        pooled = torch.nn.functional.adaptive_avg_pool2d(grid_tokens, output_size=(target_side, target_side))
        return pooled.reshape(patch_tokens.size(0), target_num_patches, token_dim)

    def extract_visual_embedding(self, image_path: str) -> torch.Tensor:
        self._ensure_model()
        if self._model is None or self._processor is None:
            if not self.fallback_enabled:
                raise RuntimeError(f"Qwen 初始化失败且 fallback 关闭: {self._init_error}")
            return self._fallback_visual(image_path=image_path)
        image = Image.open(image_path).convert("RGB")
        # 纯视觉编码不需要 text 参数
        model_inputs = self._processor(images=image, return_tensors="pt")
        if self._device == "cuda":
            model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}
        with torch.no_grad():
            vision_features = self._model.get_image_features(
                pixel_values=model_inputs["pixel_values"],
                image_grid_thw=model_inputs.get("image_grid_thw"),
            )
        if vision_features.dim() != 3:
            vision_features = vision_features.unsqueeze(1)
        if self.token_strategy == "patch_tokens":
            # 输出 [num_patches, vision_dim]，展平后用 _pad_or_trim 匹配 latent_dim
            pooled = self._pool_patch_tokens(vision_features)
            return self._pad_or_trim(pooled.squeeze(0).reshape(-1))
        pooled = vision_features.mean(dim=1)
        return self._pad_or_trim(pooled.squeeze(0))

    def generate_cot_and_state(
        self,
        image_paths: list[str],
        task_text: str,
        env_context: str,
    ) -> tuple[str, torch.Tensor]:
        self._ensure_model()
        if self._model is None or self._processor is None:
            if not self.fallback_enabled:
                raise RuntimeError(f"Qwen 初始化失败且 fallback 关闭: {self._init_error}")
            cot_text = self._fallback_cot(task_text=task_text, env_context=env_context)
            token = cot_text.split()[-1] if cot_text.split() else "fallback"
            s_t = self._deterministic_vector(f"{token}|{task_text}|{env_context}")
            return cot_text, s_t
        valid_paths = [path for path in image_paths if Path(path).exists()]
        if not valid_paths:
            cot_text = self._fallback_cot(task_text=task_text, env_context=env_context)
            return cot_text, self._deterministic_vector(cot_text)
        cot_prompt = (
            f"任务: {task_text}\n"
            f"环境上下文: {env_context}\n"
            f"你会看到按时间顺序给出的历史图像（共 {len(valid_paths)} 帧）。"
            "请输出简洁推理和下一步建议。"
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": path} for path in valid_paths] + [{"type": "text", "text": cot_prompt}],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text],
            images=[Image.open(path).convert("RGB") for path in valid_paths],
            return_tensors="pt",
        )
        if self._device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        prompt_len = int(inputs["input_ids"].shape[1])
        generated_ids = output_ids[:, prompt_len:]
        cot_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0] if generated_ids.numel() > 0 else ""
        with torch.no_grad():
            forward_out = self._model(input_ids=output_ids, output_hidden_states=True)
        s_t = forward_out.hidden_states[-1][:, -1, :].squeeze(0)
        return cot_text, self._pad_or_trim(s_t)

    def get_image_hidden_state(
        self,
        image_path: str,
        prompt: str | None = None,
        layer: int = -1,
        return_last_token_only: bool = True,
    ) -> torch.Tensor:
        """获取图像在 Qwen LLM space 中的 hidden state 表征。

        不生成文本，直接 forward 获取 hidden states。
        这是为了让 WM 直接使用 LLM 的 embedding space 作为 latent。

        Args:
            image_path: 图像路径
            prompt: 可选的文本 prompt（用于引导 LLM 理解图像）
            layer: 返回哪层 hidden state（-1 = 最后一层）
            return_last_token_only: True = 返回 last token 的 hidden state

        Returns:
            hidden_state: [D] (last token)
        """
        self._ensure_model()
        if self._model is None or self._processor is None:
            if not self.fallback_enabled:
                raise RuntimeError(f"Qwen 初始化失败且 fallback 关闭: {self._init_error}")
            return self._fallback_visual(image_path=image_path)

        image = Image.open(image_path).convert("RGB")

        # 构建输入
        if prompt:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}],
                }
            ]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            )
        else:
            inputs = self._processor(images=image, return_tensors="pt")

        if self._device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        # 方案1：直接用 vision embedding 作为 latent
        # 这是最直接的方式，返回 vision encoder 的输出
        if pixel_values is not None:
            vision_emb = self._model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            # vision_emb: [1, num_patches, vision_dim]
            if vision_emb.dim() == 3:
                vision_emb = vision_emb.squeeze(0)
            # 返回 mean pooling 或最后一层
            if return_last_token_only:
                # 返回最后一个 vision token 的 embedding
                return self._pad_or_trim(vision_emb[-1, :])
            else:
                return self._pad_or_trim(vision_emb.mean(dim=0))

        return self._fallback_visual(image_path=image_path)
