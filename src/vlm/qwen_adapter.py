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

try:
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None

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
        # 文件不存在时，使用基于路径的确定性向量
        token = f"{image_path}|fallback_visual"
        return self._deterministic_vector(token)

    def _fallback_cot(self, task_text: str, env_context: str) -> str:
        return (
            f"任务: {task_text}\n"
            f"环境: {env_context}\n"
            "分析: 当前观察可支持继续执行稳定推进策略。\n"
            "建议: 保持朝向一致，优先避免碰撞并接近目标区域。"
        )

    def _set_llm_backbone_trainable(self, trainable: bool = False) -> None:
        """设置 LLM backbone 的可训练状态。

        用于联合训练：冻结 LLM backbone，只训练 Vision Encoder。

        Args:
            trainable: True = LLM backbone 可训练，False = LLM backbone 冻结
        """
        if self._model is None:
            return
        # 冻结/解冻 LLM backbone
        # Vision Encoder 的参数名以 'visual.' 开头
        # LLM backbone 的参数名以 'model.' 开头（model.layers, model.embed_tokens 等）
        for name, param in self._model.named_parameters():
            # Vision Encoder (visual.*) 始终可训练
            if self._is_visual_param(name):
                param.requires_grad = True
            else:
                # LLM backbone 冻结或解冻
                param.requires_grad = trainable

    @staticmethod
    def _is_visual_param(name: str) -> bool:
        return name.startswith("visual.") or ".visual." in name

    def enable_visual_lora(
        self,
        *,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: list[str] | None = None,
    ) -> int:
        """仅在 visual encoder 上挂 LoRA。返回可训练参数数量。"""
        self._ensure_model()
        if self._model is None:
            raise RuntimeError(f"Qwen 模型未加载: {self._init_error}")
        if LoraConfig is None or get_peft_model is None:
            raise RuntimeError("未安装 peft，无法启用 LoRA（请安装 peft 依赖）。")

        # 先冻结全部参数，后续只打开 visual LoRA 参数。
        for _, param in self._model.named_parameters():
            param.requires_grad = False

        module_names = [name for name, _ in self._model.named_modules() if self._is_visual_param(name)]
        visual_targets: list[str]
        if target_modules:
            visual_targets = [t for t in target_modules if any(t in name for name in module_names)]
            if not visual_targets:
                raise RuntimeError(
                    f"LoRA target_modules 未匹配到 visual 模块: requested={target_modules}"
                )
        else:
            # 常见线性层命名回退，避免 silent no-op
            candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj", "fc1", "fc2"]
            visual_targets = [c for c in candidates if any(c in name for name in module_names)]
            if not visual_targets:
                raise RuntimeError("自动探测 visual LoRA target_modules 失败，请显式配置。")

        lora_cfg = LoraConfig(
            r=max(1, int(r)),
            lora_alpha=max(1, int(alpha)),
            lora_dropout=float(max(0.0, dropout)),
            target_modules=visual_targets,
            bias="none",
        )
        self._model = get_peft_model(self._model, lora_cfg)

        trainable = 0
        for name, param in self._model.named_parameters():
            if "lora_" in name and self._is_visual_param(name):
                param.requires_grad = True
                trainable += int(param.numel())
            else:
                param.requires_grad = False
        return trainable

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

    def extract_vision_tokens(
        self,
        image_path: str,
        *,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """返回 vision tokens，形状 [num_tokens, vision_dim]。"""
        self._ensure_model()
        if self._model is None or self._processor is None:
            if not self.fallback_enabled:
                raise RuntimeError(f"Qwen 初始化失败且 fallback 关闭: {self._init_error}")
            fallback = self._fallback_visual(image_path=image_path)
            return fallback.unsqueeze(0)
        if not Path(image_path).exists():
            fallback = self._fallback_visual(image_path=image_path)
            return fallback.unsqueeze(0)

        image = Image.open(image_path).convert("RGB")
        model_inputs = self._processor(images=image, return_tensors="pt")
        if self._device == "cuda":
            model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}

        if requires_grad:
            features = self._model.get_image_features(
                pixel_values=model_inputs["pixel_values"],
                image_grid_thw=model_inputs.get("image_grid_thw"),
            )
        else:
            with torch.no_grad():
                features = self._model.get_image_features(
                    pixel_values=model_inputs["pixel_values"],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                )
        if features.dim() == 3:
            return features.squeeze(0)
        if features.dim() == 2:
            return features
        return features.reshape(features.shape[0], -1)

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
        use_vision_only: bool = False,
        llm_backbone_trainable: bool = False,
    ) -> torch.Tensor:
        """获取图像在 Qwen LLM space 中的 hidden state 表征。

        架构：
            Image → Vision Encoder → vision tokens → LLM backbone → hidden state

        默认使用 LLM backbone 的 hidden state，保证 latent space 与预训练对齐。
        如果 use_vision_only=True，则只使用 Vision Encoder 输出（不使用 LLM）。

        Args:
            image_path: 图像路径
            prompt: 可选的文本 prompt（用于引导 LLM 理解图像）
            layer: 返回哪层 hidden state（-1 = 最后一层）
            return_last_token_only: True = 返回 last token 的 hidden state
            use_vision_only: True = 只用 Vision Encoder，False = 通过 LLM backbone
            llm_backbone_trainable: True = LLM backbone 可训练（当前保留接口）

        Returns:
            hidden_state: [D] (last token 的 hidden state)
        """
        # 检查文件是否存在
        if not Path(image_path).exists():
            if not self.fallback_enabled:
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            return self._fallback_visual(image_path=image_path)

        self._ensure_model()
        if self._model is None or self._processor is None:
            if not self.fallback_enabled:
                raise RuntimeError(f"Qwen 初始化失败且 fallback 关闭: {self._init_error}")
            return self._fallback_visual(image_path=image_path)

        image = Image.open(image_path).convert("RGB")

        # 构建输入
        # Qwen2.5-VL 需要使用 chat template 来正确处理图像
        if prompt:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image}],
                }
            ]

        # 使用 processor 处理完整输入
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        if self._device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # 模式1：只用 Vision Encoder（不使用 LLM backbone）
        if use_vision_only:
            pixel_values = inputs.get("pixel_values")
            image_grid_thw = inputs.get("image_grid_thw")
            if pixel_values is not None:
                vision_emb = self._model.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
                # vision_emb: [1, num_patches, vision_dim]
                if vision_emb.dim() == 3:
                    vision_emb = vision_emb.squeeze(0)
                if return_last_token_only:
                    return self._pad_or_trim(vision_emb[-1, :])
                else:
                    return self._pad_or_trim(vision_emb.mean(dim=0))

        # 模式2：通过 LLM backbone 获取 hidden state
        # 关键：使用 output_hidden_states=True 获取 LLM 各层 hidden states
        input_ids = inputs.get("input_ids")
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        if input_ids is not None and pixel_values is not None:
            # 联合训练模式：
            # - Vision Encoder: 需要梯度（llm_backbone_trainable=False 时训练 Vision Encoder）
            # - LLM backbone: 冻结（requires_grad=False），保持 latent space 对齐

            # 冻结/解冻 LLM backbone
            self._set_llm_backbone_trainable(llm_backbone_trainable)

            # 使用完整模型的 forward，但保留梯度控制
            # Qwen2.5-VLForConditionalGeneration 接受 pixel_values 参数
            outputs = self._model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states

            # 选择指定层的 hidden state
            if layer == -1:
                # 最后一层
                selected = hidden_states[-1]
            elif 0 <= layer < len(hidden_states):
                selected = hidden_states[layer]
            else:
                selected = hidden_states[-1]

            # 返回 last token 的 hidden state
            if return_last_token_only:
                return self._pad_or_trim(selected[0, -1, :])
            else:
                return self._pad_or_trim(selected[0].mean(dim=0))

        # 回退到 vision embedding
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if pixel_values is not None:
            vision_emb = self._model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            if vision_emb.dim() == 3:
                vision_emb = vision_emb.squeeze(0)
            if return_last_token_only:
                return self._pad_or_trim(vision_emb[-1, :])
            else:
                return self._pad_or_trim(vision_emb.mean(dim=0))

        return self._fallback_visual(image_path=image_path)
