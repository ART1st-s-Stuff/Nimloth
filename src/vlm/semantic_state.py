"""语义状态构造器。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.vlm.qwen_adapter import QwenVLMAdapter


@dataclass
class SemanticStateOutput:
    z_t: torch.Tensor
    s_t: torch.Tensor
    cot_text: str


class SemanticStateGenerator:
    """将图像与上下文映射到 (z_t, s_t, CoT)。"""

    def __init__(self, vlm_adapter: QwenVLMAdapter) -> None:
        self.vlm_adapter = vlm_adapter

    def infer(
        self,
        image_path: str,
        history_image_paths: list[str],
        task_text: str,
        env_context: str,
    ) -> SemanticStateOutput:
        z_t = self.vlm_adapter.extract_visual_embedding(image_path=image_path)
        cot_text, s_t = self.vlm_adapter.generate_cot_and_state(
            image_paths=history_image_paths,
            task_text=task_text,
            env_context=env_context,
        )
        return SemanticStateOutput(z_t=z_t, s_t=s_t, cot_text=cot_text)
