"""Qwen2.5-VL 的独立 vLLM behavior-policy backend。"""

from __future__ import annotations

from typing import Any, Protocol

import torch

from nimloth.agent import AgentPrompt, PolicyDecision, sample_policy_decision
from nimloth.backbone.qwen25vl.policy import (
    collect_policy_images,
    render_policy_messages,
)
from nimloth.latent import LatentActionTokens, special_token_ids


class VLLMEngine(Protocol):
    def generate(self, prompts, sampling_params, *, use_tqdm: bool = False): ...


class QwenVLLMAgentPolicy:
    """vLLM 只计算八个 action token 的 score，采样规则继续由 Agent 定义。"""

    def __init__(
        self,
        *,
        engine: VLLMEngine,
        processor: Any,
        temperature: float,
        top_p: float,
        latent_token_count: int = 1,
    ) -> None:
        self.engine = engine
        self.processor = processor
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.latent_token_count = int(latent_token_count)
        token_map = special_token_ids(
            processor.tokenizer,
            latent_token_count=latent_token_count,
        )
        self.action_token_ids = tuple(
            token_map[token] for token in LatentActionTokens().action_tokens
        )

    @classmethod
    def from_model(
        cls,
        model_path: str,
        *,
        processor: Any,
        tensor_parallel_size: int,
        temperature: float,
        top_p: float,
        max_model_len: int,
        max_images: int,
        gpu_memory_utilization: float,
        latent_token_count: int,
        distributed_executor_backend: str | None = None,
    ) -> "QwenVLLMAgentPolicy":
        from vllm import LLM

        engine_kwargs: dict[str, Any] = {}
        if distributed_executor_backend is not None:
            engine_kwargs["distributed_executor_backend"] = distributed_executor_backend
        engine = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=int(tensor_parallel_size),
            dtype="bfloat16",
            max_model_len=int(max_model_len),
            gpu_memory_utilization=float(gpu_memory_utilization),
            limit_mm_per_prompt={"image": int(max_images)},
            **engine_kwargs,
        )
        return cls(
            engine=engine,
            processor=processor,
            temperature=temperature,
            top_p=top_p,
            latent_token_count=latent_token_count,
        )

    def reset_episode(self) -> None:
        """vLLM KV cache 按 request 管理，policy 本身无 episode state。"""

    def select_action(self, prompt: AgentPrompt) -> PolicyDecision:
        from vllm import SamplingParams

        text = render_policy_messages(
            prompt.bound_messages(),
            self.processor,
            latent_token_count=self.latent_token_count,
        )
        images = collect_policy_images(prompt.bound_messages())
        request: dict[str, Any] = {"prompt": text}
        if images:
            request["multi_modal_data"] = {"image": images}
        params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1,
            logprobs=len(self.action_token_ids),
            allowed_token_ids=list(self.action_token_ids),
            detokenize=False,
        )
        output = self.engine.generate([request], params, use_tqdm=False)[0].outputs[0]
        token_logprobs = output.logprobs[0]
        scores = torch.tensor(
            [
                float(token_logprobs[token_id].logprob)
                if token_id in token_logprobs
                else float("-inf")
                for token_id in self.action_token_ids
            ],
            dtype=torch.float32,
        )
        return sample_policy_decision(
            scores,
            temperature=self.temperature,
            top_p=self.top_p,
        )


__all__ = ["QwenVLLMAgentPolicy", "VLLMEngine"]
