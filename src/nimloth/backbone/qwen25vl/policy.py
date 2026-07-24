"""Qwen2.5-VL 到公共 Agent 离散动作 policy 协议的适配。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from nimloth.agent import (
    AgentPrompt,
    PolicyDecision,
    PolicyReplayInput,
    behavior_log_probs,
    categorical_entropy_from_log_probs,
    sample_policy_decision,
)
from nimloth.latent import (
    LatentActionTokens,
    normalize_latent_state_blocks,
    special_token_ids,
)
from nimloth.util.module import evaluating

def validate_agent_policy_protocol(model_config: Any) -> int:
    """确认 checkpoint 满足 inject policy 协议并返回其 latent token 数。"""

    latent_count = int(getattr(model_config, "nimloth_latent_token_count", 1))
    query_mode = getattr(model_config, "nimloth_latent_query_mode", None)
    if latent_count < 1 or query_mode != "inject":
        raise ValueError(
            "Agent action runtime requires a positive-k inject checkpoint; "
            f"got latent_token_count={latent_count}, latent_query_mode={query_mode!r}"
        )
    return latent_count


def _rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert("RGB")
    if hasattr(value, "shape"):
        return Image.fromarray(value).convert("RGB")
    raise TypeError(f"unsupported Agent image type: {type(value)!r}")


def collect_policy_images(messages: Sequence[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image":
                images.append(_rgb_image(part["image"]))
    return images


def render_policy_messages(
    messages: Sequence[dict[str, Any]],
    processor: Any,
    *,
    latent_token_count: int,
) -> str:
    """Render a policy prompt without putting runtime images in a JSON cache key.

    Qwen's chat template only needs to know where image parts occur.  The actual
    PIL images are passed to the processor separately by
    :func:`collect_policy_images`.
    """

    renderable: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            renderable.append(dict(message))
            continue
        parts: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image":
                parts.append({**part, "image": "<image>"})
            else:
                parts.append(dict(part))
        renderable.append({**message, "content": parts})
    text = processor.apply_chat_template(
        renderable,
        tokenize=False,
        add_generation_prompt=False,
    )
    return normalize_latent_state_blocks(text, latent_token_count)


def action_logits_for_messages(
    *,
    model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    messages: list[dict[str, Any]],
    device: torch.device,
    latent_token_count: int = 1,
) -> torch.Tensor:
    """Return the 8 action-token logits for one explicit Agent query."""

    tokens = LatentActionTokens()
    text = render_policy_messages(
        messages,
        processor,
        latent_token_count=latent_token_count,
    )
    images = collect_policy_images(messages)
    inputs = processor(
        text=[text],
        images=[images] if images else None,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**model_inputs, output_hidden_states=False, return_dict=True)

    input_ids = model_inputs["input_ids"][0]
    action_start_positions = (
        input_ids == token_id_map[tokens.action_start]
    ).nonzero(as_tuple=True)[0]
    if action_start_positions.numel() == 0:
        raise RuntimeError("<|action_start|> token not found in Agent policy prompt")
    action_start_position = int(action_start_positions[-1].item())
    action_token_ids = torch.tensor(
        [token_id_map[token] for token in tokens.action_tokens],
        device=outputs.logits.device,
    )
    return outputs.logits[0, action_start_position, action_token_ids].float()


class QwenAgentPolicy:
    """运行 Qwen，并从可审计的 behavior distribution 中采样。"""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        processor: Any,
        device: torch.device,
        temperature: float,
        top_p: float,
        latent_token_count: int = 1,
        token_id_map: dict[str, int] | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.latent_token_count = latent_token_count
        self.token_id_map = token_id_map or special_token_ids(processor.tokenizer)

    def reset_episode(self) -> None:
        """Qwen direct policy 不保存跨 step 状态。"""

    def select_action(self, prompt: AgentPrompt) -> PolicyDecision:
        # 行为概率必须可被 PPO 确定性重放，不能受 LoRA dropout 影响。
        with evaluating(self.model), torch.no_grad():
            logits = action_logits_for_messages(
                model=self.model,
                processor=self.processor,
                token_id_map=self.token_id_map,
                messages=prompt.bound_messages(),
                device=self.device,
                latent_token_count=self.latent_token_count,
            )
            return sample_policy_decision(
                logits,
                temperature=self.temperature,
                top_p=self.top_p,
            )


def batch_action_log_probs(
    *,
    model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    messages: Sequence[list[dict[str, Any]]],
    taken_action_indices: Sequence[int],
    temperatures: Sequence[float],
    top_ps: Sequence[float],
    device: torch.device,
    latent_token_count: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute PPO probabilities with the same prompts and sampling transform."""

    lengths = {
        len(messages),
        len(taken_action_indices),
        len(temperatures),
        len(top_ps),
    }
    if len(lengths) != 1:
        raise ValueError("PPO policy batch fields must have equal lengths")

    selected: list[torch.Tensor] = []
    distributions: list[torch.Tensor] = []
    for prompt, action_index, temperature, top_p in zip(
        messages,
        taken_action_indices,
        temperatures,
        top_ps,
        strict=True,
    ):
        logits = action_logits_for_messages(
            model=model,
            processor=processor,
            token_id_map=token_id_map,
            messages=prompt,
            device=device,
            latent_token_count=latent_token_count,
        )
        log_probs = behavior_log_probs(
            logits,
            temperature=float(temperature),
            top_p=float(top_p),
        )
        selected.append(log_probs[int(action_index)])
        distributions.append(log_probs)
    return torch.stack(selected), torch.stack(distributions)


def replay_rollout_action_log_probs(
    *,
    samples: Sequence[PolicyReplayInput],
    model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用当前 Qwen 重放 rollout 时保存的完整 prompt 与采样变换。"""

    if not samples:
        raise ValueError("PPO policy batch must not be empty")
    latent_token_counts = {
        int(sample.latent_token_count) for sample in samples
    }
    if len(latent_token_counts) != 1:
        raise ValueError("one PPO batch cannot mix latent token counts")
    bound_messages = [sample.prompt.bound_messages() for sample in samples]
    # eval mode 关闭 dropout 但不关闭梯度，保证 PPO old/new policy 可比较。
    with evaluating(model):
        return batch_action_log_probs(
            model=model,
            processor=processor,
            token_id_map=token_id_map,
            messages=bound_messages,
            taken_action_indices=[
                sample.action_index for sample in samples
            ],
            temperatures=[
                sample.sampling_temperature for sample in samples
            ],
            top_ps=[sample.sampling_top_p for sample in samples],
            device=device,
            latent_token_count=latent_token_counts.pop(),
        )


class QwenActionLogProbReplay:
    """保存 Qwen policy 重放所需的 processor 与 token 协议。"""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        processor: Any,
        token_id_map: dict[str, int],
        device: torch.device,
    ) -> None:
        self.model = model
        self.processor = processor
        self.token_id_map = token_id_map
        self.device = device

    def __call__(
        self,
        samples: tuple[PolicyReplayInput, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return replay_rollout_action_log_probs(
            samples=samples,
            model=self.model,
            processor=self.processor,
            token_id_map=self.token_id_map,
            device=self.device,
        )
