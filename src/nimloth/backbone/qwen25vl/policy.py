"""Qwen2.5-VL 到公共 Agent 离散动作 policy 协议的适配。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import torch
from PIL import Image

from nimloth.agent import AgentPrompt, PolicyDecision, bind_image_placeholders
from nimloth.latent import (
    LatentActionTokens,
    normalize_latent_state_blocks,
    special_token_ids,
)
from nimloth.util.module import evaluating

if TYPE_CHECKING:
    from nimloth.rollout.encoding import EncodedTransition


def validate_agent_policy_protocol(model_config: Any) -> None:
    """确认 checkpoint 满足当前已实现的 k=1 inject policy 协议。"""

    latent_count = int(getattr(model_config, "nimloth_latent_token_count", 1))
    query_mode = getattr(model_config, "nimloth_latent_query_mode", None)
    if latent_count != 1 or query_mode != "inject":
        raise ValueError(
            "Agent action runtime currently requires a k=1 inject checkpoint; "
            f"got latent_token_count={latent_count}, latent_query_mode={query_mode!r}"
        )


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


def behavior_log_probs(
    action_logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Return the exact categorical distribution used to select an action."""

    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if temperature == 0.0:
        chosen = action_logits.argmax(dim=-1, keepdim=True)
        log_probs = torch.full_like(action_logits, float("-inf"))
        return log_probs.scatter(dim=-1, index=chosen, value=0.0)

    scaled_logits = action_logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            scaled_logits, dim=-1, descending=True
        )
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        sorted_keep = cumulative_before < top_p
        keep = torch.zeros_like(sorted_keep).scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_keep,
        )
        scaled_logits = scaled_logits.masked_fill(~keep, float("-inf"))
    return torch.log_softmax(scaled_logits, dim=-1)


def categorical_entropy_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """计算可能包含 top-p ``-inf`` mask 的离散分布 entropy。"""

    probabilities = log_probs.exp()
    terms = torch.where(
        probabilities > 0,
        probabilities * log_probs,
        torch.zeros_like(log_probs),
    )
    return -terms.sum(dim=-1).mean()


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
            log_probs = behavior_log_probs(
                logits,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            if self.temperature == 0.0:
                action_index = int(logits.argmax().item())
            else:
                action_index = int(torch.multinomial(log_probs.exp(), 1).item())
        return PolicyDecision(
            action_index=action_index,
            action_log_probs=tuple(float(value) for value in log_probs.cpu().tolist()),
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
    transitions: Sequence[EncodedTransition],
    model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用当前 Qwen 重放 rollout 时保存的完整 prompt 与采样变换。"""

    if not transitions:
        raise ValueError("PPO policy batch must not be empty")
    latent_token_counts = {
        int(transition.latent_token_count) for transition in transitions
    }
    if len(latent_token_counts) != 1:
        raise ValueError("one PPO batch cannot mix latent token counts")
    bound_messages = [
        bind_image_placeholders(
            transition.policy_messages,
            transition.policy_image_paths,
        )
        for transition in transitions
    ]
    # eval mode 关闭 dropout 但不关闭梯度，保证 PPO old/new policy 可比较。
    with evaluating(model):
        return batch_action_log_probs(
            model=model,
            processor=processor,
            token_id_map=token_id_map,
            messages=bound_messages,
            taken_action_indices=[
                transition.action_index for transition in transitions
            ],
            temperatures=[
                transition.sampling_temperature for transition in transitions
            ],
            top_ps=[transition.sampling_top_p for transition in transitions],
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
        transitions: tuple["EncodedTransition", ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return replay_rollout_action_log_probs(
            transitions=transitions,
            model=self.model,
            processor=self.processor,
            token_id_map=self.token_id_map,
            device=self.device,
        )
