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
    PolicyReplayOutput,
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
    continue_final_message: bool = True,
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
        continue_final_message=continue_final_message,
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
    input_ids = model_inputs["input_ids"][0]
    action_start_positions = (
        input_ids == token_id_map[tokens.action_start]
    ).nonzero(as_tuple=True)[0]
    if action_start_positions.numel() == 0:
        raise RuntimeError("<|action_start|> token not found in Agent policy prompt")
    action_start_position = int(action_start_positions[-1].item())
    logits_to_keep = _logits_to_keep_positions([action_start_position])
    outputs = model(
        **model_inputs,
        logits_to_keep=logits_to_keep,
        output_hidden_states=False,
        return_dict=True,
    )
    action_token_ids = torch.tensor(
        [token_id_map[token] for token in tokens.action_tokens],
        device=outputs.logits.device,
    )
    return outputs.logits[0, 0, action_token_ids].float()


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


def _row_entropies(log_probs: torch.Tensor) -> torch.Tensor:
    probabilities = log_probs.exp()
    terms = torch.where(
        probabilities > 0,
        probabilities * log_probs,
        torch.zeros_like(log_probs),
    )
    return -terms.sum(dim=-1)


def _logits_to_keep_positions(positions: Sequence[int]) -> list[int]:
    """Build a native position index for device-mapped Qwen replay.

    A Python list is a valid PyTorch advanced index and Accelerate leaves its
    integer elements unchanged.  A tensor, including one initially created on
    CPU, is moved by the top-level device hook to Qwen's input GPU; that is the
    wrong device when the final norm/lm_head and hidden states are placed on a
    second GPU.
    """

    return [int(position) for position in positions]


def replay_policy_token_log_probs(
    *,
    samples: Sequence[PolicyReplayInput],
    model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
    token_value_head: torch.nn.Module | None = None,
    compute_token_values: bool = True,
) -> PolicyReplayOutput:
    """只在 trajectory loss-mask 位置重放 response token 概率。

    reasoning token 使用完整词表但屏蔽 Nimloth 注入 token；action token 只使用八个
    action token。``logits_to_keep`` 在 ``lm_head`` 前选择位置，避免生成整段 prompt
    的 full-vocabulary logits。
    """

    if not samples or any(sample.token_trace is None for sample in samples):
        raise ValueError("token policy replay requires a trace for every sample")
    tokens = LatentActionTokens()
    action_token_ids = tuple(token_id_map[token] for token in tokens.action_tokens)
    injected_token_ids = tuple(token_id_map.values())
    selected_log_probs: list[torch.Tensor] = []
    selected_full_log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    selected_hidden_states: list[torch.Tensor] = []
    replayed_action_log_probs: list[torch.Tensor] = []
    planner_samples = [sample.planner_trace is not None for sample in samples]
    if any(planner_samples) and not all(planner_samples):
        raise ValueError("one policy batch cannot mix planner and direct actions")

    for sample in samples:
        trace = sample.token_trace
        assert trace is not None
        if trace.action_token_ids != action_token_ids:
            raise ValueError(
                "recorded action token mapping does not match current tokenizer"
            )
        if sample.credit_assignment in {"turn", "token"}:
            assert sample.assistant_response is not None
            response_prefix = "<think>"
            if not sample.assistant_response.startswith(response_prefix):
                raise ValueError("turn response must start with '<think>'")
            decoded_continuation = processor.tokenizer.decode(
                list(trace.token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
                spaces_between_special_tokens=False,
            )
            if decoded_continuation != sample.assistant_response[
                len(response_prefix) :
            ]:
                raise ValueError(
                    "recorded policy trace does not decode to the assistant response"
                )
        bound_messages = sample.prompt.bound_messages()
        text = render_policy_messages(
            bound_messages,
            processor,
            latent_token_count=sample.latent_token_count,
        )
        images = collect_policy_images(bound_messages)
        inputs = processor(
            text=[text],
            images=[images] if images else None,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_length = int(model_inputs["input_ids"].shape[1])
        continuation = torch.tensor(
            [trace.token_ids],
            dtype=model_inputs["input_ids"].dtype,
            device=device,
        )
        model_inputs["input_ids"] = torch.cat(
            (model_inputs["input_ids"], continuation),
            dim=1,
        )
        if "attention_mask" in model_inputs:
            extension = torch.ones(
                (1, continuation.shape[1]),
                dtype=model_inputs["attention_mask"].dtype,
                device=device,
            )
            model_inputs["attention_mask"] = torch.cat(
                (model_inputs["attention_mask"], extension),
                dim=1,
            )
        selected_indices = [
            index for index, selected in enumerate(trace.loss_mask) if selected
        ]
        replay_indices = list(selected_indices)
        action_position = trace.token_roles.index("action")
        if sample.planner_trace is not None:
            replay_indices.append(action_position)
        logits_to_keep = _logits_to_keep_positions(
            [prompt_length - 1 + index for index in replay_indices]
        )
        captured: dict[str, torch.Tensor] = {}
        handle = None
        if sample.credit_assignment == "token" and compute_token_values:
            if token_value_head is None:
                raise ValueError("token credit replay requires a TokenValueHead")
            root = getattr(model, "module", model)
            get_output_embeddings = getattr(root, "get_output_embeddings", None)
            lm_head = get_output_embeddings() if get_output_embeddings else None
            if not isinstance(lm_head, torch.nn.Module):
                raise RuntimeError("could not locate Qwen lm_head for token critic")

            def capture_lm_head_input(
                _module: torch.nn.Module,
                inputs: tuple[torch.Tensor, ...],
            ) -> None:
                if not inputs:
                    raise RuntimeError("Qwen lm_head received no hidden states")
                captured["hidden"] = inputs[0]

            handle = lm_head.register_forward_pre_hook(capture_lm_head_input)
        try:
            outputs = model(
                **model_inputs,
                logits_to_keep=logits_to_keep,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            if handle is not None:
                handle.remove()
        if sample.credit_assignment == "token" and compute_token_values:
            hidden = captured.get("hidden")
            if hidden is None or hidden.ndim != 3 or hidden.shape[:2] != (
                1,
                len(replay_indices),
            ):
                raise RuntimeError(
                    "Qwen lm_head hook did not capture selected token hidden states"
                )
            selected_hidden_states.append(hidden[0, : len(selected_indices)])
        selected_token_ids = [trace.token_ids[index] for index in selected_indices]
        selected_roles = [trace.token_roles[index] for index in selected_indices]
        for row, token_id, role in zip(
            outputs.logits[0, : len(selected_indices)],
            selected_token_ids,
            selected_roles,
            strict=True,
        ):
            logits = row.float()
            if role == "action":
                role_logits = logits[
                    torch.tensor(action_token_ids, dtype=torch.long, device=logits.device)
                ]
                try:
                    selected_index = action_token_ids.index(token_id)
                except ValueError as error:
                    raise ValueError(
                        f"recorded action token id {token_id} is outside action vocabulary"
                    ) from error
            elif role == "reasoning":
                role_logits = logits.clone()
                role_logits[
                    torch.tensor(
                        injected_token_ids,
                        dtype=torch.long,
                        device=logits.device,
                    )
                ] = float("-inf")
                selected_index = token_id
            else:
                raise ValueError("injected token appeared in PPO loss mask")
            log_probs = behavior_log_probs(
                role_logits,
                temperature=sample.sampling_temperature,
                top_p=sample.sampling_top_p,
            )
            selected_log_probs.append(log_probs[selected_index])
            entropies.append(_row_entropies(log_probs.unsqueeze(0))[0])
            if role == "reasoning":
                full_log_probs = torch.log_softmax(
                    logits / sample.sampling_temperature,
                    dim=-1,
                )
                selected_full_log_probs.append(full_log_probs[token_id])
        if sample.planner_trace is not None:
            action_logits = outputs.logits[0, len(selected_indices)].float()
            restricted = action_logits[
                torch.tensor(
                    action_token_ids,
                    dtype=torch.long,
                    device=action_logits.device,
                )
            ]
            replayed_action_log_probs.append(torch.log_softmax(restricted, dim=-1))
    token_values = None
    if selected_hidden_states:
        assert token_value_head is not None
        token_values = token_value_head(
            torch.cat(selected_hidden_states, dim=0).detach()
        )
    return PolicyReplayOutput(
        selected_log_probs=torch.stack(selected_log_probs),
        entropies=torch.stack(entropies),
        token_values=token_values,
        action_log_probs=(
            torch.stack(replayed_action_log_probs)
            if replayed_action_log_probs
            else None
        ),
        selected_full_log_probs=(
            torch.stack(selected_full_log_probs)
            if selected_full_log_probs
            else None
        ),
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
        token_value_head: torch.nn.Module | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.token_id_map = token_id_map
        self.device = device
        self.token_value_head = token_value_head

    def __call__(
        self,
        samples: tuple[PolicyReplayInput, ...],
    ) -> PolicyReplayOutput:
        traced = [sample.token_trace is not None for sample in samples]
        if any(traced) and not all(traced):
            raise ValueError("one PPO batch cannot mix traced and legacy policy samples")
        if all(traced):
            with evaluating(self.model):
                return replay_policy_token_log_probs(
                    samples=samples,
                    model=self.model,
                    processor=self.processor,
                    token_id_map=self.token_id_map,
                    device=self.device,
                    token_value_head=self.token_value_head,
                )
        selected, distributions = replay_rollout_action_log_probs(
            samples=samples,
            model=self.model,
            processor=self.processor,
            token_id_map=self.token_id_map,
            device=self.device,
        )
        return PolicyReplayOutput(
            selected_log_probs=selected,
            entropies=_row_entropies(distributions),
        )
