"""Qwen2.5-VL 的独立 vLLM behavior-policy backend。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from nimloth.agent import AgentPrompt, PolicyDecision, PolicyTokenTrace
from nimloth.backbone.qwen25vl.policy import (
    collect_policy_images,
    render_policy_messages,
)
from nimloth.backbone.qwen25vl.turn_generation import (
    TurnGenerationSpec,
    find_token_subsequence,
)
from nimloth.backbone.qwen25vl.vllm_hidden import (
    VLLMPolicyState,
    abort_policy_state_capture,
    pop_policy_state_capture,
    start_policy_state_capture,
)
from nimloth.latent import (
    LatentActionTokens,
    latent_state_tokens,
    special_token_ids,
)


class VLLMEngine(Protocol):
    def generate(self, prompts, sampling_params, *, use_tqdm: bool = False): ...

    def collective_rpc(self, method, args=()): ...


@dataclass(frozen=True)
class QwenTurnGeneration:
    """One real Qwen CoT plus selected states from that same vLLM forward."""

    qwen_decision: PolicyDecision
    policy_state: VLLMPolicyState


class QwenVLLMAgentPolicy:
    """vLLM behavior policy；支持 action-only 与单请求 turn-credit 生成。"""

    def __init__(
        self,
        *,
        engine: VLLMEngine,
        processor: Any,
        temperature: float,
        top_p: float,
        latent_token_count: int = 1,
        credit_assignment: Literal["action", "turn", "token"] = "action",
        max_reasoning_tokens: int = 64,
        capture_policy_state: bool = False,
    ) -> None:
        if credit_assignment not in {"action", "turn", "token"}:
            raise ValueError(
                f"unsupported PPO credit assignment: {credit_assignment!r}"
            )
        if max_reasoning_tokens < 1:
            raise ValueError("max_reasoning_tokens must be positive")
        self.engine = engine
        self.processor = processor
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.latent_token_count = int(latent_token_count)
        self.credit_assignment = credit_assignment
        self.max_reasoning_tokens = int(max_reasoning_tokens)
        self.capture_policy_state = bool(capture_policy_state)
        if self.capture_policy_state and credit_assignment not in {"turn", "token"}:
            raise ValueError("policy-state capture requires turn or token credit")
        self.prompt_mode = (
            "response" if credit_assignment in {"turn", "token"} else "action"
        )
        self.token_id_map = special_token_ids(
            processor.tokenizer,
            latent_token_count=latent_token_count,
        )
        self.action_token_ids = tuple(
            self.token_id_map[token]
            for token in LatentActionTokens().action_tokens
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
        credit_assignment: Literal["action", "turn", "token"] = "action",
        max_reasoning_tokens: int = 64,
        distributed_executor_backend: str | None = None,
        enforce_eager: bool = False,
        capture_policy_state: bool = False,
    ) -> "QwenVLLMAgentPolicy":
        from vllm import LLM

        engine_kwargs: dict[str, Any] = {}
        if distributed_executor_backend is not None:
            engine_kwargs["distributed_executor_backend"] = distributed_executor_backend
        if credit_assignment in {"turn", "token"}:
            engine_kwargs["logits_processors"] = [
                "nimloth.backbone.qwen25vl.vllm_logits:TurnResponseLogitsProcessor"
            ]
        if capture_policy_state:
            if not enforce_eager:
                raise ValueError(
                    "vLLM policy-state capture requires enforce_eager=True"
                )
            engine_kwargs["worker_extension_cls"] = (
                "nimloth.backbone.qwen25vl.vllm_hidden:"
                "PolicyStateCaptureWorkerExtension"
            )
        engine = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=int(tensor_parallel_size),
            dtype="bfloat16",
            max_model_len=int(max_model_len),
            gpu_memory_utilization=float(gpu_memory_utilization),
            limit_mm_per_prompt={"image": int(max_images)},
            # PPO 保存实际 temperature/top-p behavior 分布，不保存 raw logits 分布。
            logprobs_mode="processed_logprobs",
            enforce_eager=bool(enforce_eager),
            **engine_kwargs,
        )
        return cls(
            engine=engine,
            processor=processor,
            temperature=temperature,
            top_p=top_p,
            latent_token_count=latent_token_count,
            credit_assignment=credit_assignment,
            max_reasoning_tokens=max_reasoning_tokens,
            capture_policy_state=capture_policy_state,
        )

    def reset_episode(self) -> None:
        """vLLM KV cache 按 request 管理，policy 本身无 episode state。"""

    def select_action(self, prompt: AgentPrompt) -> PolicyDecision:
        if self.credit_assignment in {"turn", "token"}:
            return self._select_response(prompt)
        return self._select_action_only(prompt)

    def select_response_with_state(self, prompt: AgentPrompt) -> QwenTurnGeneration:
        """Generate one CoT and expose its latent/action-boundary states.

        Capture brackets the same vLLM request used for the real response.  A
        failed generation always clears worker state before the error escapes.
        """

        if not self.capture_policy_state:
            raise RuntimeError("this vLLM policy was not configured for state capture")
        tokens = LatentActionTokens()
        start_policy_state_capture(
            self.engine,
            latent_token_ids=tuple(
                self.token_id_map[token]
                for token in latent_state_tokens(self.latent_token_count, tokens)
            ),
            action_start_token_id=self.token_id_map[tokens.action_start],
            action_token_ids=self.action_token_ids,
        )
        try:
            decision = self._select_response(prompt)
            policy_state = pop_policy_state_capture(self.engine)
        except Exception:
            abort_policy_state_capture(self.engine)
            raise
        if policy_state.latent_hidden.shape[0] != self.latent_token_count:
            raise RuntimeError(
                "vLLM returned the wrong number of latent hidden rows: "
                f"{policy_state.latent_hidden.shape[0]} != {self.latent_token_count}"
            )
        if policy_state.action_logits.shape != (len(self.action_token_ids),):
            raise RuntimeError(
                "vLLM returned an invalid restricted action-logit shape: "
                f"{tuple(policy_state.action_logits.shape)}"
            )
        return QwenTurnGeneration(
            qwen_decision=decision,
            policy_state=policy_state,
        )

    def generate_state_prefix(self, prompt: AgentPrompt) -> str:
        """Generate a real terminal CoT prefix without executing its draft action."""

        if self.credit_assignment not in {"turn", "token"}:
            raise RuntimeError("terminal state CoT requires turn/token generation")
        decision = self._select_response(prompt)
        assert decision.response is not None
        boundary = decision.response.rfind(LatentActionTokens().action_start)
        if boundary < 0:
            raise RuntimeError("terminal Qwen response has no action boundary")
        return decision.response[
            : boundary + len(LatentActionTokens().action_start)
        ]

    def _request(self, prompt: AgentPrompt) -> tuple[dict[str, Any], list[Any]]:
        bound_messages = prompt.bound_messages()
        text = render_policy_messages(
            bound_messages,
            self.processor,
            latent_token_count=self.latent_token_count,
        )
        images = collect_policy_images(bound_messages)
        request: dict[str, Any] = {"prompt": text}
        if images:
            request["multi_modal_data"] = {"image": images}
        return request, images

    @staticmethod
    def _sampled_log_prob(output: Any, position: int, token_id: int) -> float:
        token_logprobs = output.logprobs[position]
        if token_id not in token_logprobs:
            raise RuntimeError(
                f"vLLM did not return the sampled token log-prob at position {position}"
            )
        value = float(token_logprobs[token_id].logprob)
        if not math.isfinite(value):
            raise RuntimeError("vLLM returned a non-finite sampled token log-prob")
        return value

    def _action_decision(
        self,
        request: dict[str, Any],
    ) -> tuple[Any, int, tuple[float, ...]]:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=1,
            logprobs=len(self.action_token_ids),
            allowed_token_ids=list(self.action_token_ids),
            detokenize=False,
        )
        output = self.engine.generate([request], params, use_tqdm=False)[0].outputs[0]
        if len(output.token_ids) != 1:
            raise RuntimeError("vLLM action stage must generate exactly one token")
        action_token_id = int(output.token_ids[0])
        try:
            action_index = self.action_token_ids.index(action_token_id)
        except ValueError as error:
            raise RuntimeError(
                f"vLLM generated non-action token id {action_token_id}"
            ) from error
        token_logprobs = output.logprobs[0]
        action_log_probs = tuple(
            float(token_logprobs[token_id].logprob)
            if token_id in token_logprobs
            else float("-inf")
            for token_id in self.action_token_ids
        )
        if self.temperature == 0.0:
            action_log_probs = tuple(
                0.0 if index == action_index else float("-inf")
                for index in range(len(self.action_token_ids))
            )
        return output, action_index, action_log_probs

    def _select_action_only(self, prompt: AgentPrompt) -> PolicyDecision:
        request, _images = self._request(prompt)
        _output, action_index, action_log_probs = self._action_decision(request)
        tokens = LatentActionTokens()
        return PolicyDecision(
            action_index=action_index,
            action_log_probs=action_log_probs,
            token_trace=PolicyTokenTrace(
                token_ids=(
                    self.action_token_ids[action_index],
                    self.token_id_map[tokens.action_end],
                ),
                old_log_probs=(action_log_probs[action_index], None),
                loss_mask=(True, False),
                token_roles=("action", "injected"),
                action_token_ids=self.action_token_ids,
            ),
        )

    def _select_response(self, prompt: AgentPrompt) -> PolicyDecision:
        from vllm import SamplingParams

        if prompt.messages[-1].get("content") != "<think>":
            raise ValueError("turn-credit policy prompt must end with '<think>'")
        request, _images = self._request(prompt)
        tokens = LatentActionTokens()
        close_ids = tuple(
            int(value)
            for value in self.processor.tokenizer.encode(
                "</think>",
                add_special_tokens=False,
            )
        )
        injected_tokens = (
            *latent_state_tokens(self.latent_token_count, tokens),
            tokens.action_start,
        )
        injected_ids = tuple(self.token_id_map[token] for token in injected_tokens)
        spec = TurnGenerationSpec(
            close_token_ids=close_ids,
            injected_token_ids=injected_ids,
            action_token_ids=self.action_token_ids,
            action_end_token_id=self.token_id_map[tokens.action_end],
            protocol_token_ids=tuple(self.token_id_map.values()),
            max_reasoning_tokens=self.max_reasoning_tokens,
        )
        params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=spec.max_output_tokens,
            logprobs=len(self.action_token_ids),
            stop_token_ids=[spec.action_end_token_id],
            ignore_eos=True,
            extra_args=spec.to_extra_args(),
            detokenize=False,
            skip_special_tokens=False,
        )
        request_output = self.engine.generate(
            [request],
            params,
            use_tqdm=False,
        )[0]
        output = request_output.outputs[0]
        continuation_ids = tuple(int(value) for value in output.token_ids)
        close_start = find_token_subsequence(continuation_ids, close_ids)
        if close_start is None:
            raise RuntimeError("vLLM turn response did not contain '</think>'")
        close_end = close_start + len(close_ids)
        expected_prefix_end = close_end + len(injected_ids)
        if continuation_ids[close_end:expected_prefix_end] != injected_ids:
            raise RuntimeError("vLLM turn response has an invalid injected prefix")
        if len(continuation_ids) != expected_prefix_end + 2:
            raise RuntimeError("vLLM turn response has an invalid action suffix length")
        action_token_id = continuation_ids[expected_prefix_end]
        action_end_id = continuation_ids[expected_prefix_end + 1]
        if action_end_id != spec.action_end_token_id:
            raise RuntimeError("vLLM turn response did not end at action_end")
        try:
            action_index = self.action_token_ids.index(action_token_id)
        except ValueError as error:
            raise RuntimeError(
                f"vLLM generated non-action token id {action_token_id}"
            ) from error

        action_token_logprobs = output.logprobs[expected_prefix_end]
        action_log_probs = tuple(
            float(action_token_logprobs[token_id].logprob)
            if token_id in action_token_logprobs
            else float("-inf")
            for token_id in self.action_token_ids
        )
        if self.temperature == 0.0:
            action_log_probs = tuple(
                0.0 if index == action_index else float("-inf")
                for index in range(len(self.action_token_ids))
            )

        reasoning_truncated = close_end > self.max_reasoning_tokens
        token_roles: list[Literal["reasoning", "action", "injected"]] = []
        loss_mask: list[bool] = []
        old_log_probs: list[float | None] = []
        for index, token_id in enumerate(continuation_ids):
            if index < close_end:
                sampled = index < self.max_reasoning_tokens
                role: Literal["reasoning", "action", "injected"] = (
                    "reasoning" if sampled else "injected"
                )
            elif index == expected_prefix_end:
                sampled = True
                role = "action"
            else:
                sampled = False
                role = "injected"
            token_roles.append(role)
            loss_mask.append(sampled)
            if not sampled:
                old_log_probs.append(None)
            elif role == "action":
                old_log_probs.append(action_log_probs[action_index])
            elif self.temperature == 0.0:
                old_log_probs.append(0.0)
            else:
                old_log_probs.append(self._sampled_log_prob(output, index, token_id))

        thought = self.processor.tokenizer.decode(
            list(continuation_ids[:close_start]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=False,
        )

        latent_block = "".join(latent_state_tokens(self.latent_token_count, tokens))
        response = (
            f"<think>{thought}</think>{latent_block}{tokens.action_start}"
            f"{tokens.action_tokens[action_index]}{tokens.action_end}"
        )
        decoded_continuation = self.processor.tokenizer.decode(
            list(continuation_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=False,
        )
        if decoded_continuation != response[len("<think>") :]:
            raise RuntimeError(
                "vLLM token continuation does not round-trip to assistant response"
            )
        return PolicyDecision(
            action_index=action_index,
            action_log_probs=action_log_probs,
            response=response,
            token_trace=PolicyTokenTrace(
                token_ids=tuple(continuation_ids),
                old_log_probs=tuple(old_log_probs),
                loss_mask=tuple(loss_mask),
                token_roles=tuple(token_roles),
                action_token_ids=self.action_token_ids,
                reasoning_text=str(thought),
                finish_reason="length" if reasoning_truncated else "stop",
                reasoning_truncated=reasoning_truncated,
            ),
        )


__all__ = ["QwenTurnGeneration", "QwenVLLMAgentPolicy", "VLLMEngine"]
