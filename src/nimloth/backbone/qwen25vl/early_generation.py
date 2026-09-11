"""Unconstrained vLLM generation for format/query evaluation; no planner masks."""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from nimloth.agent.evaluation_protocol import EarlyProtocol
from nimloth.agent.template import bind_image_placeholders


@dataclass(frozen=True)
class RawGeneration:
    text: str
    sampled_token_ids: tuple[int, ...]
    inserted_token_ids: tuple[int, ...]
    finish_reason: str
    stop_reason: int | str | None = None


@dataclass(frozen=True)
class Stage1RawGenerationValidation:
    """Exact termination and body evidence for one unconstrained generation."""

    format_correct: bool
    reason: str
    raw_response: str
    parsed_body: str | None
    eos_generated: bool
    eos_token_index: int | None
    length_reached: bool
    parser_result: dict[str, Any] | None
    generation_text_matches: bool = True
    inserted_tokens_absent: bool = True


def validate_stage1_raw_generation(
    generation: RawGeneration,
    tokenizer: Any,
) -> Stage1RawGenerationValidation:
    """Validate sampled Stage 1 termination before parsing or action execution."""

    validation = validate_stage1_sampled_tokens(
        generation.sampled_token_ids,
        tokenizer,
        finish_reason=generation.finish_reason,
    )
    canonical_text = decode_response(tokenizer, list(generation.sampled_token_ids))
    text_matches = generation.text == canonical_text
    inserted_absent = not generation.inserted_token_ids
    validation = replace(
        validation,
        generation_text_matches=text_matches,
        inserted_tokens_absent=inserted_absent,
    )
    if validation.reason in {
        "length_reached",
        "missing_eos",
        "content_after_eos",
        "empty_body",
    }:
        return validation
    if not inserted_absent or not text_matches:
        return Stage1RawGenerationValidation(
            False,
            "generation_text_mismatch",
            validation.raw_response,
            validation.parsed_body,
            validation.eos_generated,
            validation.eos_token_index,
            validation.length_reached,
            validation.parser_result,
            text_matches,
            inserted_absent,
        )
    return validation


def validate_stage1_sampled_tokens(
    sampled_token_ids: tuple[int, ...] | list[int],
    tokenizer: Any,
    *,
    finish_reason: str,
) -> Stage1RawGenerationValidation:
    """Pure shared validator for HF and vLLM sampled Stage 1 token streams."""

    sampled = [int(token_id) for token_id in sampled_token_ids]
    raw_response = tokenizer.decode(
        sampled,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise TypeError("Stage 1 evaluation requires one integer EOS token ID")
    pad_token_id = tokenizer.pad_token_id
    eos_index = next(
        (index for index, token_id in enumerate(sampled) if token_id == eos_token_id),
        None,
    )
    length_reached = finish_reason == "length"
    if length_reached:
        return Stage1RawGenerationValidation(
            False,
            "length_reached",
            raw_response,
            None,
            eos_index is not None,
            eos_index,
            True,
            None,
        )
    if eos_index is None:
        return Stage1RawGenerationValidation(
            False,
            "missing_eos",
            raw_response,
            None,
            False,
            None,
            False,
            None,
        )
    suffix = sampled[eos_index + 1 :]
    if suffix and (
        isinstance(pad_token_id, bool)
        or not isinstance(pad_token_id, int)
        or any(token_id != pad_token_id for token_id in suffix)
    ):
        return Stage1RawGenerationValidation(
            False,
            "content_after_eos",
            raw_response,
            None,
            True,
            eos_index,
            False,
            None,
        )
    body_ids = sampled[:eos_index]
    parsed_body = tokenizer.decode(
        body_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not body_ids or not parsed_body.strip():
        return Stage1RawGenerationValidation(
            False,
            "empty_body",
            raw_response,
            parsed_body,
            True,
            eos_index,
            False,
            None,
        )
    parsed = EarlyProtocol("stage1").parse(parsed_body)
    correct = bool(parsed["format_correct"])
    return Stage1RawGenerationValidation(
        correct,
        "ok" if correct else str(parsed["error"]),
        raw_response,
        parsed_body,
        True,
        eos_index,
        False,
        parsed,
    )


def decode_response(tokenizer: Any, token_ids: list[int]) -> str:
    ids = list(token_ids)
    terminal_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    while ids and ids[-1] in terminal_ids:
        ids.pop()
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


class EarlyVLLMGenerator:
    def __init__(self, config: Any, protocol: Any) -> None:
        from transformers import AutoProcessor
        from vllm import LLM
        self.processor = AutoProcessor.from_pretrained(str(config.checkpoint))
        self.tokenizer = self.processor.tokenizer
        self.config = config
        self.protocol = protocol
        self.query_ids = self.tokenizer.encode(''.join(protocol.query_tokens), add_special_tokens=False)
        for token in (*protocol.query_tokens, *(() if protocol.stage == 'vagen' else
                        ('<|action_start|>', '<|action_end|>', *(f'<|action_({i})|>' for i in range(8))))):
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1 or self.tokenizer.convert_ids_to_tokens(ids[0]) != token:
                raise ValueError(f'checkpoint lacks atomic protocol token: {token}')
        if config.vllm_mm_processor_cache_gb != 0:
            raise ValueError('original vLLM early evaluator does not support nonzero mm processor cache GB')
        kwargs = {
            'model': str(config.checkpoint), 'dtype': 'bfloat16',
            'tensor_parallel_size': config.tensor_parallel_size,
            'max_model_len': config.max_model_len,
            'max_num_batched_tokens': config.max_model_len,
            'gpu_memory_utilization': config.gpu_memory_utilization,
            'enforce_eager': True, 'enable_chunked_prefill': False,
            'enable_prefix_caching': config.vllm_enable_prefix_caching,
            'limit_mm_per_prompt': {'image': config.history_turns + 1},
        }
        if config.max_pixels is not None:
            kwargs['mm_processor_kwargs'] = {'max_pixels': config.max_pixels}
        if config.vllm_distributed_executor_backend:
            kwargs['distributed_executor_backend'] = config.vllm_distributed_executor_backend
        self.llm = LLM(**kwargs)

    def generate(self, messages: list[dict], images: list[Any]) -> RawGeneration:
        return self.generate_batch([(messages, images)])[0]

    def generate_batch(self, inputs: list[tuple[list[dict], list[Any]]]) -> list[RawGeneration]:
        """Batch independent prompts and then only eligible query continuations."""
        from vllm import SamplingParams

        if not inputs:
            return []
        prompts = [self.processor.apply_chat_template(
            bind_image_placeholders(messages, images), tokenize=False,
            add_generation_prompt=True) for messages, images in inputs]
        requests = [{'prompt': prompt, 'multi_modal_data': {'image': images}}
                    for prompt, (_, images) in zip(prompts, inputs, strict=True)]

        def sample(requests, budgets, stop=None):
            params = [SamplingParams(
                temperature=self.config.temperature, top_p=self.config.top_p,
                max_tokens=budget, seed=self.config.generation_seed,
                skip_special_tokens=False, stop=stop,
                include_stop_str_in_output=True) for budget in budgets]
            outputs = self.llm.generate(requests, params if len(params) > 1 else params[0], use_tqdm=False)
            if len(outputs) != len(requests):
                raise ValueError('vLLM batch response count differs from requests')
            return outputs

        inject = self.protocol.query_mode == 'inject'
        first = sample(requests, [self.config.max_response_tokens] * len(requests),
                       ['</think>'] if inject else None)
        results = []
        continuations, budgets, indices = [], [], []
        for index, response in enumerate(first):
            output = response.outputs[0]
            sampled = list(output.token_ids)
            text = decode_response(self.tokenizer, sampled)
            result = RawGeneration(text, tuple(sampled), (), str(output.finish_reason),
                                   getattr(output, 'stop_reason', None))
            if (inject and getattr(output, 'stop_reason', None) == '</think>'
                    and re.search(r'</think>\s*$', text) is not None):
                budget = self.config.max_response_tokens - len(sampled) - len(self.query_ids)
                if budget < 1:
                    result = RawGeneration(text, tuple(sampled), (), 'length')
                else:
                    indices.append(index)
                    budgets.append(budget)
                    continuations.append({
                        'prompt_token_ids': self.tokenizer.encode(prompts[index], add_special_tokens=False)
                        + sampled + self.query_ids,
                        'multi_modal_data': {'image': inputs[index][1]}})
            results.append(result)
        if continuations:
            for index, response in zip(indices, sample(continuations, budgets), strict=True):
                output = response.outputs[0]
                tail = list(output.token_ids)
                first_result = results[index]
                results[index] = RawGeneration(
                    first_result.text + ''.join(self.protocol.query_tokens) + decode_response(self.tokenizer, tail),
                    first_result.sampled_token_ids + tuple(tail), tuple(self.query_ids),
                    str(output.finish_reason), getattr(output, 'stop_reason', None))
        return results
