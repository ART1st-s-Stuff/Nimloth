"""Unconstrained vLLM generation for format/query evaluation; no planner masks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from nimloth.agent.template import bind_image_placeholders


@dataclass(frozen=True)
class RawGeneration:
    text: str
    sampled_token_ids: tuple[int, ...]
    inserted_token_ids: tuple[int, ...]
    finish_reason: str


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
        kwargs = dict(model=str(config.checkpoint), dtype='bfloat16',
                      tensor_parallel_size=config.tensor_parallel_size,
                      max_model_len=config.max_model_len,
                      max_num_batched_tokens=config.max_model_len,
                      gpu_memory_utilization=config.gpu_memory_utilization,
                      enforce_eager=True, enable_chunked_prefill=False,
                      enable_prefix_caching=config.vllm_enable_prefix_caching,
                      limit_mm_per_prompt={'image': config.history_turns + 1})
        if config.max_pixels is not None:
            kwargs['mm_processor_kwargs'] = {'max_pixels': config.max_pixels}
        if config.vllm_distributed_executor_backend:
            kwargs['distributed_executor_backend'] = config.vllm_distributed_executor_backend
        self.llm = LLM(**kwargs)

    def generate(self, messages: list[dict], images: list[Any]) -> RawGeneration:
        from vllm import SamplingParams
        bound = bind_image_placeholders(messages, images)
        prompt = self.processor.apply_chat_template(bound, tokenize=False, add_generation_prompt=True)
        request = {'prompt': prompt, 'multi_modal_data': {'image': images}}
        def sample(req, budget, stop=None):
            params = SamplingParams(temperature=self.config.temperature, top_p=self.config.top_p,
                                    max_tokens=budget, seed=self.config.generation_seed,
                                    skip_special_tokens=False, stop=stop,
                                    include_stop_str_in_output=True)
            return self.llm.generate([req], params, use_tqdm=False)[0]
        inject = self.protocol.query_mode == 'inject'
        first = sample(request, self.config.max_response_tokens, ['</think>'] if inject else None)
        output = first.outputs[0]
        sampled = list(output.token_ids)
        text = decode_response(self.tokenizer, sampled)
        # 只有模型实际生成边界才注入query；到达长度/EOS时不补CoT或动作。
        if (not inject or getattr(output, 'stop_reason', None) != '</think>'
                or re.search(r'</think>\s*$', text) is None):
            return RawGeneration(text, tuple(sampled), (), str(output.finish_reason))
        budget = self.config.max_response_tokens - len(sampled) - len(self.query_ids)
        if budget < 1:
            return RawGeneration(text, tuple(sampled), (), 'length')
        continuation = {'prompt_token_ids': self.tokenizer.encode(prompt, add_special_tokens=False) + sampled + self.query_ids,
                        'multi_modal_data': {'image': images}}
        second = sample(continuation, budget).outputs[0]
        tail = list(second.token_ids)
        return RawGeneration(text + ''.join(self.protocol.query_tokens) + decode_response(self.tokenizer, tail),
                             tuple(sampled + tail), tuple(self.query_ids), str(second.finish_reason))
