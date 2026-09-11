from nimloth.backbone.qwen25vl.early_generation import decode_response


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 1
    def decode(self, ids, **kwargs):
        assert kwargs['skip_special_tokens'] is False
        return ''.join({2: '<|action_start|>', 3: '<|action_(0)|>', 4: '<|action_end|>'}[i] for i in ids)


def test_raw_special_tokens_not_removed():
    assert decode_response(Tokenizer(), [2, 3, 4, 0, 1]) == '<|action_start|><|action_(0)|><|action_end|>'


def test_inject_only_after_actual_generated_boundary(monkeypatch):
    import sys
    from types import SimpleNamespace

    from nimloth.agent.evaluation_protocol import EarlyProtocol
    from nimloth.backbone.qwen25vl.early_generation import EarlyVLLMGenerator
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=lambda **kw: kw))
    class TextTokenizer:
        eos_token_id = 0
        pad_token_id = 1
        def encode(self, text, **kwargs):
            return [10]
        def decode(self, ids, **kwargs):
            return ''.join({2: '<think>real', 3: '</think>\n', 4: '<|action_start|><|action_(1)|><|action_end|>'}[i] for i in ids)
    calls = []
    class LLM:
        def generate(self, requests, params, **kwargs):
            calls.append((requests[0], params))
            ids = [2, 3] if len(calls) == 1 else [4]
            return [SimpleNamespace(prompt_token_ids=[10], outputs=[SimpleNamespace(token_ids=ids, finish_reason='stop', stop_reason='</think>' if len(calls) == 1 else None)])]
    generator = EarlyVLLMGenerator.__new__(EarlyVLLMGenerator)
    generator.protocol = EarlyProtocol('stage2', 1, 'inject')
    generator.query_ids = [8]
    generator.tokenizer = TextTokenizer()
    generator.processor = SimpleNamespace(apply_chat_template=lambda *a, **kw: 'prompt')
    generator.config = SimpleNamespace(temperature=0., top_p=1., generation_seed=0, max_response_tokens=10)
    generator.llm = LLM()
    output = generator.generate([{'role': 'user', 'content': 'test'}], [])
    assert calls[1][0]['prompt_token_ids'] == [10, 2, 3, 8]
    assert output.sampled_token_ids == (2, 3, 4)
    assert output.inserted_token_ids == (8,)
    assert '</think>\n<|latent_state|><|action_start|>' in output.text
    assert not calls[1][1].get('stop')  # action/EOS remain unconstrained
    calls.clear()
    class Incomplete:
        def generate(self, *args, **kwargs):
            return [SimpleNamespace(prompt_token_ids=[10], outputs=[SimpleNamespace(token_ids=[2], finish_reason='length')])]
    generator.llm = Incomplete()
    assert generator.generate([{'role': 'user', 'content': 'test'}], []).inserted_token_ids == ()
