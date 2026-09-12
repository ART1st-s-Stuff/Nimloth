import pytest
import torch

from nimloth.training.sft.stage1.fsdp import (
    _bf16_master_forward,
    prepare_embedding_masters,
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
        self.head = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
        self.body = torch.nn.Linear(4, 4, dtype=torch.bfloat16)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head


def test_masters_preserve_scope_precision_and_functional_generation():
    model = Model()
    parameters = list(model.parameters())
    prepare_embedding_masters(model, 'float32')
    assert list(model.parameters()) == parameters
    assert model.embed.weight.dtype == model.head.weight.dtype == torch.float32
    assert model.body.weight.dtype == torch.bfloat16
    model.embed.weight.data[0, 0] = .020001
    before = model.embed.weight.detach().clone()
    head_before = model.head.weight.detach().clone()
    with _bf16_master_forward(model):
        hidden = model.embed(torch.tensor([0, 1]))
        output = model.head(hidden)
        assert hidden.dtype == output.dtype == torch.bfloat16
        assert torch.equal(output, torch.nn.functional.linear(hidden, head_before.bfloat16()))
    assert torch.equal(before, model.embed.weight)
    assert 'forward' not in model.embed.__dict__
    assert model.embed.weight.dtype == torch.float32


def test_generation_forward_restores_after_error():
    model = Model()
    prepare_embedding_masters(model, 'float32')
    with pytest.raises(RuntimeError), _bf16_master_forward(model):
        raise RuntimeError('test')
    assert 'forward' not in model.head.__dict__


def test_fp32_optimizer_updates_accumulate_below_bf16_spacing():
    model = Model()
    prepare_embedding_masters(model, 'float32')
    with torch.no_grad():
        model.head.weight.fill_(.02)
    opt = torch.optim.AdamW([model.head.weight], lr=5e-6, weight_decay=0)
    before = model.head.weight.detach().clone()
    model.head.weight.grad = torch.ones_like(model.head.weight)
    opt.step()
    assert not torch.equal(before, model.head.weight)
    assert opt.state[model.head.weight]['exp_avg'].dtype == torch.float32
    assert torch.equal(before.bfloat16(), model.head.weight.bfloat16())


def test_saved_adapter_retains_sub_bf16_master_values(tmp_path):
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import LlamaConfig, LlamaForCausalLM

    from nimloth.training.sft.stage1.checkpoint import load_lora_adapter_state
    from nimloth.training.sft.stage1.fsdp import save_full_pretrained

    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, tie_word_embeddings=True)
    def build():
        model = get_peft_model(LlamaForCausalLM(config).bfloat16(),
                              LoraConfig(task_type='CAUSAL_LM', r=2, target_modules=['q_proj'],
                                         modules_to_save=['embed_tokens', 'lm_head']))
        prepare_embedding_masters(model, 'float32')
        return model
    model = build()
    with torch.no_grad():
        model.get_output_embeddings().weight.fill_(.020001)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    save_full_pretrained(model, tmp_path, state)
    saved = load_file(str(tmp_path / 'adapter_model.safetensors'))
    heads = [value for key, value in saved.items() if 'lm_head' in key]
    assert heads and all(value.dtype == torch.float32 for value in heads)
    restored = build()
    load_lora_adapter_state(restored, tmp_path)
    torch.testing.assert_close(restored.get_output_embeddings().weight,
                               model.get_output_embeddings().weight, rtol=0, atol=0)
    assert float(restored.get_output_embeddings().weight[0, 0]) != float(torch.tensor(.020001).bfloat16())


def test_generation_setup_failure_restores_already_patched_module(monkeypatch):
    model = Model()
    prepare_embedding_masters(model, 'float32')
    original_to = torch.Tensor.to
    conversions = 0

    def fail_second(tensor, *args, **kwargs):
        nonlocal conversions
        if args == (torch.bfloat16,):
            conversions += 1
            if conversions == 2:
                assert 'forward' in model.embed.__dict__
                raise RuntimeError('simulated cached allocation failure')
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', fail_second)
    with pytest.raises(RuntimeError, match='cached allocation failure'), _bf16_master_forward(model):
        pytest.fail('setup should not finish')
    assert 'forward' not in model.embed.__dict__
    assert 'forward' not in model.head.__dict__
