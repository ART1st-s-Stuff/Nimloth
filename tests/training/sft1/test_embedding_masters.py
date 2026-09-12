import pytest
import torch

from nimloth.training.sft.stage1.fsdp import (
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


def test_export_reload_restores_exact_master_values(tmp_path):
    from safetensors.torch import save_file
    from types import SimpleNamespace
    from nimloth.training.sft.stage1.fsdp import restore_exported_embedding_masters
    model = Model()
    model.config = SimpleNamespace(nimloth_embedding_master_dtype="float32")
    expected = torch.full((8, 4), .020001, dtype=torch.float32)
    save_file({"embed.weight": expected, "head.weight": expected.clone()}, tmp_path / "model.safetensors")
    restore_exported_embedding_masters(model, tmp_path)
    assert torch.equal(model.embed.weight, expected)
    assert torch.equal(model.head.weight, expected)
    assert model.body.weight.dtype == torch.bfloat16


def test_fsdp_master_policy_preserves_optimizer_precision(monkeypatch):
    from nimloth.training.sft.stage1 import fsdp
    model = Model()
    prepare_embedding_masters(model, "float32")
    captured = {}
    monkeypatch.setattr(fsdp, "FSDP", lambda model, **kwargs: captured.update(kwargs))
    fsdp.wrap_fsdp(model, torch.device("cpu"))
    policy = captured["auto_wrap_policy"]
    policies = policy._run_policy(model, set(), {})
    mixed = policies[model.embed]["mixed_precision"]
    assert mixed.param_dtype == torch.bfloat16
    assert mixed.reduce_dtype == torch.float32
    assert mixed.keep_low_precision_grads is False
    assert "mixed_precision" not in policies[model.body]


def test_fp32_saved_masters_are_verified_before_bf16_export(tmp_path, monkeypatch):
    import hashlib
    from nimloth.training.sft.stage1.fsdp import restore_exported_embedding_masters
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from nimloth.training.sft.stage1 import checkpoint_export as export
    from nimloth.training.sft.stage1.fsdp import save_full_pretrained

    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        tie_word_embeddings=True)
    base = LlamaForCausalLM(config).to(torch.bfloat16)
    base_dir, adapter_dir, out = tmp_path / 'base', tmp_path / 'adapter', tmp_path / 'out'
    base.save_pretrained(base_dir)
    model = get_peft_model(base.float(), LoraConfig(task_type='CAUSAL_LM', r=2,
        target_modules=['q_proj', 'v_proj'], modules_to_save=['embed_tokens', 'lm_head']))
    with torch.no_grad():
        model.get_input_embeddings().weight.fill_(0.020001)
        model.get_output_embeddings().weight.fill_(0.030001)
    save_full_pretrained(model, adapter_dir, model.state_dict())
    saved_path = adapter_dir / 'adapter_model.safetensors'
    before = hashlib.sha256(saved_path.read_bytes()).hexdigest()
    assert export.has_fp32_embedding_masters(adapter_dir)

    class Processor:
        tokenizer = tuple(range(32))

        def save_pretrained(self, path):
            pass

    monkeypatch.setattr(export, 'Qwen2_5_VLForConditionalGeneration', LlamaForCausalLM)
    export.merge_checkpoint(base_dir, adapter_dir, out, processor=Processor(), dtype=torch.bfloat16)
    reloaded = LlamaForCausalLM.from_pretrained(out, torch_dtype=torch.bfloat16)
    assert reloaded.config.nimloth_embedding_master_dtype == 'float32'
    restore_exported_embedding_masters(reloaded, out)
    assert torch.equal(reloaded.get_input_embeddings().weight, torch.full((32, 16), 0.020001, dtype=torch.float32))
    assert torch.equal(reloaded.get_output_embeddings().weight, torch.full((32, 16), 0.030001, dtype=torch.float32))
    assert hashlib.sha256(saved_path.read_bytes()).hexdigest() == before
