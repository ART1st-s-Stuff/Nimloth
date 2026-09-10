"""CPU ownership policy checks; actual sharding requires the explicit GPU probe."""
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from nimloth.training.sft.stage1.fsdp import _auto_wrap_targets


def test_frozen_tied_originals_keep_common_owner_and_trainable_copies_split():
    base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, tie_word_embeddings=True)).to(dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=2,
        target_modules=["q_proj", "v_proj"], modules_to_save=["embed_tokens", "lm_head"]))
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    assert embedding.original_module.weight is head.original_module.weight
    assert not embedding.original_module.weight.requires_grad
    assert embedding.modules_to_save.default.weight is not head.modules_to_save.default.weight
    targets = _auto_wrap_targets(model)
    assert embedding.original_module not in targets
    assert head.original_module not in targets
    assert embedding.modules_to_save.default in targets
    assert head.modules_to_save.default in targets
    shared_id = id(embedding.original_module.weight)
    assert all(shared_id not in {id(p) for p in module.parameters()} for module in targets)
    assert any(module.__class__.__name__ == "LlamaDecoderLayer" for module in targets)
