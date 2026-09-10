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


def test_full_export_uses_real_meta_topology_without_live_state_dict(tmp_path, monkeypatch):
    import json
    import random

    import numpy as np
    from safetensors.torch import load_file

    from nimloth.training.sft.stage1.fsdp import save_full_pretrained

    base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, tie_word_embeddings=True)).to(dtype=torch.bfloat16)
    original_config = tmp_path / "base"
    base.config.save_pretrained(original_config)
    base.config._name_or_path = str(original_config)
    base.name_or_path = str(original_config)
    base.resize_token_embeddings(40, mean_resizing=False)
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=2,
        target_modules=["q_proj", "v_proj"], modules_to_save=["embed_tokens", "lm_head"]))
    full = {key: value.clone() for key, value in model.state_dict().items()}
    reference = tmp_path / "reference"
    model.save_pretrained(reference, safe_serialization=True, state_dict=full)

    def forbidden(*args, **kwargs):
        raise AssertionError("rank-zero export touched live model state_dict")

    # These are precisely the hidden PEFT calls which deadlocked nested FSDP.
    monkeypatch.setattr(model.get_input_embeddings().modules_to_save.default, "state_dict", forbidden)
    monkeypatch.setattr(model.get_output_embeddings().modules_to_save.default, "state_dict", forbidden)
    destination = tmp_path / "actual"
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    save_full_pretrained(model, destination, full)
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    expected = load_file(str(reference / "adapter_model.safetensors"))
    actual = load_file(str(destination / "adapter_model.safetensors"))
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    actual_config = json.loads((destination / "adapter_config.json").read_text())
    expected_config = json.loads((reference / "adapter_config.json").read_text())
    # PEFT stores target_modules as a set; JSON list ordering is not semantic.
    actual_config["target_modules"] = sorted(actual_config["target_modules"])
    expected_config["target_modules"] = sorted(expected_config["target_modules"])
    assert actual_config == expected_config
