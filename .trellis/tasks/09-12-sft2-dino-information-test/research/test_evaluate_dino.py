import torch
import pytest
from evaluate_dino import recovery_metrics


def test_exact_recovery_beats_mean_and_cross_trajectory_shuffle():
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(9, 4, 8, generator=generator)
    ids = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3])
    result = recovery_metrics(target, target, ids, torch.zeros(4, 8))
    assert result["metrics"]["state_mse"]["mean"] == 0
    assert result["metrics"]["relative_mse_reduction_vs_train_mean"]["mean"] == 1
    assert result["metrics"]["shuffled_state_mse"]["mean"] > 0
    assert torch.all(ids != ids[result["shuffle_source_indices"]])
    assert result == recovery_metrics(target, target, ids, torch.zeros(4, 8))


def test_macro_weighting_does_not_reward_long_trajectory():
    target = torch.ones(6, 2, 3)
    prediction = target.clone()
    prediction[:3] = 3
    ids = torch.tensor([0, 0, 0, 1, 1, 2])
    result = recovery_metrics(prediction, target, ids, torch.zeros(2, 3))
    assert result["metrics"]["state_mse"]["mean"] == pytest.approx(4 / 3)


def test_rejects_impossible_shuffle():
    target = torch.ones(4, 2, 3)
    with pytest.raises(ValueError, match="dominates"):
        recovery_metrics(target, target, torch.tensor([0, 0, 0, 1]), torch.zeros(2, 3))


def test_evaluation_loads_fp32_masters_before_forward_cast(tmp_path):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    from nimloth.training.sft.stage1.fsdp import prepare_embedding_masters, save_full_pretrained
    from evaluate_dino import load_evaluation_adapter

    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, tie_word_embeddings=False)

    def build():
        return get_peft_model(LlamaForCausalLM(config).bfloat16(),
                              LoraConfig(task_type="CAUSAL_LM", r=2, target_modules=["q_proj"],
                                         modules_to_save=["embed_tokens", "lm_head"]))

    source = build()
    prepare_embedding_masters(source, "float32")
    with torch.no_grad():
        source.get_input_embeddings().weight.fill_(.020001)
        source.get_output_embeddings().weight.fill_(.030001)
    save_full_pretrained(source, tmp_path, source.state_dict())
    saved = (tmp_path / "adapter_model.safetensors").read_bytes()
    restored = build()
    load_evaluation_adapter(restored, tmp_path)
    assert restored.get_input_embeddings().weight.dtype == torch.bfloat16
    assert restored.get_output_embeddings().weight.dtype == torch.bfloat16
    assert torch.equal(restored.get_output_embeddings().weight,
                       source.get_output_embeddings().weight.bfloat16())
    assert all(p.dtype == torch.float32 for n, p in restored.named_parameters() if "lora_" in n)
    with torch.no_grad():
        assert torch.isfinite(restored(input_ids=torch.tensor([[1, 2]])).logits).all()
    assert saved == (tmp_path / "adapter_model.safetensors").read_bytes()
