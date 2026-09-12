"""CPU precision/interface checks, not multi-rank execution evidence."""

import hashlib

import pytest
import torch

from nimloth.training.sft.stage1.cli import parse_args
from nimloth.training.sft.stage1.config import sft1_yaml_defaults


def arguments(tmp_path):
    result = []
    for name in ('model', 'train-jsonl', 'val-jsonl', 'format-eval-jsonl', 'output-dir'):
        result += ['--' + name, str(tmp_path / name)]
    return result


def test_master_precision_cli_and_yaml(tmp_path):
    common = arguments(tmp_path)
    assert parse_args(common)[0].embedding_master_dtype == 'bfloat16'
    with pytest.raises(ValueError, match='require format'):
        parse_args(common + ['--embedding-master-dtype', 'float32'])
    args, _ = parse_args(common + ['--embedding-master-dtype', 'float32', '--lora', '--distributed-strategy', 'fsdp'])
    assert args.embedding_master_dtype == 'float32'
    cfg = tmp_path / 'config.yaml'
    cfg.write_text('train:\n  embedding_master_dtype: float32\n')
    assert sft1_yaml_defaults(cfg)['embedding_master_dtype'] == 'float32'


def test_master_precision_changes_resume_identity_only_when_nondefault(tmp_path):
    from nimloth.training.sft.stage1.checkpoint import objective_identities_match
    from nimloth.training.sft.stage1.trainer import _resume_identity

    args, _ = parse_args(arguments(tmp_path))
    for path in (args.train_jsonl, args.val_jsonl, args.format_eval_jsonl):
        path.write_text('{}\n')
    args.train_cache_fingerprint = 'test'
    old = _resume_identity(args, stage='format', world=1, train_size=1)
    assert 'embedding_master_dtype' not in old
    args.embedding_master_dtype = 'float32'
    new = _resume_identity(args, stage='format', world=1, train_size=1)
    assert new['embedding_master_dtype'] == 'float32'
    assert not objective_identities_match(old, new)
    assert objective_identities_match(new, dict(new))


def test_fp32_saved_masters_are_verified_before_bf16_export(tmp_path, monkeypatch):
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
    export.merge_checkpoint(base_dir, adapter_dir, out, processor=Processor())
    reloaded = LlamaForCausalLM.from_pretrained(out, torch_dtype=torch.bfloat16)
    assert reloaded.config.nimloth_embedding_master_dtype == 'float32'
    assert torch.equal(reloaded.get_input_embeddings().weight, torch.full((32, 16), 0.020001, dtype=torch.bfloat16))
    assert torch.equal(reloaded.get_output_embeddings().weight, torch.full((32, 16), 0.030001, dtype=torch.bfloat16))
    assert hashlib.sha256(saved_path.read_bytes()).hexdigest() == before
