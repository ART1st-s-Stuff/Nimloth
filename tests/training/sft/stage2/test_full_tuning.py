from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nimloth.training.sft.stage1.cli import parse_args
from nimloth.training.sft.stage1.trainer import build_optimizer
from nimloth.training.sft.stage2.full_tuning import prepare_full_language


class DenseLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.layers = nn.Sequential(nn.Linear(4, 4))
        self.visual = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.config = SimpleNamespace()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_dense_scope_optimizer_update_and_restore():
    model = nn.Module()
    model.language_model = DenseLanguage().bfloat16()
    model.projector = nn.Linear(4, 3).bfloat16()
    prepare_full_language(model)
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    assert set(frozen) == {"language_model.visual.0.weight", "language_model.visual.0.bias",
                           "language_model.visual.1.weight", "language_model.visual.1.bias"}
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    optimizer = build_optimizer(model, 2e-5, 2e-5, 0.01, 2e-5)
    members = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(members) == len(set(map(id, members)))
    assert set(map(id, members)) == {id(p) for p in model.parameters() if p.requires_grad}
    before = {n: p.clone() for n, p in model.named_parameters()}
    sum(p.sum() for p in members).backward()
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]) == (not parameter.requires_grad)
    restored = build_optimizer(model, 2e-5, 2e-5, 0.01, 2e-5)
    restored.load_state_dict(optimizer.state_dict())
    assert all(g["lr"] == 2e-5 for g in restored.param_groups)
    assert len(restored.state) == len(members)


def test_full_mode_cli_and_default():
    base = ["--model", "/m", "--train-jsonl", "/t", "--val-jsonl", "/v",
            "--output-dir", "/o", "--dino-cache-root", "/d"]
    default, _ = parse_args(base, stage="query")
    assert default.lora and default.tuning_mode == "selected_lora"
    full, _ = parse_args(base + ["--tuning-mode", "full_language"], stage="query")
    assert not full.lora and full.query_token_lr is None and full.protocol_token_lr is None
    full_query, _ = parse_args(
        base + ["--tuning-mode", "full_language", "--query-token-lr", "1e-4",
                "--protocol-token-lr", "2e-5"],
        stage="query",
    )
    assert full_query.query_token_lr == 1e-4
    assert full_query.protocol_token_lr == 2e-5
    with pytest.raises(ValueError, match="both query and protocol"):
        parse_args(
            base + ["--tuning-mode", "full_language", "--query-token-lr", "1e-4"],
            stage="query",
        )
    with pytest.raises(ValueError, match="incompatible"):
        parse_args(base + ["--tuning-mode", "full_language", "--lora"], stage="query")
    with pytest.raises(ValueError, match="FSDP"):
        parse_args(base + ["--tuning-mode", "full_language", "--distributed-strategy", "ddp"], stage="query")


def test_full_language_updates_only_query_action_and_boundary_rows():
    from nimloth.training.sft.stage2.selected_token_rows import (
        materialize_selected_state_dict,
    )

    model = nn.Module()
    model.language_model = DenseLanguage().float()
    model.projector = nn.Linear(4, 3).float()
    prepare_full_language(
        model,
        query_ids=(12, 13),
        protocol_ids=tuple(range(8)) + (10, 11),
    )
    embedding = model.language_model.embed_tokens
    head = model.language_model.lm_head
    dense_before = embedding.weight.detach().clone()
    query_before = embedding.nimloth_query_rows.detach().clone()
    optimizer = build_optimizer(
        model, 2e-5, 2e-5, 0.0, 2e-5,
        query_token_lr=1e-4, protocol_token_lr=2e-5,
    )
    assert [group["lr"] for group in optimizer.param_groups] == [2e-5] * 2 + [1e-4] + [2e-5] * 2
    hidden = embedding(torch.tensor([[1, 12, 13]]))
    (head(hidden).sum() + model.projector(hidden).sum()).backward()
    optimizer.step()
    assert torch.equal(embedding.weight, dense_before)
    assert not torch.equal(embedding.nimloth_query_rows, query_before)
    exported = materialize_selected_state_dict(model.state_dict())
    assert not any("nimloth_" in key for key in exported)
    torch.testing.assert_close(
        exported["language_model.embed_tokens.weight"][[12, 13]],
        embedding.nimloth_query_rows,
    )
    torch.testing.assert_close(
        exported["language_model.embed_tokens.weight"][list(range(8)) + [10, 11]],
        embedding.nimloth_protocol_rows,
    )
    frozen_ids = [8, 9, 14, 15]
    assert torch.equal(
        exported["language_model.embed_tokens.weight"][frozen_ids],
        dense_before[frozen_ids],
    )


def test_dense_query_export_preserves_fp32_weights(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    from nimloth.training.sft.stage1.fsdp import save_full_pretrained
    from nimloth.training.sft.stage2.config import QueryAlignmentConfig
    from nimloth.training.sft.stage2.model import QueryAlignmentModel
    from nimloth.wm.grid import SharedSlotProjector

    language = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, tie_word_embeddings=False))
    language.config.nimloth_tuning_mode = "full_language"
    language.config.nimloth_embedding_master_dtype = "float32"
    objective = QueryAlignmentConfig(projector_hidden_dim=8)
    model = QueryAlignmentModel(language, SharedSlotProjector(16, 1024, hidden_dim=8,
                                grid_tokens=16), list(range(16)), objective)
    full = {key: value.clone() for key, value in model.state_dict().items()}
    save_full_pretrained(model, tmp_path, full)
    assert not (tmp_path / "adapter_config.json").exists()
    restored = LlamaForCausalLM.from_pretrained(tmp_path, torch_dtype=torch.float32)
    assert restored.config.nimloth_tuning_mode == "full_language"
    for name, value in restored.state_dict().items():
        assert torch.equal(value, full["language_model." + name])
    from nimloth.latent import latent_state_tokens

    token_ids = dict(zip(latent_state_tokens(16), range(16)))
    tokenizer = SimpleNamespace(convert_tokens_to_ids=token_ids.__getitem__)
    query = QueryAlignmentModel.build(restored, tokenizer, objective)
    assert next(query.projector.parameters()).dtype == torch.float32
    query.restore_projector(tmp_path)
    for name, value in query.projector.state_dict().items():
        assert torch.equal(value, full["projector." + name])


def test_full_scope_preserves_real_qwen_rotary_buffers():
    import inspect
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    text = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                tie_word_embeddings=False,
                rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]})
    text_kwargs = ({"text_config": text}
                   if "text_config" in inspect.signature(Qwen2_5_VLConfig).parameters
                   else text)
    config = Qwen2_5_VLConfig(**text_kwargs, vision_config=dict(
        depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16))
    model = nn.Module()
    model.language_model = Qwen2_5_VLForConditionalGeneration(config)
    model.projector = nn.Linear(16, 3)
    visual = model.language_model.visual
    buffers = {name: tensor.clone() for name, tensor in visual.named_buffers()}
    assert buffers and any("inv_freq" in name for name in buffers)
    prepare_full_language(model)
    for name, tensor in visual.named_buffers():
        assert tensor.dtype == buffers[name].dtype
        assert torch.equal(tensor, buffers[name])
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in visual.parameters())
    assert visual.rotary_pos_emb(4).dtype == torch.float32
