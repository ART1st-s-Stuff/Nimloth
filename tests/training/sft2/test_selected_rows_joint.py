"""Selected-row scope and exact optimizer continuation through backbone export."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nimloth.backbone.qwen25vl.model import Qwen25VLBackbone
from nimloth.backbone.selected_token_rows import (
    install_full_language_selected_rows,
    materialize_selected_state_dict,
    restore_selected_rows,
    selected_rows_state,
)
from nimloth.training.sft.stage3.trainer import _build_optimizer


class TinyDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(24, 4, dtype=torch.bfloat16)
        self.head = nn.Linear(4, 24, bias=False, dtype=torch.bfloat16)
        self.language = nn.Linear(4, 4)
        self.visual = nn.Linear(4, 4)
        self.config = SimpleNamespace()

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    def forward(self):
        hidden = self.language(self.embed(torch.tensor([[12, 13, 0, 20]])).float())
        hidden = hidden + self.visual(torch.ones_like(hidden))
        return self.head(hidden.bfloat16()).float().square().mean()

    def save_pretrained(self, path, *, state_dict, **kwargs):
        path.mkdir(exist_ok=True)
        torch.save(state_dict, path / "dense.pt")


def build(model):
    wm = SimpleNamespace(state_proj=nn.Linear(4, 4), value_head=nn.Linear(4, 1),
                         wm_predictor=nn.Linear(4, 4), outcome_head=None)
    args = SimpleNamespace(query_tune="selected_rows", query_lr=1e-4,
                           protocol_lr=2e-5, lr_qwen_start=2e-6,
                           state_proj_lr=8e-5, value_head_lr=1e-4,
                           wm_predictor_lr=3e-4, weight_decay=0.01)
    agent = SimpleNamespace(backbone=SimpleNamespace(model=model), wm=wm)
    return _build_optimizer(args, agent=agent, query_adapter=None, train_wm_predictor=True)


def test_joint_groups_frozen_rows_and_exact_export_resume(tmp_path):
    torch.manual_seed(42)
    model = TinyDense()
    install_full_language_selected_rows(model, [12, 13], list(range(10)))
    optimizer = build(model)
    groups = {g["name"]: g for g in optimizer.param_groups}
    assert groups["qwen"]["lr"] == 2e-6
    assert {id(p) for p in groups["qwen"]["params"]} == {
        id(p) for module in (model.language, model.visual) for p in module.parameters()
    }
    assert groups["selected_query_rows"]["lr"] == 1e-4
    assert groups["selected_protocol_rows"]["lr"] == 2e-5
    params = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(params) == len(set(params))
    frozen = (model.embed.weight.clone(), model.head.weight.clone())
    model().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert torch.equal(model.embed.weight, frozen[0])
    assert torch.equal(model.head.weight, frozen[1])
    assert model.embed.nimloth_query_rows.dtype == torch.float32
    backbone = Qwen25VLBackbone(model, token_id_map={}, device=torch.device("cpu"),
                               latent_token_count=2, lora=False, vision_tune="full")
    backbone.save_pretrained(tmp_path)
    saved = torch.load(tmp_path / "dense.pt", weights_only=True)
    assert not any("nimloth_" in k for k in saved)
    restored = TinyDense()
    restored.load_state_dict(saved)
    install_full_language_selected_rows(restored, [12, 13], list(range(10)))
    rows = torch.load(tmp_path / "selected_token_rows.pt", weights_only=True)
    restore_selected_rows(restored, rows)
    for k, v in selected_rows_state(restored).items():
        assert torch.equal(v, selected_rows_state(model)[k])
    resumed_optimizer = build(restored)
    resumed_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    # A second real optimizer update must exactly match uninterrupted execution.
    for current, opt in ((model, optimizer), (restored, resumed_optimizer)):
        current().backward()
        opt.step()
    # The inactive dense backing rows are materialized on export; compare the
    # effective weights and exact masters, not those shadowed backing values.
    exported = materialize_selected_state_dict(model.state_dict())
    resumed = materialize_selected_state_dict(restored.state_dict())
    for key, value in exported.items():
        assert torch.equal(value, resumed[key]), key
    for key, value in selected_rows_state(model).items():
        assert torch.equal(value, selected_rows_state(restored)[key]), key


def test_resume_rejects_changed_token_identity_and_missing_masters():
    model = TinyDense()
    install_full_language_selected_rows(model, [12, 13], list(range(10)))
    rows = selected_rows_state(model)
    with pytest.raises(ValueError, match="keys"):
        restore_selected_rows(model, {})
    rows["embed.nimloth_query_ids"] = torch.tensor([13, 12])
    with pytest.raises(ValueError, match="token IDs"):
        restore_selected_rows(model, rows)


@pytest.mark.parametrize("keep_sidecar", [True, False])
def test_factory_resume_loads_fp32_before_restoring_exact_rows(tmp_path, monkeypatch, keep_sidecar):
    from nimloth.backbone.qwen25vl import factory
    from nimloth.latent import LatentActionTokens, latent_state_tokens

    model = TinyDense().float()
    install_full_language_selected_rows(model, [12, 13], list(range(10)))
    with torch.no_grad():
        model.language.weight.fill_(1.0000123)
        model.embed.nimloth_query_rows.fill_(1.0000234)
    Qwen25VLBackbone(model, token_id_map={}, device=torch.device("cpu"),
                    latent_token_count=2, lora=False, vision_tune="full").save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    torch.save({"step": 1}, tmp_path / "training_state.pt")
    if not keep_sidecar:
        (tmp_path / "selected_token_rows.pt").unlink()
    calls = []

    def from_pretrained(path, **kwargs):
        calls.append(kwargs["torch_dtype"])
        instance = TinyDense().to(kwargs["torch_dtype"])
        if path == tmp_path:
            instance.load_state_dict(torch.load(tmp_path / "dense.pt", weights_only=True))
        return instance

    tokens = LatentActionTokens()
    mapping = dict(zip((*tokens.action_tokens, tokens.action_start, tokens.action_end), range(10)))
    mapping.update(dict(zip(latent_state_tokens(2), (12, 13))))
    monkeypatch.setattr(factory, "Qwen2_5_VLForConditionalGeneration", SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setattr(factory, "load_qwen_processor", lambda *a, **kw: SimpleNamespace(
        processor=object(), token_id_map=mapping, added_special_token_count=0))
    monkeypatch.setattr(factory, "_configure_shape", lambda *a, **kw: None)
    monkeypatch.setattr(factory, "configure_qwen_tuning", lambda model, args: model)
    args = SimpleNamespace(model="initial-model", query_tune="selected_rows", llm_tune="full",
                           vision_tune="full", max_pixels=16, attn_implementation="eager", resume=True)
    if not keep_sidecar:
        with pytest.raises(ValueError, match="exact FP32 token-row state"):
            factory.load_backbone(args, device=torch.device("cpu"), latent_token_count=2,
                                  resume_dir=tmp_path, resume_state_path=tmp_path / "training_state.pt")
        return
    loaded = factory.load_backbone(args, device=torch.device("cpu"), latent_token_count=2,
                                   resume_dir=tmp_path, resume_state_path=tmp_path / "training_state.pt")
    assert calls == [torch.float32, torch.float32]
    assert loaded.query_adapter is None
    assert torch.equal(loaded.backbone.model.language.weight, model.language.weight)
    assert torch.equal(loaded.backbone.model.embed.nimloth_query_rows, model.embed.nimloth_query_rows)
    assert not loaded.backbone.model.embed.weight.requires_grad
    assert not loaded.backbone.model.head.weight.requires_grad
