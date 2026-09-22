"""Check FSDP ownership without claiming multi-rank GPU execution."""
from types import SimpleNamespace

import torch
from torch import nn

from nimloth.training.sft.stage1 import fsdp
from nimloth.training.sft.stage2.full_tuning import prepare_full_language


def test_full_visual_residuals_have_separate_fsdp_owner(monkeypatch):
    language = nn.Module()
    language.visual = nn.Sequential(nn.Conv3d(3, 4, 1), nn.LayerNorm(4), nn.Linear(4, 4))
    language.embed_tokens = nn.Embedding(8, 4)
    language.lm_head = nn.Linear(4, 8)
    language.norm = nn.LayerNorm(4)
    language.config = SimpleNamespace()
    language.get_input_embeddings = lambda: language.embed_tokens
    language.get_output_embeddings = lambda: language.lm_head
    model = nn.Module()
    model.language_model = language
    model.projector = nn.Linear(4, 4)
    model.config = language.config
    prepare_full_language(model)
    captured = {}
    monkeypatch.setattr(fsdp, "CustomPolicy", lambda predicate: predicate)
    def fake_fsdp(module, **kwargs):
        captured.update(kwargs)
        return module
    monkeypatch.setattr(fsdp, "FSDP", fake_fsdp)
    fsdp.wrap_fsdp(model, torch.device("cpu"))
    assert captured["mixed_precision"].buffer_dtype is None
    predicate = captured["auto_wrap_policy"]
    assert predicate(language.visual)
    for module in language.visual.modules():
        settings = predicate(module)
        if settings:
            precision = settings["mixed_precision"]
            assert precision.param_dtype is None
            assert precision.buffer_dtype is None
            assert not precision.cast_forward_inputs
            assert not precision.cast_root_forward_inputs
    for module in (language.embed_tokens, language.lm_head, model.projector):
        precision = predicate(module)["mixed_precision"]
        assert precision.param_dtype == torch.bfloat16
        assert precision.reduce_dtype == torch.float32
        assert precision.cast_forward_inputs
    targets = {m for m in model.modules() if predicate(m)}
    # Assign each direct parameter to its closest wrapping ancestor.
    dtypes = {}
    def visit(module, owner):
        owner = module if module in targets else owner
        dtypes.setdefault(owner, set()).update(p.dtype for p in module.parameters(recurse=False))
        for child in module.children():
            visit(child, owner)
    visit(model, model)
    assert all(len(values) <= 1 for values in dtypes.values())
    assert dtypes[language.visual] == {torch.bfloat16}
    assert dtypes[model] == {torch.float32}
