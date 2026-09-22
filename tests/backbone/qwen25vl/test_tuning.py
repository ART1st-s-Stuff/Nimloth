from __future__ import annotations

import argparse

import pytest

from nimloth.backbone.qwen25vl.tuning import resolve_tune_modes
from nimloth.backbone.qwen25vl.vision_ema import resolve_vision_ema, vision_is_trainable


def test_resolve_tune_modes_legacy_lora() -> None:
    args = argparse.Namespace(lora=True, llm_tune="freeze", vision_tune="full")
    assert resolve_tune_modes(args) == ("lora", "freeze")


def test_vision_is_trainable() -> None:
    assert vision_is_trainable("full")
    assert vision_is_trainable("lora")
    assert not vision_is_trainable("freeze")


def test_resolve_vision_ema_defaults_on_for_full_vision() -> None:
    args = argparse.Namespace(vision_ema=None, no_vision_ema=False)
    assert resolve_vision_ema(args, "full") is True
    assert resolve_vision_ema(args, "freeze") is False


def test_resolve_vision_ema_explicit_disable() -> None:
    args = argparse.Namespace(vision_ema=None, no_vision_ema=True)
    assert resolve_vision_ema(args, "full") is False


def _tiny_qwen():
    import inspect
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    text = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                tie_word_embeddings=False,
                rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]})
    kwargs = ({"text_config": text}
              if "text_config" in inspect.signature(Qwen2_5_VLConfig).parameters else text)
    config = Qwen2_5_VLConfig(**kwargs, vision_config=dict(
        depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16))
    config.image_token_id = 29
    config.video_token_id = 30
    config.vision_start_token_id = 31
    config._attn_implementation = "sdpa"
    return Qwen2_5_VLForConditionalGeneration(config)


def _args(language, vision):
    return argparse.Namespace(llm_tune=language, vision_tune=vision, lora=False,
                              lora_r=2, lora_alpha=4, lora_dropout=0.,
                              gradient_checkpointing=False)


@pytest.mark.parametrize("language,vision", [("full", "full"), ("full", "freeze"),
                                             ("freeze", "full"), ("freeze", "freeze"),
                                             ("full", "lora"), ("lora", "full"),
                                             ("lora", "freeze")])
def test_real_qwen_tuning_regions(language, vision):
    import torch
    from nimloth.backbone.qwen25vl.tuning import configure_qwen_tuning, _parameter_regions

    model = _tiny_qwen()
    language_ids, vision_ids, dense_ids = _parameter_regions(model)
    original = {id(p): p for p in model.parameters()}
    configured = configure_qwen_tuning(model, _args(language, vision))
    assert all(original[i].requires_grad == (language == "full") for i in language_ids)
    assert all(original[i].requires_grad == (vision == "full") for i in vision_ids)
    if "lora" in (language, vision):
        adapters = [(n, p) for n, p in configured.named_parameters() if "lora_" in n]
        assert adapters and all(p.requires_grad for _, p in adapters)
        # Historical PEFT target suffixes are retained for adapter compatibility.
    if language == "full":
        configured(input_ids=torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]]),
                   use_cache=False).loss.backward()
        assert any(original[i].grad is not None and original[i].grad.abs().sum() > 0
                   for i in dense_ids)


def test_missing_language_decoder_fails_closed():
    import torch
    from nimloth.backbone.qwen25vl.tuning import configure_qwen_tuning

    model = _tiny_qwen()
    model.set_decoder(torch.nn.Module())
    with pytest.raises((ValueError, AttributeError)):
        configure_qwen_tuning(model, _args("full", "full"))


def test_nested_multimodal_decoder_ownership():
    """Exercise the newer container layout with actual tiny Qwen submodules."""
    import torch
    from nimloth.backbone.qwen25vl.tuning import configure_qwen_tuning

    base = _tiny_qwen()
    visual = getattr(base, "visual", None)
    if visual is None:
        visual = base.model.visual
    language = base.get_decoder()
    if hasattr(language, "language_model"):
        language = language.language_model

    class Nested(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = language
            self.model.visual = visual
            self.lm_head = base.get_output_embeddings()

        def get_decoder(self):
            return self.model

        def get_input_embeddings(self):
            return base.get_input_embeddings()

        def get_output_embeddings(self):
            return self.lm_head

    configured = configure_qwen_tuning(Nested(), _args("full", "freeze"))
    assert all(p.requires_grad for p in configured.model.language_model.parameters())
    assert all(not p.requires_grad for p in configured.model.visual.parameters())
    assert configured.lm_head.weight.requires_grad
