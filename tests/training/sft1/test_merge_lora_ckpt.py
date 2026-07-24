"""Regression tests for the SFT1 LoRA merge export."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from experiments.training.sft1.merge_lora_ckpt import (
    finalize_merged_vocab,
    restore_saved_untied_embeddings,
)


class FakeMergedModel:
    def __init__(self, vocab_size: int, *, tied: bool = False) -> None:
        self.input_embeddings = torch.nn.Embedding(vocab_size, 4)
        self.output_embeddings = torch.nn.Linear(4, vocab_size, bias=False)
        if tied:
            self.output_embeddings.weight = self.input_embeddings.weight
        self.config = SimpleNamespace(
            vocab_size=0,
            tie_word_embeddings=True,
            text_config=SimpleNamespace(vocab_size=0, tie_word_embeddings=True),
        )
        self.generation_config = SimpleNamespace(vocab_size=0)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.input_embeddings

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.output_embeddings

    def set_output_embeddings(self, module: torch.nn.Module) -> None:
        self.output_embeddings = module

    def resize_token_embeddings(self, _vocab_size: int) -> None:
        raise AssertionError("merged model must not be resized")


def test_finalize_merged_vocab_preserves_independent_lm_head() -> None:
    model = FakeMergedModel(vocab_size=7)
    input_before = model.input_embeddings.weight.detach().clone()
    output_before = model.output_embeddings.weight.detach().clone()

    finalize_merged_vocab(model, vocab_size=7)

    assert torch.equal(model.input_embeddings.weight, input_before)
    assert torch.equal(model.output_embeddings.weight, output_before)
    assert model.input_embeddings.weight.data_ptr() != model.output_embeddings.weight.data_ptr()
    assert model.config.vocab_size == 7
    assert model.config.text_config.vocab_size == 7
    assert model.generation_config.vocab_size == 7
    assert model.config.tie_word_embeddings is False
    assert model.config.text_config.tie_word_embeddings is False


def test_finalize_merged_vocab_rejects_tied_lm_head() -> None:
    model = FakeMergedModel(vocab_size=7, tied=True)

    with pytest.raises(RuntimeError, match="shares storage"):
        finalize_merged_vocab(model, vocab_size=7)


def test_restore_saved_embedding_layers_reconstructs_untied_head(tmp_path) -> None:
    model = FakeMergedModel(vocab_size=7, tied=True)
    saved_input = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    saved_output = saved_input.flip(0).clone()
    save_file(
        {
            "base_model.model.model.language_model.embed_tokens.weight": saved_input,
            "base_model.model.lm_head.weight": saved_output,
        },
        tmp_path / "adapter_model.safetensors",
    )

    keys = restore_saved_untied_embeddings(model, tmp_path)
    finalize_merged_vocab(model, vocab_size=7)

    assert keys == (
        "base_model.model.model.language_model.embed_tokens.weight",
        "base_model.model.lm_head.weight",
    )
    assert torch.equal(model.input_embeddings.weight, saved_input)
    assert torch.equal(model.output_embeddings.weight, saved_output)
    assert model.input_embeddings.weight.data_ptr() != model.output_embeddings.weight.data_ptr()
