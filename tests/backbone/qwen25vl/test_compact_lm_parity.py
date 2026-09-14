"""Compact LM projection preserves the real Qwen selected-row objective."""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.qwen25vl.latent import _capture_last_hidden, extract_qwen_latents
from nimloth.backbone.selected_token_rows import install_full_language_selected_rows
from nimloth.latent.extraction import LatentActionTokens


def _model():
    config = Qwen2_5_VLConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        tie_word_embeddings=False,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
                           num_heads=2, out_hidden_size=16),
        image_token_id=29, video_token_id=30, vision_start_token_id=31,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2_5_VLForConditionalGeneration(config)
    install_full_language_selected_rows(model, [10], list(range(11, 21)))
    return model


@pytest.mark.parametrize("bf16_forward", [False, True])
@pytest.mark.parametrize("weights", [[1., 1.], [1., 0.], [0., 0.]])
def test_selected_row_real_qwen_compact_loss_and_all_gradients(weights, bf16_forward):
    torch.manual_seed(71)
    compact = _model()
    dense = copy.deepcopy(compact)
    ids = torch.tensor([[1, 10, 2, 3, 4], [4, 10, 5, 6, 7]])
    labels = torch.tensor([[-100, -100, 2, 3, 4], [-100, -100, -100, 6, -100]])
    enc = {"input_ids": ids, "labels": labels}
    row_weights = torch.tensor(weights)
    head_calls = []
    handle = compact.get_output_embeddings().register_forward_hook(
        lambda module, args, output: head_calls.append(tuple(output.shape)))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16_forward):
        states, loss = extract_qwen_latents(
            compact, enc, {LatentActionTokens().latent_state: 10}, torch.device("cpu"),
            lm_row_weights=row_weights,
        )
    handle.remove()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16_forward):
        dense_hidden, dense_output = _capture_last_hidden(dense, {"input_ids": ids}, full_logits=True)
    row_losses = []
    for row in range(2):
        valid = labels[row, 1:] != -100
        row_losses.append(F.cross_entropy(dense_output.logits[row, :-1][valid].float(),
                                         labels[row, 1:][valid]))
    expected_loss = (torch.stack(row_losses) * row_weights).sum() / row_weights.sum().clamp_min(1)
    torch.testing.assert_close(states, dense_hidden[:, 1])
    torch.testing.assert_close(loss, expected_loss)
    (loss + states.square().mean()).backward()
    (expected_loss + dense_hidden[:, 1].square().mean()).backward()
    assert len(head_calls) == 1
    assert head_calls[0][-1] == 32
    # The result may retain failed-row positions or use one graph-connected dummy.
    assert torch.tensor(head_calls[0][:-1]).prod() <= 4
    assert enc["labels"] is labels and enc["input_ids"] is ids
    for (name, parameter), (other_name, other) in zip(compact.named_parameters(), dense.named_parameters()):
        assert name == other_name
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-7, msg=name)
    for name, parameter in compact.get_output_embeddings().named_parameters():
        if "rows" in name:
            assert parameter.grad is not None
