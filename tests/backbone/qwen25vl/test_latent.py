from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nimloth.backbone.qwen25vl.latent import (
    extract_qwen_action_boundary_hidden,
    extract_qwen_latents,
)
from nimloth.latent.extraction import LatentActionTokens, latent_state_tokens


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(language_model=SimpleNamespace(norm=nn.LayerNorm(4)))
        self.output_hidden_states_seen: bool | None = None
        self.logits_to_keep_seen = None
        self.lm_head = nn.Linear(4, 8, bias=False)
        nn.init.ones_(self.lm_head.weight)
        self.logit_shape = None

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, output_hidden_states: bool, return_dict: bool, **kwargs):
        assert return_dict is True
        self.output_hidden_states_seen = output_hidden_states
        self.logits_to_keep_seen = kwargs.get("logits_to_keep")
        batch, seq_len = input_ids.shape
        hidden = torch.arange(batch * seq_len * 4, dtype=torch.float32).reshape(batch, seq_len, 4)
        final_hidden = self.model.language_model.norm(hidden)
        logits = self.lm_head(final_hidden)
        self.logit_shape = logits.shape
        loss = logits.sum() * 0.0
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=None)


def test_extract_qwen_latents_uses_final_norm_hook_without_all_hidden_states() -> None:
    tokens = LatentActionTokens()
    token_id_map = {token: i + 10 for i, token in enumerate(tokens.all_special_tokens)}
    input_ids = torch.tensor([[1, token_id_map[tokens.latent_state], 2]])
    model = _FakeQwen()

    latent, loss = extract_qwen_latents(
        model,
        {"input_ids": input_ids},
        token_id_map,
        torch.device("cpu"),
    )

    assert model.output_hidden_states_seen is False
    assert model.logits_to_keep_seen is None
    assert model.logit_shape[1] == 1
    assert loss is not None
    expected = model.model.language_model.norm(
        torch.arange(1 * 3 * 4, dtype=torch.float32).reshape(1, 3, 4)
    )[0, 1]
    torch.testing.assert_close(latent, expected.unsqueeze(0))


def test_extract_qwen_latents_can_return_multi_token_block() -> None:
    tokens = LatentActionTokens()
    all_tokens = (*latent_state_tokens(3), tokens.action_start, tokens.action_end, *tokens.action_tokens)
    token_id_map = {token: i + 10 for i, token in enumerate(all_tokens)}
    input_ids = torch.tensor([
        [
            1,
            token_id_map[tokens.latent_state],
            token_id_map["<|latent_state_1|>"],
            token_id_map["<|latent_state_2|>"],
            2,
        ]
    ])
    model = _FakeQwen()

    latent, _loss = extract_qwen_latents(
        model,
        {"input_ids": input_ids},
        token_id_map,
        torch.device("cpu"),
        latent_token_count=3,
    )

    expected_hidden = model.model.language_model.norm(
        torch.arange(1 * 5 * 4, dtype=torch.float32).reshape(1, 5, 4)
    )[0, 1:4]
    assert latent.shape == (1, 3, 4)
    torch.testing.assert_close(latent[0], expected_hidden)


def test_extract_qwen_action_boundary_hidden_uses_last_boundary_per_row() -> None:
    tokens = LatentActionTokens()
    token_id_map = {token: i + 10 for i, token in enumerate(tokens.all_special_tokens)}
    action_start = token_id_map[tokens.action_start]
    input_ids = torch.tensor(
        [
            [action_start, 1, action_start, 2],
            [3, action_start, 4, 5],
        ]
    )
    model = _FakeQwen()

    boundary = extract_qwen_action_boundary_hidden(
        model,
        {"input_ids": input_ids},
        token_id_map,
        torch.device("cpu"),
    )

    expected_hidden = model.model.language_model.norm(
        torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    )
    assert model.logits_to_keep_seen is None
    assert model.logit_shape[1] == 1
    torch.testing.assert_close(boundary[0], expected_hidden[0, 2])
    torch.testing.assert_close(boundary[1], expected_hidden[1, 1])


def test_extract_qwen_action_boundary_hidden_rejects_labels() -> None:
    tokens = LatentActionTokens()
    token_id_map = {token: i + 10 for i, token in enumerate(tokens.all_special_tokens)}
    input_ids = torch.tensor([[token_id_map[tokens.action_start]]])

    with pytest.raises(ValueError, match="must not include labels"):
        extract_qwen_action_boundary_hidden(
            _FakeQwen(),
            {"input_ids": input_ids, "labels": input_ids.clone()},
            token_id_map,
            torch.device("cpu"),
        )


def test_extract_qwen_latents_keeps_full_supervised_lm_loss() -> None:
    tokens = LatentActionTokens()
    token_id_map = {token: i + 10 for i, token in enumerate(tokens.all_special_tokens)}
    input_ids = torch.tensor([[1, token_id_map[tokens.latent_state], 2]])
    model = _FakeQwen()

    _latent, loss = extract_qwen_latents(
        model,
        {"input_ids": input_ids, "labels": input_ids.clone()},
        token_id_map,
        torch.device("cpu"),
    )

    assert model.logits_to_keep_seen is None
    assert loss is not None


@pytest.mark.parametrize("weights", [[1., 0.], [0., 0.], [1., 1.]])
def test_window_lm_selection_preserves_state_and_excludes_failed_rows(weights):
    class LM(_FakeQwen):
        def __init__(self):
            super().__init__()
            self.scores = nn.Parameter(torch.randn(2, 4, 8))
            self.model.language_model.norm = nn.Identity()
            self.lm_head = nn.Linear(8, 8, bias=False)
            with torch.no_grad():
                self.lm_head.weight.copy_(torch.eye(8))
        def forward(self, input_ids, **kwargs):
            hidden = self.model.language_model.norm(self.scores)
            logits = self.lm_head(hidden)
            self.logit_shape = logits.shape
            return SimpleNamespace(logits=logits, loss=None)
    model = LM()
    tokens = LatentActionTokens()
    mapping = {tokens.latent_state: 10}
    ids = torch.tensor([[1, 10, 2, 3], [4, 10, 5, 6]])
    labels = torch.tensor([[-100, -100, 2, 3], [-100, -100, -100, 6]])
    hidden, loss = extract_qwen_latents(model, {"input_ids": ids, "labels": labels},
                                      mapping, torch.device("cpu"),
                                      lm_row_weights=torch.tensor(weights))
    expected = [torch.nn.functional.cross_entropy(model.scores[0, 1:3], labels[0, 2:]),
                torch.nn.functional.cross_entropy(model.scores[1, 2:3], labels[1, 3:])]
    target = sum(v * w for v, w in zip(expected, weights)) / max(1., sum(weights))
    torch.testing.assert_close(loss, target)
    assert hidden.shape == (2, 8)
    assert model.logit_shape == ()
    loss.backward()
    for row, weight in enumerate(weights):
        assert bool(model.scores.grad[row].abs().sum() > 0) == bool(weight)


@pytest.mark.parametrize("full_logits", [False, True])
def test_legacy_forward_without_logits_keyword_preserves_hidden_and_gradients(full_logits):
    from nimloth.backbone.qwen25vl.latent import _capture_last_hidden

    class LegacyQwen(_FakeQwen):
        def forward(self, input_ids, output_hidden_states, return_dict, use_cache):
            assert use_cache is False
            return super().forward(input_ids, output_hidden_states, return_dict)

    model = LegacyQwen()
    ids = torch.tensor([[1, 2, 3, 4]])
    hidden, out = _capture_last_hidden(model, {"input_ids": ids}, full_logits=full_logits)
    assert hidden.shape == (1, 4, 4)
    assert out.logits.shape == (1, 4 if full_logits else 1, 8)
    hidden.square().sum().backward()
    assert model.model.language_model.norm.weight.grad is not None
    assert not model.lm_head._forward_pre_hooks
    assert not model.model.language_model.norm._forward_hooks


def test_projection_hook_removed_after_failed_forward():
    from nimloth.backbone.qwen25vl.latent import _capture_last_hidden

    class BrokenQwen(_FakeQwen):
        def forward(self, **kwargs):
            raise RuntimeError("forward failed")

    model = BrokenQwen()
    with pytest.raises(RuntimeError, match="forward failed"):
        _capture_last_hidden(model, {"input_ids": torch.tensor([[1, 2]])})
    assert not model.lm_head._forward_pre_hooks
    assert not model.model.language_model.norm._forward_hooks


@pytest.mark.parametrize("mode", ["target", "primary", "sigreg"])
def test_real_qwen_right_padded_feature_forward_disables_cache(mode):
    import inspect
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    from nimloth.backbone.qwen25vl.latent import _capture_last_hidden

    text = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                tie_word_embeddings=False, use_cache=True,
                rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]})
    text_kwargs = ({"text_config": text}
                   if "text_config" in inspect.signature(Qwen2_5_VLConfig).parameters
                   else text)
    config = Qwen2_5_VLConfig(**text_kwargs, vision_config=dict(
        depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16))
    config.image_token_id = 29
    config.video_token_id = 30
    config.vision_start_token_id = 31
    config._attn_implementation = "sdpa"
    model = Qwen2_5_VLForConditionalGeneration(config).train()
    ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    inputs = {"input_ids": ids, "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
              "use_cache": True}
    if mode == "primary":
        inputs["labels"] = ids.masked_fill(inputs["attention_mask"] == 0, -100)
    seen = []
    handle = model.register_forward_pre_hook(
        lambda _module, _args, kwargs: seen.append(kwargs["use_cache"]), with_kwargs=True)
    try:
        with torch.set_grad_enabled(mode != "target"):
            hidden, output = _capture_last_hidden(model, inputs)
            assert hidden.shape == (2, 4, 16)
            assert torch.isfinite(hidden).all()
            assert output.past_key_values is None
            if mode != "target":
                loss = output.loss if mode == "primary" else hidden.square().mean()
                loss.backward()
                assert model.get_input_embeddings().weight.grad is not None
    finally:
        handle.remove()
    assert seen == [False]
    assert inputs["use_cache"] is True
    assert torch.equal(inputs["attention_mask"], torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]))


def test_packed_projection_keeps_full_hidden_and_removes_hooks_on_failure():
    from nimloth.backbone.qwen25vl.latent import _capture_last_hidden
    model = _FakeQwen()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]])
    mask = torch.tensor([[False, True, False, True], [True, False, False, False]])
    inputs = {"input_ids": ids, "use_cache": True}
    hidden, output = _capture_last_hidden(model, inputs, logit_mask=mask)
    assert hidden.shape == (2, 4, 4)
    assert output.logits.shape == (1, 3, 8)
    torch.testing.assert_close(output.logits[0], model.lm_head(hidden[mask]))
    assert inputs["use_cache"] is True
    assert torch.equal(inputs["input_ids"], ids)
    assert not model.lm_head._forward_pre_hooks

    def fail(_module, _args, _output):
        raise RuntimeError("projection failed")
    handle = model.lm_head.register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="projection failed"):
            _capture_last_hidden(model, inputs, logit_mask=mask)
    finally:
        handle.remove()
    assert not model.lm_head._forward_pre_hooks
    assert not model.model.language_model.norm._forward_hooks


@pytest.mark.parametrize("weights", [[1., 0.], [0., 0.]])
def test_empty_answer_rejected_even_on_unsuccessful_row(weights):
    ids = torch.tensor([[1, 10, 2], [1, 10, 2]])
    labels = torch.tensor([[-100, -100, 2], [-100, -100, -100]])
    with pytest.raises(ValueError, match="no supervised answer tokens"):
        extract_qwen_latents(_FakeQwen(), {"input_ids": ids, "labels": labels},
                            {LatentActionTokens().latent_state: 10}, torch.device("cpu"),
                            lm_row_weights=torch.tensor(weights))
