from __future__ import annotations

import torch
from torch import nn

from nimloth.training.common.vision_ema import VisionEncoderEMA, iter_trainable_vision_params


class _FakeVisual(nn.Linear):
    pass


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _FakeVisual(4, 4)
        self.language_model = nn.Linear(4, 4)
        for param in self.language_model.parameters():
            param.requires_grad = False
        for param in self.visual.parameters():
            param.requires_grad = True


def test_vision_ema_update_moves_shadow_toward_weights() -> None:
    model = _FakeQwen()
    ema = VisionEncoderEMA(decay=0.9)
    with torch.no_grad():
        model.visual.weight.fill_(1.0)
    ema.reset(model)
    before = ema.shadow["visual.weight"].mean().item()

    with torch.no_grad():
        model.visual.weight.fill_(2.0)
    ema.update(model)

    after = ema.shadow["visual.weight"].mean().item()
    assert before == 1.0
    assert after > before
    assert after < 2.0


def test_vision_ema_use_weights_swaps_and_restores() -> None:
    model = _FakeQwen()
    ema = VisionEncoderEMA(decay=0.9)
    ema.reset(model)
    original = model.visual.weight.detach().clone()

    with torch.no_grad():
        ema.shadow["visual.weight"].fill_(5.0)

    with ema.use_ema_weights(model):
        assert torch.allclose(model.visual.weight, torch.full_like(model.visual.weight, 5.0))
    assert torch.allclose(model.visual.weight, original)


def test_vision_ema_save_load_roundtrip(tmp_path) -> None:
    model = _FakeQwen()
    ema = VisionEncoderEMA(decay=0.95)
    ema.reset(model)
    path = tmp_path / "vision_ema.pt"
    ema.save_checkpoint(path)

    loaded = VisionEncoderEMA.load_checkpoint(path)
    assert loaded.decay == 0.95
    assert set(loaded.shadow.keys()) == set(ema.shadow.keys())
    for key in ema.shadow:
        assert torch.allclose(loaded.shadow[key], ema.shadow[key])


def test_iter_trainable_vision_params_skips_frozen_llm() -> None:
    model = _FakeQwen()
    names = [name for name, _ in iter_trainable_vision_params(model)]
    assert names == ["visual.weight", "visual.bias"]
