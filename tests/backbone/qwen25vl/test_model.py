from __future__ import annotations

from types import SimpleNamespace

import torch

from nimloth.backbone import BackboneBatch
from nimloth.backbone.qwen25vl.model import Qwen25VLBackbone


class _FakeQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.config = SimpleNamespace(
            image_token_id=99,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )


def test_chunked_forward_preserves_row_order_and_global_ce_mean(monkeypatch) -> None:
    model = _FakeQwen()
    calls: list[int] = []

    def fake_extract(qwen, enc, *_args, **_kwargs):
        calls.append(int(enc["input_ids"].shape[0]))
        hidden = enc["input_ids"][:, :1].float() * qwen.scale
        labels = enc.get("labels")
        loss = None
        if labels is not None:
            valid = labels[:, 1:][labels[:, 1:] != -100].float()
            loss = valid.mean() * qwen.scale
        return hidden, loss

    monkeypatch.setattr("nimloth.backbone.qwen25vl.model.extract_qwen_latents", fake_extract)
    backbone = Qwen25VLBackbone(
        model,
        token_id_map={},
        device=torch.device("cpu"),
        latent_token_count=1,
        lora=False,
        vision_tune="freeze",
    )
    batch = BackboneBatch(
        {
            "input_ids": torch.tensor([[1, 99, 0], [2, 99, 0], [3, 99, 0]]),
            "labels": torch.tensor([[-100, 2, -100], [-100, 4, 6], [-100, 8, 10]]),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2], [1, 2, 2]]),
            "pixel_values": torch.zeros(12, 3),
        }
    )

    output = backbone.forward_chunked(batch, max_rows=1, include_lm_loss=True)

    assert calls == [1, 1, 1]
    torch.testing.assert_close(output.hidden[:, 0], torch.tensor([2.0, 4.0, 6.0]))
    torch.testing.assert_close(output.lm_loss, torch.tensor(12.0))
    output.lm_loss.backward()
    torch.testing.assert_close(model.scale.grad, torch.tensor(6.0))


def test_chunked_forward_only_trains_and_supervises_selected_current_row(monkeypatch) -> None:
    model = _FakeQwen()

    def fake_extract(qwen, enc, *_args, **_kwargs):
        hidden = enc["input_ids"][:, :1].float() * qwen.scale
        labels = enc.get("labels")
        loss = None
        if labels is not None:
            valid = labels[:, 1:][labels[:, 1:] != -100].float()
            loss = valid.mean() * qwen.scale
        return hidden, loss

    monkeypatch.setattr("nimloth.backbone.qwen25vl.model.extract_qwen_latents", fake_extract)
    backbone = Qwen25VLBackbone(
        model,
        token_id_map={},
        device=torch.device("cpu"),
        latent_token_count=1,
        lora=False,
        vision_tune="freeze",
    )
    batch = BackboneBatch(
        {
            "input_ids": torch.tensor([[1, 99, 0], [2, 99, 0], [3, 99, 0]]),
            "labels": torch.tensor([[-100, 2, -100], [-100, 4, 6], [-100, 8, 10]]),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2], [1, 2, 2]]),
            "pixel_values": torch.zeros(12, 3),
        }
    )

    output = backbone.forward_chunked(
        batch,
        max_rows=1,
        include_lm_loss=True,
        gradient_rows={2},
        lm_loss_rows={2},
    )

    torch.testing.assert_close(output.lm_loss, torch.tensor(18.0))
    output.lm_loss.backward()
    torch.testing.assert_close(model.scale.grad, torch.tensor(9.0))
