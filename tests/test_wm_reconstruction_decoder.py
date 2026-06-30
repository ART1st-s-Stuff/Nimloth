from pathlib import Path

import torch

from nimloth.wm.reconstruction import WMImageDecoder, WMImageDecoderConfig


def test_wm_image_decoder_shape_and_range() -> None:
    decoder = WMImageDecoder(WMImageDecoderConfig(emb_dim=32, image_size=64, patch_size=16, hidden_dim=64, depth=1, heads=4))
    out = decoder(torch.randn(2, 32))
    assert out.shape == (2, 3, 64, 64)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_wm_image_decoder_checkpoint_roundtrip(tmp_path: Path) -> None:
    decoder = WMImageDecoder(WMImageDecoderConfig(emb_dim=16, image_size=32, patch_size=16, hidden_dim=32, depth=1, heads=4))
    decoder.save_checkpoint(tmp_path)
    loaded = WMImageDecoder.load_checkpoint(tmp_path)
    assert loaded.config.emb_dim == 16
    assert loaded.config.image_size == 32
    out = loaded(torch.randn(1, 16))
    assert out.shape == (1, 3, 32, 32)
