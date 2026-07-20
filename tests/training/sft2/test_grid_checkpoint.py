from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from nimloth.wm.grid import (
    EMATargetGridEncoder,
    LeWMGridDecoder,
    LeWMGridEncoder,
    LeWMSpatialPredictor,
)
from nimloth.wm.value_head import ValueHead


def _entrypoint_module():
    path = Path(__file__).resolve().parents[3] / "experiments/training/sft2/train_grid.py"
    spec = importlib.util.spec_from_file_location("nimloth_test_train_grid", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_sft2_checkpoint_round_trip_and_invariants(tmp_path) -> None:
    entry = _entrypoint_module()
    encoder = LeWMGridEncoder(emb_dim=1024, hidden_dim=4)
    target = EMATargetGridEncoder(encoder, decay=0.99)
    wm = LeWMSpatialPredictor(
        grid_tokens=16,
        emb_dim=1024,
        action_dim=8,
        depth=1,
        heads=1,
        dim_head=1,
        mlp_dim=4,
        dropout=0.0,
    )
    decoder = LeWMGridDecoder(emb_dim=1024, hidden_dim=4)
    value = ValueHead(emb_dim=1024)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *wm.parameters(), *decoder.parameters(), *value.parameters()]
    )
    args = SimpleNamespace(
        sft1_checkpoint=tmp_path / "sft1",
        dino_model="facebook/dinov2-large",
        dino_identity={"source": "facebook/dinov2-large", "revision": "rev", "processor_fingerprint": "proc", "hidden_size": 1024},
        dino_cache_root=tmp_path / "cache",
        dino_cache_fingerprint="fingerprint",
        latent_weight=1.0,
        dino_weight=0.5,
        sigreg_weight=0.1,
        value_weight=1.0,
        ema_decay=0.99,
        wm_depth=1,
        wm_heads=1,
        wm_dim_head=1,
        wm_mlp_dim=4,
        wm_dropout=0.0,
    )
    expected = {key: value.detach().clone() for key, value in encoder.state_dict().items()}
    checkpoint = tmp_path / "checkpoint"

    entry._save_checkpoint(
        encoder,
        target,
        wm,
        decoder,
        value,
        optimizer,
        checkpoint,
        epoch=2,
        step=7,
        best=0.25,
        args=args,
    )
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.add_(1.0)
    start_epoch, step, best = entry._load_checkpoint(
        encoder,
        target,
        wm,
        decoder,
        value,
        optimizer,
        checkpoint,
        args,
    )

    assert (start_epoch, step, best) == (3, 7, 0.25)
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, expected[key])
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["dino_weight"] == 0.5
    assert state["ema_decay"] == 0.99
    assert state["dino_cache_fingerprint"] == "fingerprint"
