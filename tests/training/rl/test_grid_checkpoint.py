"""DINO-grid SFT2 checkpoint 到 RL world model 的恢复契约。"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from nimloth.config.rl import parse_rl_config
from nimloth.training.rl.trainer import _build_grid_world_model
from nimloth.wm.grid import (
    EMATargetGridEncoder,
    GridPredictorConfig,
    GridStateProjector,
    GridWorldModel,
    LeWMGridDecoder,
    LeWMGridEncoder,
    SharedSlotProjector,
    TemporalSpatialGridPredictor,
)
from nimloth.wm.value_head import ValueHead


def test_rl_rebuilds_grid_slots_from_sft2_state_checkpoint(tmp_path) -> None:
    """RL 不应依赖 HF model 目录里不存在的 SFT1 projector sidecar。"""

    torch.manual_seed(7)
    state_proj = GridStateProjector(
        SharedSlotProjector(
            input_dim=3,
            output_dim=2,
            hidden_dim=5,
            grid_tokens=2,
        ),
        LeWMGridEncoder(emb_dim=2, hidden_dim=6),
    )
    predictor = TemporalSpatialGridPredictor(
        GridPredictorConfig(
            grid_tokens=2,
            emb_dim=2,
            history_size=2,
            depth=1,
            heads=1,
            dim_head=2,
            mlp_dim=4,
            dropout=0.0,
        )
    )
    source = GridWorldModel(
        state_proj=state_proj,
        target_encoder=EMATargetGridEncoder(
            state_proj.online_encoder,
            decay=0.99,
        ),
        wm_predictor=predictor,
        dino_decoder=LeWMGridDecoder(emb_dim=2, hidden_dim=7),
        value_head=ValueHead(emb_dim=2, hidden_dim=4),
    )
    torch.save(state_proj.state_dict(), tmp_path / "state_proj.pt")
    predictor.save_checkpoint(tmp_path / "wm_predictor")
    source.value_head.save_checkpoint(tmp_path / "value_head")
    source.save_checkpoint_extras(tmp_path)

    config = parse_rl_config(
        {
            "freeze": {"state_proj": True},
            "predictor": {"emb_dim": 2, "history_size": 2},
        }
    )
    args = argparse.Namespace(
        wm_checkpoint=tmp_path / "wm_predictor",
        state_proj_checkpoint=tmp_path / "state_proj.pt",
        value_head_checkpoint=tmp_path / "value_head",
    )
    restored = _build_grid_world_model(
        args,
        config,
        llm=SimpleNamespace(config=SimpleNamespace(hidden_size=3)),
        device=torch.device("cpu"),
    )

    assert isinstance(restored, GridWorldModel)
    assert restored.state_proj.slot_projector.hidden_dim == 5
    assert restored.state_proj.online_encoder.hidden_dim == 6
    assert restored.dino_decoder.hidden_dim == 7
    assert all(not parameter.requires_grad for parameter in restored.state_proj.parameters())
    assert all(not parameter.requires_grad for parameter in restored.target_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in restored.dino_decoder.parameters())
    for name, expected in source.state_proj.state_dict().items():
        torch.testing.assert_close(restored.state_proj.state_dict()[name], expected)
