"""DINO-grid SFT2 checkpoint 到 RL world-model 的严格装载测试。"""

from __future__ import annotations

import torch

from nimloth.wm.factory import (
    load_world_model,
    world_model_artifacts_are_complete,
)
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


def test_rl_loads_self_contained_grid_state_without_sft1_sidecars(tmp_path) -> None:
    online_encoder = LeWMGridEncoder(emb_dim=2, hidden_dim=4)
    state_proj = GridStateProjector(
        SharedSlotProjector(
            input_dim=3,
            output_dim=2,
            hidden_dim=5,
            grid_tokens=2,
        ),
        online_encoder,
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
        target_encoder=EMATargetGridEncoder(online_encoder, decay=0.99),
        wm_predictor=predictor,
        dino_decoder=LeWMGridDecoder(emb_dim=2, hidden_dim=6),
        value_head=ValueHead(emb_dim=2),
    )
    torch.save(state_proj.state_dict(), tmp_path / "state_proj.pt")
    predictor.save_checkpoint(tmp_path / "wm_predictor")
    source.value_head.save_checkpoint(tmp_path / "value_head")
    source.save_checkpoint_extras(tmp_path)

    assert world_model_artifacts_are_complete(tmp_path)

    loaded = load_world_model(
        predictor_checkpoint=tmp_path / "wm_predictor",
        state_proj_checkpoint=tmp_path / "state_proj.pt",
        value_head_checkpoint=tmp_path / "value_head",
        qwen_hidden_dim=3,
        expected_emb_dim=2,
        expected_history_size=2,
        freeze_state_proj=True,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded, GridWorldModel)
    assert loaded.wm_predictor.config.history_size == 2
    assert loaded.state_proj.slot_projector.hidden_dim == 5
    assert loaded.dino_decoder.hidden_dim == 6
    assert all(not parameter.requires_grad for parameter in loaded.state_proj.parameters())
    assert all(
        not parameter.requires_grad for parameter in loaded.target_encoder.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in loaded.dino_decoder.parameters()
    )
    assert loaded.train_dino_decoder is False
    assert loaded.update_target_encoder is False
