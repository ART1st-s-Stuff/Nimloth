from __future__ import annotations

import torch

from nimloth.training.sft2.grid_step import compute_grid_sft2_loss
from nimloth.wm.grid import (
    EMATargetGridEncoder,
    LeWMGridDecoder,
    LeWMGridEncoder,
    LeWMSpatialPredictor,
    SharedSlotProjector,
)
from nimloth.wm.value_head import ValueHead


def test_lewm_spatial_predictor_is_noncausal_across_grid_slots() -> None:
    torch.manual_seed(0)
    predictor = LeWMSpatialPredictor(
        grid_tokens=4,
        emb_dim=8,
        action_dim=6,
        depth=1,
        heads=2,
        dim_head=4,
        mlp_dim=16,
        dropout=0.0,
    ).eval()
    # LeWM initializes AdaLN gates to zero. Open the attention gate solely for
    # this connectivity test, then perturb one normalized feature in the last slot.
    with torch.no_grad():
        bias = predictor.layers[0].adaLN_modulation[-1].bias
        bias[2 * predictor.emb_dim : 3 * predictor.emb_dim].fill_(1.0)
    state = torch.randn(1, 4, 8)
    changed = state.clone()
    changed[:, -1, 0] += 10.0
    action = torch.tensor([2])

    first_before = predictor(state, action)[:, 0]
    first_after = predictor(changed, action)[:, 0]

    assert not torch.allclose(first_before, first_after)


def test_ema_target_encoder_updates_without_gradients() -> None:
    online = LeWMGridEncoder(emb_dim=8, hidden_dim=16)
    target = EMATargetGridEncoder(online, decay=0.5)
    before = [parameter.detach().clone() for parameter in target.encoder.parameters()]
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(2.0)
    target.update(online)

    for old, online_parameter, target_parameter in zip(
        before, online.parameters(), target.encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(target_parameter, old * 0.5 + online_parameter * 0.5)
        assert target_parameter.requires_grad is False


def test_grid_sft2_trains_encoder_wm_decoder_and_value_with_dino() -> None:
    torch.manual_seed(0)
    projector = SharedSlotProjector(6, 8, 10, grid_tokens=4)
    projector.requires_grad_(False)
    encoder = LeWMGridEncoder(emb_dim=8, hidden_dim=16)
    target_encoder = EMATargetGridEncoder(encoder, decay=0.99)
    wm = LeWMSpatialPredictor(
        grid_tokens=4,
        emb_dim=8,
        action_dim=6,
        depth=1,
        heads=2,
        dim_head=4,
        mlp_dim=16,
        dropout=0.0,
    )
    decoder = LeWMGridDecoder(emb_dim=8, hidden_dim=16)
    value = ValueHead(emb_dim=8, hidden_dim=8)
    current = torch.randn(3, 4, 6)
    next_hidden = torch.randn(3, 4, 6)
    dino_target = torch.randn(3, 4, 8)

    loss, metrics = compute_grid_sft2_loss(
        current_query_hidden=current,
        next_query_hidden=next_hidden,
        dino_target_grid=dino_target,
        action_indices=torch.tensor([0, 2, 5]),
        value_targets=torch.tensor([1.0, 0.5, 0.0]),
        slot_projector=projector,
        online_encoder=encoder,
        target_encoder=target_encoder,
        grid_wm=wm,
        decoder=decoder,
        value_head=value,
        latent_weight=1.0,
        dino_weight=0.5,
        sigreg_weight=0.1,
        value_weight=1.0,
        sigreg_num_proj=8,
        sigreg_knots=5,
    )
    loss.backward()

    assert loss.ndim == 0
    assert set(metrics) == {
        "latent_mse",
        "dino_grid_mse",
        "sigreg",
        "value_mse",
        "total",
    }
    assert all(parameter.grad is None for parameter in projector.parameters())
    assert all(parameter.grad is None for parameter in target_encoder.parameters())
    assert any(parameter.grad is not None for parameter in encoder.parameters())
    assert any(parameter.grad is not None for parameter in wm.parameters())
    assert any(parameter.grad is not None for parameter in decoder.parameters())
    assert any(parameter.grad is not None for parameter in value.parameters())


def test_terminal_transition_uses_dino_and_value_without_latent_target() -> None:
    projector = SharedSlotProjector(6, 8, 10, grid_tokens=4).requires_grad_(False)
    encoder = LeWMGridEncoder(emb_dim=8, hidden_dim=16)
    target_encoder = EMATargetGridEncoder(encoder, decay=0.99)
    wm = LeWMSpatialPredictor(
        grid_tokens=4,
        emb_dim=8,
        action_dim=6,
        depth=1,
        heads=2,
        dim_head=4,
        mlp_dim=16,
        dropout=0.0,
    )
    decoder = LeWMGridDecoder(emb_dim=8, hidden_dim=16)
    value = ValueHead(emb_dim=8, hidden_dim=8)

    loss, metrics = compute_grid_sft2_loss(
        current_query_hidden=torch.randn(2, 4, 6),
        next_query_hidden=torch.empty(0, 4, 6),
        latent_indices=[],
        dino_target_grid=torch.randn(2, 4, 8),
        action_indices=torch.tensor([1, 2]),
        value_targets=torch.tensor([0.0, 1.0]),
        slot_projector=projector,
        online_encoder=encoder,
        target_encoder=target_encoder,
        grid_wm=wm,
        decoder=decoder,
        value_head=value,
        sigreg_num_proj=8,
        sigreg_knots=5,
    )
    loss.backward()

    assert metrics["latent_mse"] == 0.0
    assert metrics["dino_grid_mse"] > 0.0
    assert any(parameter.grad is not None for parameter in wm.parameters())
    assert any(parameter.grad is not None for parameter in decoder.parameters())
