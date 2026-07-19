from __future__ import annotations

import torch

from nimloth.training.sft1.dino_grid import compute_dino_grid_alignment_loss
from nimloth.wm.grid import (
    GridLatentWMPredictor,
    SharedSlotProjector,
    load_sft1_slot_projector,
)


class FakeGridTeacher:
    hidden_size = 4
    grid_size = 2

    def encode_image_paths_grid(self, paths, *, device, grid_size):
        assert grid_size == 2
        rows = []
        for path in paths:
            base = float(len(str(path)))
            rows.append(
                torch.tensor(
                    [
                        [base, 0, 0, 0],
                        [0, base, 0, 0],
                        [0, 0, base, 0],
                        [0, 0, 0, base],
                    ],
                    device=device,
                )
            )
        return torch.stack(rows)


def test_shared_slot_projector_applies_one_mlp_to_every_slot() -> None:
    projector = SharedSlotProjector(input_dim=3, output_dim=2, hidden_dim=5)
    hidden = torch.randn(2, 4, 3)

    actual = projector(hidden)
    expected = torch.stack(
        [torch.stack([projector.net(slot) for slot in row]) for row in hidden]
    )

    assert actual.shape == (2, 4, 2)
    torch.testing.assert_close(actual, expected)


def test_dino_grid_alignment_is_slotwise_and_has_gradient() -> None:
    projector = SharedSlotProjector(input_dim=3, output_dim=4, hidden_dim=5)
    hidden = torch.randn(2, 4, 3, requires_grad=True)
    items = [
        {"current_image_path": "a.png"},
        {"current_image_path": "longer.png"},
    ]

    loss, metrics = compute_dino_grid_alignment_loss(
        current_query_hidden=hidden,
        items=items,
        slot_projector=projector,
        dino_encoder=FakeGridTeacher(),
        grid_size=2,
    )
    loss.backward()

    assert loss.ndim == 0
    assert metrics["dino_grid_mse"] == loss.detach().item()
    assert hidden.grad is not None
    assert any(parameter.grad is not None for parameter in projector.parameters())


def test_grid_wm_jointly_predicts_complete_grid() -> None:
    torch.manual_seed(0)
    wm = GridLatentWMPredictor(
        grid_tokens=4,
        emb_dim=8,
        action_dim=6,
        depth=2,
        heads=2,
        mlp_dim=16,
    )
    state = torch.randn(3, 4, 8, requires_grad=True)
    actions = torch.tensor([0, 1, 5])

    predicted = wm(state, actions)
    predicted.sum().backward()

    assert predicted.shape == state.shape
    assert state.grad is not None
    assert wm.spatial_position.shape == (1, 4, 8)


def test_sft1_slot_projector_interface_is_fail_closed(tmp_path) -> None:
    projector = SharedSlotProjector(3, 4, 5, grid_tokens=16)
    torch.save(projector.state_dict(), tmp_path / "slot_projector.pt")
    (tmp_path / "grid_state_config.json").write_text(
        '{"grid_tokens":16,"qwen_hidden_dim":3,"state_dim":4,'
        '"projector_hidden_dim":5,"shared_slot_projector":true,"ordering":"row_major"}'
    )

    loaded = load_sft1_slot_projector(
        tmp_path,
        qwen_hidden_dim=3,
        state_dim=4,
        grid_tokens=16,
    )
    hidden = torch.randn(2, 16, 3)
    torch.testing.assert_close(loaded(hidden), projector(hidden))

    try:
        load_sft1_slot_projector(tmp_path, qwen_hidden_dim=8, state_dim=4, grid_tokens=16)
    except ValueError as exc:
        assert "interface mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("interface mismatch must fail")


def test_grid_wm_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(1)
    wm = GridLatentWMPredictor(
        grid_tokens=4,
        emb_dim=8,
        action_dim=6,
        depth=1,
        heads=2,
        mlp_dim=16,
    )
    state = torch.randn(2, 4, 8)
    actions = torch.tensor([1, 3])
    expected = wm(state, actions)

    wm.save_checkpoint(tmp_path)
    loaded = GridLatentWMPredictor.load_checkpoint(tmp_path)

    torch.testing.assert_close(loaded(state, actions), expected)


def test_grid_wm_rejects_wrong_grid_shape() -> None:
    wm = GridLatentWMPredictor(grid_tokens=16, emb_dim=8, action_dim=6, depth=1, heads=2, mlp_dim=16)
    try:
        wm(torch.randn(2, 9, 8), torch.tensor([0, 1]))
    except ValueError as exc:
        assert "expected state shape" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("wrong grid shape must fail")
