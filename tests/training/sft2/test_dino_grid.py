from types import SimpleNamespace
import json

import torch
import pytest

from nimloth.training.common import world_model_loss
from nimloth.training.sft.stage3.batch import Stage3BatchAssembler
from nimloth.training.sft.stage3.dino_grid import DINOGridBatchAssembler
from nimloth.training.sft.stage3.trainer import _build_world_model
from nimloth.wm.grid import SharedSlotProjector, ResidualTemporalSpatialGridPredictor, TemporalSpatialGridPredictor


@pytest.mark.parametrize("kind", ["direct", "residual"])
def test_grid_world_model_keeps_trainable_grid_modules_in_fp32(tmp_path, kind) -> None:
    slot_projector = SharedSlotProjector(
        input_dim=6,
        output_dim=4,
        hidden_dim=8,
        grid_tokens=2,
    )
    (tmp_path / "grid_state_config.json").write_text(
        json.dumps(
            {
                "grid_tokens": 2,
                "qwen_hidden_dim": 6,
                "state_dim": 4,
                "projector_hidden_dim": 8,
                "shared_slot_projector": True,
                "ordering": "row_major",
            }
        ),
        encoding="utf-8",
    )
    torch.save(slot_projector.state_dict(), tmp_path / "slot_projector.pt")
    model = torch.nn.Linear(1, 1, bias=False).to(dtype=torch.bfloat16)
    model.config = SimpleNamespace(hidden_size=6)
    args = SimpleNamespace(
        objective="dino_grid",
        model=tmp_path,
        emb_dim=4,
        latent_token_count=2,
        history_size=1,
        grid_wm_depth=1,
        grid_wm_heads=1,
        grid_wm_dim_head=4,
        grid_wm_mlp_dim=8,
        grid_wm_dropout=0.0,
        grid_predictor_kind=kind,
        resume=False,
    )

    world_model, world_model_device = _build_world_model(
        args,
        model=model,
        device=torch.device("cpu"),
        pair_parallel=False,
        resume_ckpt_dir=None,
        train_wm_predictor=True,
    )

    assert world_model_device == torch.device("cpu")
    expected_type = ResidualTemporalSpatialGridPredictor if kind == "residual" else TemporalSpatialGridPredictor
    assert type(world_model.wm_predictor) is expected_type
    if kind == "residual":
        state = torch.randn(2, 2, 4, requires_grad=True)
        prediction = world_model.wm_predictor(state, torch.tensor([0, 1]))
        assert torch.equal(prediction, state)
        prediction.sum().backward()
        torch.testing.assert_close(state.grad, torch.ones_like(state))
    assert next(world_model.state_proj.parameters()).dtype == torch.float32
    assert all(
        parameter.requires_grad
        for parameter in world_model.state_proj.parameters()
    )
    for module in (
        world_model.wm_predictor,
        world_model.value_head,
    ):
        assert next(module.parameters()).dtype == torch.float32


def test_dino_loss_directly_supervises_predicted_state() -> None:
    predicted_state = torch.zeros(1, 4, 8, requires_grad=True)
    target = torch.ones(1, 4, 8)

    objective = world_model_loss(
        predicted_state,
        predicted_state.detach(),
        state_weight=1.0,
        dino_grid_target=target,
        dino_grid_weight=1.0,
    )
    assert objective.dino_grid_mse is not None
    objective.dino_grid_mse.backward()

    torch.testing.assert_close(objective.dino_grid_mse, torch.tensor(1.0))
    assert predicted_state.grad is not None
    assert torch.count_nonzero(predicted_state.grad) == predicted_state.numel()



def test_native_dino_targets_follow_all_window_successors(trajectory_factory):
    builder, _, trajectory = trajectory_factory(length=6)
    base = Stage3BatchAssembler(input_builder=builder, device=torch.device("cpu"), prediction_horizon=4)
    class Targets:
        grid_size = 2
        identity = SimpleNamespace(hidden_size=1024)
        def __init__(self):
            self.calls = []
        def load(self, paths, *, device):
            self.calls.append(tuple(paths))
            return torch.tensor([float(path.split("_")[-1].split(".")[0]) for path in paths], device=device)[:,None,None].expand(-1,4,1024)
    targets = Targets()
    batch = DINOGridBatchAssembler(base, targets).prepare([trajectory])
    assert batch.dino_grid_target.shape == (3,4,4,1024)
    assert batch.dino_grid_target[:,:,0,0].tolist() == [[1,2,3,4], [2,3,4,5], [3,4,5,6]]
    assert batch.current_dino_target[:,0,0].tolist() == [0,1,2]
    assert targets.calls == [tuple(f"a_{i}.png" for i in range(7))]
    assert batch.observed_dino_target.shape == (7, 4, 1024)
    assert batch.observed_state_weights.tolist() == [1.] * 7
