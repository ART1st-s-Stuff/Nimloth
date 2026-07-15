import torch

from nimloth.eval.wm_predicted_visual_comparison import (
    _state_metrics,
    _visual_metrics,
    load_projected_adapter,
    prepare_wm_conditions,
)
from nimloth.training.reconstruction.state_to_vision_tokens import (
    StateToVisionTokens,
    VisionTokenAdapterConfig,
)


def _record(record_id: str, actions: list[int], shape: tuple[int, ...]) -> dict[int, dict]:
    return {
        step: {
            "record_id": record_id,
            "step_index": step,
            "action_index": actions[min(step, 4)],
            "state_emb": torch.full(shape, float(step)),
            "current_image_path": f"/{record_id}/{step}.png",
        }
        for step in range(6)
    }


class _FakeQwenPredictor:
    def rollout_states(self, initial: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack([initial + float(step) for step in range(1, 6)], dim=1)


class _FakeCurrentPredictor:
    def rollout_states(self, initial: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack([initial + float(step * 2) for step in range(1, 6)], dim=1)


def test_prepare_wm_conditions_aligns_gt_and_autoregressive_rollouts() -> None:
    actions = [0, 4, 0, 5, 0]
    selection = [{"run_index": 0, "record_id": "record", "expected_actions": actions}]
    qwen_gt, qwen_pred, projected_gt, projected_pred = prepare_wm_conditions(
        selection,
        {"record": _record("record", actions, (16, 512))},
        {"record": _record("record", actions, (1024,))},
        _FakeQwenPredictor(),
        _FakeCurrentPredictor(),
        torch.device("cpu"),
    )
    assert qwen_gt.shape == qwen_pred.shape == (5, 16, 512)
    assert projected_gt.shape == projected_pred.shape == (5, 1024)
    torch.testing.assert_close(qwen_gt, qwen_pred)
    torch.testing.assert_close(projected_pred[2], torch.full((1024,), 6.0))


def test_load_projected_adapter_uses_checkpoint_invariants(tmp_path) -> None:
    config = VisionTokenAdapterConfig(input_tokens=1, input_dim=1024)
    adapter = StateToVisionTokens(config)
    checkpoint = tmp_path / "adapter.pt"
    torch.save(
        {
            "invariants": {"projected_config": {
                "input_tokens": 1,
                "input_dim": 1024,
                "output_tokens": 16,
                "output_dim": 512,
                "depth": 2,
                "heads": 8,
                "mlp_ratio": 4,
            }},
            "projected_adapter": adapter.state_dict(),
        },
        checkpoint,
    )
    loaded = load_projected_adapter(checkpoint, torch.device("cpu"))
    output = loaded(torch.zeros(2, 1024))
    assert output.shape == (2, 16, 512)
    assert torch.isfinite(output).all()


def test_visual_metrics_reports_each_horizon() -> None:
    rows = [
        {"step_index": step}
        for _run in range(2)
        for step in range(1, 6)
    ]
    gt = torch.zeros(10, 3, 2, 2)
    branches = {
        "qwen_gt": torch.full_like(gt, 0.1),
        "qwen_wm_pred": torch.full_like(gt, 0.2),
        "query_gt": torch.full_like(gt, 0.3),
        "projected_gt": torch.full_like(gt, 0.4),
        "current_wm_pred": torch.full_like(gt, 0.5),
    }
    qwen_state = _state_metrics(torch.ones(10, 4), torch.zeros(10, 4))
    current_state = _state_metrics(torch.ones(10, 4), torch.zeros(10, 4))
    metrics, horizons = _visual_metrics(rows, gt, branches, qwen_state, current_state)
    assert metrics["visual/qwen_gt_to_gt_l1"] == torch.tensor(0.1).item()
    assert metrics["visual/current_wm_pred_to_gt_l1"] == 0.5
    assert set(horizons) == {"1", "2", "3", "4", "5"}
    assert horizons["3"]["qwen_wm_state_mse"] == 1.0
