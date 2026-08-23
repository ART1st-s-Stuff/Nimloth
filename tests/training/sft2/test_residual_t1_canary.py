from __future__ import annotations

import numpy as np

from nimloth.training.sft2.residual_t1_canary import _train_model, residual_t1_metrics


def test_residual_t1_metrics_gate_is_per_supported_action() -> None:
    current = np.zeros((8, 2, 2), dtype=np.float32)
    target = np.ones_like(current)
    prediction = np.full_like(current, 0.5)
    dino = np.ones_like(current)
    actions = np.asarray([0, 0, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    metrics = residual_t1_metrics(
        current_state=current,
        actual_next_state=target,
        predicted_next_state=prediction,
        next_dino=dino,
        actions=actions,
        minimum_primary_count=2,
    )
    assert metrics["overall"]["copy_relative_skill"] == 0.75
    assert metrics["primary_actions"] == [0, 2, 3, 4]
    assert metrics["gate"]["all_primary_action_skill_positive"] is True
    assert metrics["gate"]["passed"] is False  # std ratio is outside [0.9, 1.1]


def test_residual_t1_metrics_rejects_action_that_regresses() -> None:
    rng = np.random.default_rng(4)
    current = rng.normal(size=(6, 2, 3)).astype(np.float32)
    target = current + 1.0
    prediction = target.copy()
    actions = np.asarray([0, 0, 2, 2, 3, 3], dtype=np.int64)
    prediction[actions == 2] = current[actions == 2] - 1.0
    metrics = residual_t1_metrics(
        current_state=current,
        actual_next_state=target,
        predicted_next_state=prediction,
        next_dino=target,
        actions=actions,
        minimum_primary_count=2,
    )
    assert metrics["per_action"]["2"]["copy_relative_skill"] < 0
    assert metrics["gate"]["all_primary_action_skill_positive"] is False
    assert metrics["gate"]["passed"] is False


def test_tiny_residual_training_improves_copy() -> None:
    import torch

    from nimloth.wm.grid import GridPredictorConfig, ResidualTemporalSpatialGridPredictor

    rng = np.random.default_rng(8)
    current = rng.normal(size=(48, 4, 8)).astype(np.float32)
    actions = np.tile(np.asarray([0, 2, 3, 4], dtype=np.int64), 12)
    target = current + (actions[:, None, None].astype(np.float32) + 1.0) * 0.05
    model = ResidualTemporalSpatialGridPredictor(
        GridPredictorConfig(
            grid_tokens=4,
            emb_dim=8,
            history_size=1,
            depth=1,
            heads=2,
            dim_head=4,
            mlp_dim=16,
            dropout=0.0,
        )
    )
    step_log: list[dict[str, object]] = []
    model, _, _ = _train_model(
        model=model,
        current=current,
        target=target,
        actions=actions,
        device=torch.device("cpu"),
        epochs=8,
        batch_size=16,
        learning_rate=1e-2,
        weight_decay=0.0,
        seed=9,
        phase="test",
        step_log=step_log,
    )
    with torch.inference_mode():
        prediction = model(torch.from_numpy(current), torch.from_numpy(actions)).numpy()
    copy_mse = float(np.mean(np.square(current - target)))
    prediction_mse = float(np.mean(np.square(prediction - target)))
    assert prediction_mse < copy_mse
    assert step_log
