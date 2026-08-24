import json

import numpy as np
import torch

from nimloth.training.sft2.state_interface_canary import (
    ResidualStateInterfaceCanary,
    StateInterfaceCanaryConfig,
    canary_gate,
    grouped_record_selection,
    normalized_multitask_loss,
    save_canary_checkpoint,
    visual_state_metrics,
)


def test_zero_initialized_residual_is_exact_copy_and_bounded() -> None:
    config = StateInterfaceCanaryConfig(
        hidden_dim=8,
        state_dim=4,
        grid_tokens=3,
        adapter_rank=2,
        goal_classes=5,
        movement_actions=(0, 2, 3),
        max_residual_fraction=0.1,
    )
    model = ResidualStateInterfaceCanary(config)
    hidden = torch.randn(6, 3, 8)
    baseline = torch.randn(6, 3, 4)
    assert torch.equal(model.calibrated_state(hidden, baseline), baseline)

    with torch.no_grad():
        model.adapter[-1].weight.fill_(100.0)
        model.adapter[-1].bias.fill_(100.0)
    candidate = model.calibrated_state(hidden, baseline)
    residual_norm = (candidate - baseline).flatten(1).norm(dim=1)
    baseline_norm = baseline.flatten(1).norm(dim=1)
    assert torch.all(residual_norm <= 0.100001 * baseline_norm)


def test_heads_read_one_unified_state_and_action_specific_logits() -> None:
    model = ResidualStateInterfaceCanary(
        StateInterfaceCanaryConfig(
            hidden_dim=8,
            state_dim=4,
            grid_tokens=3,
            adapter_rank=2,
            goal_classes=5,
            movement_actions=(0, 2, 3),
            max_residual_fraction=0.1,
        )
    )
    state = torch.randn(4, 3, 4)
    assert model.goal_logits(state).shape == (4, 5)
    logits = model.outcome_logits(state, torch.tensor([0, 2, 3, 0]))
    assert logits.shape == (4,)
    try:
        model.outcome_logits(state, torch.tensor([0, 1, 3, 0]))
    except ValueError as error:
        assert "unsupported movement action" in str(error)
    else:
        raise AssertionError("unsupported action was accepted")


def test_grouped_record_selection_never_splits_group() -> None:
    groups = np.asarray(["a", "a", "b", "c", "c", "d", "e", "f"])
    labels = np.asarray(["x", "x", "x", "y", "y", "y", "x", "y"])
    selected = grouped_record_selection(groups, labels, modulo=3)
    for group in set(groups):
        assert len(set(selected[groups == group].tolist())) == 1
    assert selected.any() and (~selected).any()
    for label in set(labels):
        assert np.any((labels == label) & ~selected)


def test_visual_metrics_and_gate_require_all_interfaces() -> None:
    dino = np.ones((5, 2, 3), dtype=np.float32)
    baseline = dino * 0.8
    candidate = dino * 0.88
    visual = visual_state_metrics(candidate, dino, baseline)
    assert visual["candidate_dino_rmse"] < visual["baseline_dino_rmse"]
    passed = canary_gate(
        visual_metrics=visual,
        goal_gate_passed=True,
        outcome_checks={"forward": True, "right": True, "left": True},
        hidden_probe_supports_calibration=True,
    )
    assert passed["passed"] is True
    failed = canary_gate(
        visual_metrics={**visual, "candidate_dino_cosine": 0.1},
        goal_gate_passed=True,
        outcome_checks={"forward": True, "right": True, "left": True},
        hidden_probe_supports_calibration=True,
    )
    assert failed["passed"] is False


def test_multitask_objective_updates_only_canary_parameters() -> None:
    torch.manual_seed(7)
    config = StateInterfaceCanaryConfig(
        hidden_dim=8,
        state_dim=4,
        grid_tokens=3,
        adapter_rank=2,
        goal_classes=2,
        movement_actions=(0, 2, 3),
        max_residual_fraction=0.1,
    )
    model = ResidualStateInterfaceCanary(config)
    hidden = torch.randn(12, 3, 8)
    baseline = torch.randn(12, 3, 4)
    dino = baseline + 0.1 * torch.randn(12, 3, 4)
    labels = torch.arange(12) % 2
    actions = torch.tensor([0, 2, 3] * 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    losses = []
    for _ in range(15):
        optimizer.zero_grad(set_to_none=True)
        loss, components = normalized_multitask_loss(
            model=model,
            visual_hidden=hidden,
            visual_baseline=baseline,
            visual_dino=dino,
            goal_hidden=hidden,
            goal_baseline=baseline,
            goal_labels=labels,
            outcome_hidden=hidden,
            outcome_baseline=baseline,
            outcome_actions=actions,
            outcome_labels=labels.bool(),
            visual_reference_loss=0.1,
            goal_reference_loss=np.log(2),
            outcome_reference_loss=np.log(2),
            anchor_weight=0.25,
        )
        assert set(components) == {"visual", "goal", "outcome", "anchor"}
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert not torch.equal(model.calibrated_state(hidden, baseline), baseline)


def test_checkpoint_is_optimizer_free_and_reloads_exactly(tmp_path) -> None:
    config = StateInterfaceCanaryConfig(
        hidden_dim=8,
        state_dim=4,
        grid_tokens=3,
        adapter_rank=2,
        goal_classes=5,
        movement_actions=(0, 2, 3),
        max_residual_fraction=0.1,
    )
    model = ResidualStateInterfaceCanary(config)
    with torch.no_grad():
        model.adapter[-1].bias.fill_(0.25)
    path = tmp_path / "checkpoint"
    save_canary_checkpoint(model, path)
    metadata = json.loads((path / "canary_config.json").read_text())
    assert metadata["optimizer_state_present"] is False
    payload = torch.load(path / "diagnostic_adapter.pt", weights_only=True)
    assert set(payload) == set(model.adapter.state_dict())
    assert not any("optimizer" in name for name in payload)
