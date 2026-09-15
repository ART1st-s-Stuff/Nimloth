import importlib.util
import sys
from pathlib import Path

import pytest
import torch

DIRECTORY = Path(__file__).resolve().parents[3]/"experiments/training/sft/stage3"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("state_gradient_diagnostic", DIRECTORY/"diagnose_state_gradients.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_descent_direction_sign():
    result = module.gradient_geometry(torch.tensor([2., 0.]), torch.tensor([-3., 0.]))
    assert result["gradient_cosine"] == -1
    assert result["dino_directional_derivative_along_unit_negative_sigreg"] == 2
    aligned = module.gradient_geometry(torch.tensor([2., 0.]), torch.tensor([3., 0.]))
    assert aligned["dino_directional_derivative_along_unit_negative_sigreg"] == -2


def test_real_sigreg_preserves_inputs_and_rng():
    state = torch.randn(4, 2, 8)
    target = torch.randn_like(state)
    original = state.clone()
    rng = torch.random.get_rng_state().clone()
    result = module.evaluate(state, target, torch.tensor([[0, 1], [1, 2], [2, 3]]), seed=42, num_proj=8)
    assert torch.equal(state, original)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert state.grad is None
    assert result["weighted_sigreg_gradient_norm"] > 0
    assert result["finite_difference_weighted_dino_derivative"] == pytest.approx(
        result["dino_directional_derivative_along_unit_negative_sigreg"], abs=2e-4)
