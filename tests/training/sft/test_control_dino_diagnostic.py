"""Matched diagnostic counts and decomposition, independent of GPU runtime."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

DIRECTORY = Path(__file__).resolve().parents[3] / "experiments/training/sft/stage3"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("control_diagnostic", DIRECTORY / "run_control_dino_diagnostic.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(values):
    tensor = torch.tensor(values, dtype=torch.float64).reshape(-1, 1, 1)
    return {"actions": torch.zeros(len(values)), "dino": tensor,
            "current_dino": torch.zeros(1, 1), "online_direct": tensor.clone()}


def test_constant_shift_decomposes_and_deduplicates():
    initial = {("t", 0): row([1, 2]), ("t", 1): row([2, 3])}
    final = {key: {**value, "online_direct": value["online_direct"] + 4} for key, value in initial.items()}
    metrics = module.summarize(initial, final)
    assert metrics["observations"] == 3
    assert metrics["models"]["stage2"]["mse"] == 0
    assert metrics["output_shift"]["mean_bias_mse"] == 16
    assert metrics["output_shift"]["centered_error_mse"] == 0
    assert metrics["models"]["control"]["mse"] == 16


def test_centered_error_identity():
    value = module.decompose(torch.tensor([[[1., 3.]], [[-1., 1.]]]))
    assert value["mse"] == pytest.approx(value["mean_bias_mse"] + value["centered_error_mse"])
    assert value["centered_error_mse"] > 0


def test_reject_mismatched_targets():
    with pytest.raises(ValueError, match="dino mismatch"):
        module.summarize({("t", 0): row([1])}, {("t", 0): row([2])})


def test_reject_inconsistent_overlapping_observation():
    initial = {("t", 0): row([1, 2]), ("t", 1): row([2, 3])}
    final = {key: {**value, "online_direct": value["online_direct"].clone()}
             for key, value in initial.items()}
    final[("t", 1)]["online_direct"][0] += 1
    with pytest.raises(ValueError, match="inconsistent duplicate observation"):
        module.summarize(initial, final)


def test_commands_allow_paused_parent_and_never_treatment(tmp_path):
    prior = {"status": "paused", "phases": [{"phase": "formal", "arm": "control", "argv":
        ["python", "--master_port=1234", "--output-dir", "old", "--outcome-eval-dir", "old_eval",
         "--wandb-run-name", "old_name"]}]}
    commands = module.build_commands(prior, tmp_path, tmp_path / "epoch_002")
    assert [c["name"] for c in commands] == ["stage2", "control"]
    assert "--resume" not in commands[0]["argv"]
    assert "--resume" in commands[1]["argv"]
    assert all("--eval-only" in c["argv"] for c in commands)
