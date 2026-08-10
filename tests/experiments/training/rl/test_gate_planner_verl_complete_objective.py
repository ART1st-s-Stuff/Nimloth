from __future__ import annotations

from pathlib import Path

import pytest

from experiments.training.rl.gate_planner_verl_complete_objective import (
    _finite_metrics,
    _validate_worker_placement,
)
from nimloth.config.rl import load_rl_config


ROOT = Path(__file__).resolve().parents[4]


def test_complete_objective_gate_config_uses_one_epoch_and_eight_fsdp_ranks() -> None:
    config = load_rl_config(
        ROOT
        / "configs/training/rl/planner_policy_h1_complete_objective_fsdp_gate.yaml"
    )
    assert config.distributed.nodes == 1
    assert config.distributed.world_size == 8
    assert config.distributed.gpus_per_rank == 1
    assert config.planner_policy.ppo_epochs == 1
    assert config.predictor.train_wm is True
    assert config.predictor.lambda_wm == 1.0
    assert config.predictor.lambda_dino == 0.5
    assert config.gradient.state_source == "recompute"
    assert config.gradient.representation_to_backbone is True


def test_complete_objective_gate_requires_every_finite_loss_component() -> None:
    metrics = {
        "wm_mse": 1.0,
        "dino_grid_mse": 2.0,
        "value_loss": 3.0,
        "planner_policy_loss": -0.5,
        "planner_policy_entropy": 1.5,
        "total_loss": 5.0,
    }
    second = {**metrics, "wm_mse": 3.0, "total_loss": 7.0}
    assert _finite_metrics([metrics, second]) == {
        **metrics,
        "wm_mse": 2.0,
        "total_loss": 6.0,
    }

    invalid = dict(metrics, dino_grid_mse=0.0)
    with pytest.raises(RuntimeError, match="WM/DINO"):
        _finite_metrics([invalid, dict(invalid)])


def test_complete_objective_worker_placement_supports_strict_two_by_four() -> None:
    rows = [
        {
            "rank": rank,
            "hostname": f"node-{rank // 4}",
            "ray_node_id": f"ray-{rank // 4}",
            "cuda_device_uuid": f"gpu-{rank}",
        }
        for rank in range(8)
    ]
    _validate_worker_placement(
        rows,
        8,
        expected_node_count=2,
        workers_per_node=4,
    )

    rows[-1]["hostname"] = "node-0"
    with pytest.raises(RuntimeError, match="Slurm worker placement"):
        _validate_worker_placement(
            rows,
            8,
            expected_node_count=2,
            workers_per_node=4,
        )


def test_complete_objective_launcher_is_strict_weights_only_one_by_eight() -> None:
    launcher = (
        ROOT
        / "experiments/training/rl/gate_planner_verl_complete_objective_1x8.slurm"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=normal" in launcher
    assert "#SBATCH --nodes=1" in launcher
    assert "#SBATCH --gpus-per-task=8" in launcher
    assert "--world-size 8" in launcher
    assert 'GATE_MINIMUM_STATE_TOKENS=${GATE_MINIMUM_STATE_TOKENS:-6000}' in launcher
    assert "GATE_MINIMUM_STATE_TOKENS >= 14000" in launcher
    assert "--diagnostic-nonbehavior-prefix" in launcher
    assert "GATE_EXPECTED_MANIFEST_SHA256" in launcher
    assert "GATE_EXPECTED_CURRENT_POLICY_SHA256" in launcher
    assert "--expected-current-planner-policy-head-sha256" in launcher
    assert "--fresh-rollout-manifest" in launcher
    assert "--planner-policy-head-checkpoint" in launcher
    assert "--resume" not in launcher
    assert 'REQUESTED_WANDB_PROJECT="${WANDB_PROJECT}"' in launcher
    source = "source /project/peilab/atst/flower/.env"
    restore = 'export WANDB_PROJECT="${REQUESTED_WANDB_PROJECT}"'
    assert launcher.index(source) < launcher.index(restore)


def test_complete_objective_launcher_supports_preempt_two_by_four() -> None:
    launcher = (
        ROOT
        / "experiments/training/rl/"
        "gate_planner_verl_complete_objective_2x4_preempt.slurm"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=preempt" in launcher
    assert "#SBATCH --nodes=2" in launcher
    assert "#SBATCH --gres=gpu:4" in launcher
    assert '--num-gpus=4' in launcher
    assert '--workers-per-node 4' in launcher
    assert '--expected-node-count 2' in launcher
    assert 'counts != [4.0, 4.0]' in launcher
    assert 'export RAY_ADDRESS=' in launcher
