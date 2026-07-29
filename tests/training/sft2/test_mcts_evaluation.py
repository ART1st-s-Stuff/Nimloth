"""SFT2 H=1/K-step MCTS pre-RL evaluation contract tests."""

from __future__ import annotations

import json

import pytest
import torch

from experiments.training.sft2.eval_mcts_rollout import (
    build_rollout_argv,
    parse_args,
    write_or_validate_contract,
)
from nimloth.training.sft2.algorithm import SFT2_VALUE_OBJECTIVE
from nimloth.training.sft2.mcts_evaluation import (
    load_sft2_mcts_evaluation_contract,
)


def _checkpoint(
    tmp_path,
    *,
    history_size: int = 1,
    horizon: int = 4,
    value_action_count: int = 8,
):
    checkpoint = tmp_path / "final"
    (checkpoint / "wm_predictor").mkdir(parents=True)
    (checkpoint / "value_head").mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "wm_predictor" / "config.json").write_text(
        json.dumps(
            {
                "grid_tokens": 16,
                "emb_dim": 8,
                "action_dim": 8,
                "history_size": history_size,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    torch.save({}, checkpoint / "state_proj.pt")
    torch.save({}, checkpoint / "wm_predictor" / "predictor.pt")
    torch.save(
        {"net.2.weight": torch.empty(value_action_count, 8)},
        checkpoint / "value_head" / "value_head.pt",
    )
    torch.save(
        {
            "step": 17,
            "epoch": 2,
            "epoch_complete": True,
            "training_invariants": {
                "objective": "dino_grid",
                "history_size": history_size,
                "prediction_horizon": horizon,
                "value_objective": SFT2_VALUE_OBJECTIVE,
            },
        },
        checkpoint / "training_state.pt",
    )
    return checkpoint


def test_contract_reads_h1_and_prediction_horizon_from_checkpoint(tmp_path) -> None:
    contract = load_sft2_mcts_evaluation_contract(_checkpoint(tmp_path))

    assert contract.history_size == 1
    assert contract.prediction_horizon == 4
    assert contract.action_count == 8
    assert contract.step == 17
    assert contract.epoch == 2


def test_contract_rejects_non_h1_checkpoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="history_size=1"):
        load_sft2_mcts_evaluation_contract(
            _checkpoint(tmp_path, history_size=2)
        )


def test_contract_rejects_value_head_action_count_mismatch(tmp_path) -> None:
    with pytest.raises(ValueError, match="action counts disagree"):
        load_sft2_mcts_evaluation_contract(
            _checkpoint(tmp_path, value_action_count=7)
        )


def test_eval_entry_uses_checkpoint_k_and_balanced_dataset_seeds(tmp_path) -> None:
    contract = load_sft2_mcts_evaluation_contract(_checkpoint(tmp_path))
    args = parse_args(
        [
            "--sft2-checkpoint", str(contract.checkpoint),
            "--env-url", "http://env",
            "--output-dir", str(tmp_path / "eval"),
            "--eval-sets", "base", "common_sense",
            "--split", "test",
            "--episodes-per-eval-set", "2",
            "--seed-offset", "1",
            "--max-steps", "20",
            "--temperature", "0.7",
            "--top-p", "0.95",
            "--max-response-tokens", "512",
            "--num-simulations", "64",
            "--exploration-constant", "1.0",
            "--tensor-parallel-size", "4",
            "--planner-device", "cuda:0",
        ]
    )

    rollout_argv = build_rollout_argv(args, contract)

    horizon_index = rollout_argv.index("--planning-horizon")
    episode_index = rollout_argv.index("--num-episodes")
    assert rollout_argv[horizon_index + 1] == "4"
    assert rollout_argv[episode_index + 1] == "4"
    assert "--seed-per-eval-set" in rollout_argv
    assert rollout_argv[rollout_argv.index("--planning-search-mode") + 1] == "mcts"


def test_eval_entry_requires_at_least_one_simulation_per_root_action(tmp_path) -> None:
    contract = load_sft2_mcts_evaluation_contract(_checkpoint(tmp_path))
    args = parse_args(
        [
            "--sft2-checkpoint", str(contract.checkpoint),
            "--env-url", "http://env",
            "--output-dir", str(tmp_path / "eval"),
            "--eval-sets", "base",
            "--split", "eval",
            "--episodes-per-eval-set", "1",
            "--seed-offset", "1",
            "--max-steps", "20",
            "--temperature", "0.7",
            "--top-p", "0.95",
            "--max-response-tokens", "512",
            "--num-simulations", "7",
            "--exploration-constant", "1.0",
            "--tensor-parallel-size", "1",
            "--planner-device", "cuda:0",
        ]
    )

    with pytest.raises(ValueError, match="visit every root action"):
        build_rollout_argv(args, contract)


def test_eval_resume_forwards_rollout_resume_flag(tmp_path) -> None:
    contract = load_sft2_mcts_evaluation_contract(_checkpoint(tmp_path))
    args = parse_args(
        [
            "--sft2-checkpoint", str(contract.checkpoint),
            "--env-url", "http://env",
            "--output-dir", str(tmp_path / "eval"),
            "--resume",
            "--eval-sets", "base",
            "--split", "test",
            "--episodes-per-eval-set", "30",
            "--seed-offset", "31",
            "--max-steps", "20",
            "--temperature", "0.7",
            "--top-p", "0.95",
            "--max-response-tokens", "512",
            "--num-simulations", "100",
            "--exploration-constant", "1.0",
            "--tensor-parallel-size", "1",
            "--planner-device", "cuda:0",
        ]
    )

    assert "--resume-existing-rollouts" in build_rollout_argv(args, contract)


def test_eval_resume_requires_exact_existing_contract(tmp_path) -> None:
    output = tmp_path / "eval"
    metadata = {"history_size": 1, "prediction_horizon": 4}
    write_or_validate_contract(output, metadata, resume=False)
    write_or_validate_contract(output, metadata, resume=True)

    with pytest.raises(ValueError, match="does not match"):
        write_or_validate_contract(
            output,
            {"history_size": 1, "prediction_horizon": 3},
            resume=True,
        )


def test_eval_fresh_run_rejects_existing_contract(tmp_path) -> None:
    output = tmp_path / "eval"
    metadata = {"history_size": 1, "prediction_horizon": 4}
    write_or_validate_contract(output, metadata, resume=False)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_or_validate_contract(output, metadata, resume=False)
