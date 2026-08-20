from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/training/rl/rollout_env.py"
SPEC = importlib.util.spec_from_file_location("rollout_env_browser_wiring", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _args(**overrides):
    values = {
        "planner_enabled": False,
        "planning_search_mode": None,
        "split": "test",
        "seed_offset": 1,
        "seed_per_eval_set": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_policy_family_is_derived_from_real_behavior_route() -> None:
    assert MODULE.rollout_browser_policy_family(_args()) == "sft_policy"
    assert MODULE.rollout_browser_policy_family(
        _args(planner_enabled=True, planning_search_mode="mcts")
    ) == "sft2_mcts"
    assert MODULE.rollout_browser_policy_family(
        _args(planner_enabled=True, planning_search_mode="policy")
    ) == "planner_policy"


def test_eval_set_and_seed_alignment_matches_collector_order() -> None:
    eval_sets = ("base", "long_horizon")
    assert MODULE.rollout_browser_source_seed(
        _args(seed_offset=4, seed_per_eval_set=True), eval_sets, 3
    ) == ("long_horizon", 5)
    assert MODULE.rollout_browser_source_seed(
        _args(seed_offset=4, seed_per_eval_set=False), eval_sets, 3
    ) == ("long_horizon", 7)
