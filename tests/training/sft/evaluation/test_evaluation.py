"""CPU interface checks; these do not execute VAGEN or establish policy quality."""
from dataclasses import replace

import pytest

from nimloth.training.sft.evaluation import (
    EvaluationConfig,
    cli,
    eval_direct,
    eval_wm,
    rollout,
)
from nimloth.training.sft.stage3.mcts_evaluation import SFT2MCTSEvaluationContract


def config(tmp_path, **overrides):
    values = {
        "mode": "direct", "checkpoint": tmp_path / "model", "env_url": "http://env",
        "output_dir": tmp_path / "output", "eval_sets": ("base", "common_sense"),
        "split": "test", "episodes_per_eval_set": 3, "seed_offset": 7, "max_steps": 20,
        "temperature": 0.0, "top_p": 1.0, "max_response_tokens": 100,
        "tensor_parallel_size": 1,
    }
    values.update(overrides)
    checkpoint = values["checkpoint"]
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text("{}")
    return EvaluationConfig(**values)


def test_direct_and_wm_share_episode_generation_and_stop_contract(tmp_path):
    direct = config(tmp_path)
    wm = replace(direct, mode="wm", num_simulations=32,
                 exploration_constant=1.0, planner_device="cuda:1")
    contract = SFT2MCTSEvaluationContract(direct.checkpoint.resolve(), 1, 4, 8, 17, 2)
    direct_args = rollout.parse_args(cli.build_rollout_argv(direct, None))
    wm_args = rollout.parse_args(cli.build_rollout_argv(wm, contract))
    for name in ("model", "env_url", "eval_sets", "split", "num_episodes",
                 "seed_offset", "seed_per_eval_set", "max_steps", "temperature",
                 "top_p", "max_response_tokens", "max_episode_attempts"):
        assert getattr(direct_args, name) == getattr(wm_args, name)
    assert direct_args.num_episodes == 6
    assert not direct_args.planner_enabled
    assert direct_args.wm_checkpoint is None
    assert wm_args.planner_enabled
    assert wm_args.planning_horizon == 4
    assert wm_args.planning_search_mode == "mcts"
    assert wm_args.wm_checkpoint == contract.wm_checkpoint
    assert wm_args.value_head_checkpoint == contract.value_head_checkpoint
    assert wm_args.state_proj_checkpoint == contract.state_proj_checkpoint


@pytest.mark.parametrize("overrides", [
    {"eval_sets": ("base_train",)}, {"eval_sets": ("base", "base")},
    {"split": "train"}, {"max_steps": 0}, {"episodes_per_eval_set": 0},
    {"temperature": float("nan")}, {"top_p": 0}, {"planner_device": "cuda:1"},
    {"mode": "wm"}, {"mode": "wm", "num_simulations": 8,
                       "exploration_constant": float("nan"), "planner_device": "cuda:1"},
])
def test_invalid_evaluation_fails_before_runtime(tmp_path, overrides):
    with pytest.raises(ValueError):
        config(tmp_path, **overrides)


def test_callable_dispatch_and_resume_contract(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "rollout_main", lambda argv: calls.append(argv) or 0)
    direct = config(tmp_path)
    assert eval_direct(direct) == 0
    assert len(calls) == 1
    assert (direct.output_dir / "evaluation_contract.json").exists()
    assert eval_direct(replace(direct, resume=True)) == 0
    with pytest.raises(ValueError, match="does not match"):
        eval_direct(replace(direct, resume=True, temperature=0.7))
    with pytest.raises(ValueError, match="does not match"):
        eval_direct(replace(direct, resume=True, max_pixels=65025))
    with pytest.raises(ValueError, match="mode"):
        eval_wm(direct)
    assert len(calls) == 2


def test_wm_dispatch_validates_checkpoint_and_preserves_search(tmp_path, monkeypatch):
    wm = config(tmp_path, mode="wm", planner_device="cuda:1", num_simulations=8,
                exploration_constant=1.0)
    contract = SFT2MCTSEvaluationContract(wm.checkpoint.resolve(), 1, 4, 8, 17, 2)
    contract.wm_checkpoint.mkdir()
    contract.value_head_checkpoint.mkdir()
    contract.state_proj_checkpoint.write_bytes(b"projector")
    loaded = []
    def load(checkpoint):
        loaded.append(checkpoint)
        return contract
    monkeypatch.setattr(cli, "load_sft2_mcts_evaluation_contract", load)
    calls = []
    monkeypatch.setattr(cli, "rollout_main", lambda argv: calls.append(argv) or 0)
    assert eval_wm(wm) == 0
    assert loaded == [wm.checkpoint]
    assert rollout.parse_args(calls[0]).planning_horizon == 4
    with pytest.raises(ValueError, match="every root action"):
        cli.build_rollout_argv(replace(wm, num_simulations=7), contract)


def test_legacy_rollout_helpers_use_canonical_implementation():
    from experiments.training.rl import rollout_env
    assert rollout_env.main is rollout.main
    assert rollout_env.summarize_eval_set_rollouts is rollout.summarize_eval_set_rollouts


@pytest.mark.parametrize("mode", ["direct", "wm"])
def test_resume_rejects_checkpoint_replaced_at_same_path(tmp_path, monkeypatch, mode):
    args = config(tmp_path)
    if mode == "wm":
        args = replace(args, mode="wm", num_simulations=8,
                       exploration_constant=1.0, planner_device="cuda:1")
        contract = SFT2MCTSEvaluationContract(args.checkpoint.resolve(), 1, 4, 8, 17, 2)
        contract.wm_checkpoint.mkdir()
        contract.value_head_checkpoint.mkdir()
        contract.state_proj_checkpoint.write_bytes(b"original projector")
        monkeypatch.setattr(cli, "load_sft2_mcts_evaluation_contract", lambda path: contract)
        changed = contract.state_proj_checkpoint
    else:
        changed = args.checkpoint / "model.safetensors"
        changed.write_bytes(b"original policy")
    calls = []
    monkeypatch.setattr(cli, "rollout_main", lambda argv: calls.append(argv) or 0)
    cli.run_evaluation(args)
    cli.run_evaluation(replace(args, resume=True))
    changed.write_bytes(b"replacement at the same path")
    with pytest.raises(ValueError, match="does not match"):
        cli.run_evaluation(replace(args, resume=True))
    assert len(calls) == 2
