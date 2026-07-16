"""Correctness tests for synchronized online RL rollout."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image

from nimloth.training.rl.rollout import (
    ACTION_NAMES,
    EnvRolloutCollector,
    RolloutTrajectory,
    build_nimloth_policy_messages,
    sample_action_from_logits,
    validate_rollout_trajectory,
)


def test_policy_prompt_uses_real_windowed_images() -> None:
    messages, images = build_nimloth_policy_messages(
        ["obs0.png", "obs1.png", "obs2.png"],
        "Find the chair.",
        ["moveahead", "rotateleft"],
        history_window=1,
    )
    assert images == ["obs1.png", "obs2.png"]
    image_contents = [
        part["image"]
        for message in messages
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("type") == "image"
    ]
    assert image_contents == ["obs1.png", "obs2.png"]
    assert "<|action_(5)|>" in messages[2]["content"][0]["text"]


def test_policy_prompt_rejects_misaligned_history() -> None:
    with pytest.raises(ValueError, match="one more image"):
        build_nimloth_policy_messages(
            ["obs0.png"], "Find it.", ["moveahead"], history_window=4
        )


def test_sampling_is_deterministic_and_temperature_scaled() -> None:
    logits = torch.arange(8, dtype=torch.float32)
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)
    action_a, log_probs_a = sample_action_from_logits(
        logits, temperature=0.5, top_p=0.9, generator=generator_a
    )
    action_b, log_probs_b = sample_action_from_logits(
        logits, temperature=0.5, top_p=0.9, generator=generator_b
    )
    assert action_a == action_b
    assert log_probs_a == log_probs_b
    assert torch.allclose(
        torch.tensor(log_probs_a), torch.log_softmax(logits / 0.5, dim=-1)
    )


def test_trajectory_validation_rejects_nonfinite_log_probs() -> None:
    trajectory = RolloutTrajectory(
        record_id="bad",
        image_paths=["before.png", "after.png"],
        action_indices=[0],
        action_names=["moveahead"],
        action_log_probs=[[float("nan")] * 8],
        nav_instruction="Move.",
        split="train",
    )
    with pytest.raises(ValueError, match="non-finite"):
        validate_rollout_trajectory(trajectory)


def test_resume_seed_cursor_accounts_for_completed_rollouts() -> None:
    collector = EnvRolloutCollector(
        None,
        None,
        "http://env",
        None,
        seed_offset=100,
        eval_sets=("base_train",),
        split="train",
    )
    collector.set_resume_iteration(
        start_iteration=6,
        envs_per_iteration=8,
        validation_enabled=True,
        validation_interval=2,
        validation_envs=3,
    )
    assert collector._ep_counter == 100 + 5 * 8 + 2 * 3


class _FakeEnvClient:
    def __init__(self, call_log: Path) -> None:
        self.call_log = call_log

    def _record(self, operation: str) -> None:
        with self.call_log.open("a", encoding="utf-8") as stream:
            stream.write(operation + "\n")

    def create_environments_batch(self, configs) -> None:
        self._record("create")

    def get_system_prompts_batch(self, env_ids):
        self._record("prompt")
        return {env_ids[0]: "Move near the couch."}

    def reset_batch(self, seeds):
        self._record("reset")
        env_id = next(iter(seeds))
        return {env_id: (Image.new("RGB", (4, 4), "black"), {})}

    def step_batch(self, actions):
        self._record("step")
        env_id = next(iter(actions))
        return {
            env_id: (
                Image.new("RGB", (4, 4), "white"),
                10.0,
                True,
                {"last_action_success": True},
            )
        }

    def close_batch(self, env_ids) -> None:
        self._record("close")


class _ForbiddenEnvClient:
    def __getattr__(self, name):
        raise AssertionError(f"nonzero rank accessed env client method {name}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _distributed_collect_worker(rank: int, world: int, port: int, root: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
    )
    try:
        import nimloth.training.rl.distributed_rollout as distributed_module
        from nimloth.training.rl.distributed_rollout import DistributedEnvRolloutCollector

        def fake_distribution(*args, **kwargs):
            logits = torch.arange(8, dtype=torch.float32)
            return logits, torch.log_softmax(logits, dim=-1)

        distributed_module.compute_nimloth_action_distribution = fake_distribution
        collector = DistributedEnvRolloutCollector(
            object(),
            object(),
            "http://env",
            torch.device("cpu"),
            seed_offset=7,
            temperature=0.7,
            top_p=0.95,
            eval_sets=("base_train",),
            split="train",
            history_window=2,
        )
        call_log = Path(root) / "env_calls.txt"
        collector._client = _FakeEnvClient(call_log) if rank == 0 else _ForbiddenEnvClient()
        trajectories = collector.collect(
            num_episodes=1,
            max_steps_per_episode=1,
            output_dir=Path(root) / "rollout",
        )
        result_path = Path(root) / f"rank_{rank}.json"
        result_path.write_text(
            json.dumps([trajectory.to_record() for trajectory in trajectories], sort_keys=True),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_distributed_collector_rank0_env_and_identical_trajectories(tmp_path: Path) -> None:
    port = _free_port()
    mp.spawn(
        _distributed_collect_worker,
        args=(2, port, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rank0 = (tmp_path / "rank_0.json").read_text(encoding="utf-8")
    rank1 = (tmp_path / "rank_1.json").read_text(encoding="utf-8")
    assert rank0 == rank1
    records = json.loads(rank0)
    assert len(records) == 1
    assert len(records[0]["image_paths"]) == 2
    assert len(records[0]["action_log_probs"][0]) == len(ACTION_NAMES)
    calls = (tmp_path / "env_calls.txt").read_text(encoding="utf-8").splitlines()
    assert calls == ["create", "prompt", "reset", "step", "close"]
    assert (tmp_path / "rollout" / "trajectories.jsonl").is_file()
