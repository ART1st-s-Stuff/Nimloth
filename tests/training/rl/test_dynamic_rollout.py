"""Correctness tests for synchronized online RL rollout."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from types import SimpleNamespace

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
    materialize_policy_images,
    sample_action_from_logits,
    validate_rl_policy_protocol,
    validate_rollout_trajectory,
)


def test_k8_inject_policy_protocol_is_supported() -> None:
    assert validate_rl_policy_protocol(SimpleNamespace(
        nimloth_latent_token_count=8,
        nimloth_latent_query_mode="inject",
    )) == 8
    with pytest.raises(ValueError, match="inject"):
        validate_rl_policy_protocol(SimpleNamespace(
            nimloth_latent_token_count=8,
            nimloth_latent_query_mode="generate",
        ))


def test_k8_pilot_uses_disjoint_fixed_heldout_protocol() -> None:
    import yaml

    config = yaml.safe_load(Path(
        "configs/training/rl/dynamic_fsdp_k8_pilot.yaml"
    ).read_text(encoding="utf-8"))
    assert config["rollout"]["eval_sets"] == ["base_train"]
    assert config["rl"]["iterations"] == 20
    assert config["rl"]["envs_per_iteration"] == 8
    assert config["rl"]["max_steps_per_episode"] == 20
    assert config["validation"] == {
        "enabled": True,
        "baseline": True,
        "interval": 5,
        "envs": 20,
        "max_steps_per_episode": 20,
        "eval_sets": ["base"],
        "seed_offset": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "history_window": 4,
        "env_timeout": 600,
    }


def test_k8_snapshot_is_immutable_and_omits_sft_optimizer(tmp_path: Path) -> None:
    from experiments.training.rl.prepare_k8_sft2_init import (
        ROOT_FILES,
        TREE_FILES,
        snapshot_checkpoint,
    )

    source = tmp_path / "latest"
    source.mkdir()
    for relative_name in (*ROOT_FILES, *TREE_FILES):
        path = source / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"content:{relative_name}".encode())
    torch.save({
        "step": 123,
        "epoch": 2,
        "latent_token_count": 8,
        "latent_query_mode": "inject",
        "base_model_path": "/base",
        "optimizer": {"huge": torch.ones(3)},
    }, source / "training_state.pt")

    output = tmp_path / "snapshot"
    manifest = snapshot_checkpoint(source, output)

    assert manifest["source_step"] == 123
    assert manifest["stable_during_copy"] is True
    assert (output / "SNAPSHOT_READY").is_file()
    state = torch.load(output / "training_state.pt", weights_only=False)
    assert state["latent_token_count"] == 8
    assert "optimizer" not in state
    with pytest.raises(FileExistsError):
        snapshot_checkpoint(source, output)


def test_zero_update_run_refuses_final_checkpoint() -> None:
    from nimloth.training.rl.trainer import _require_optimizer_progress

    with pytest.raises(RuntimeError, match="zero optimizer steps"):
        _require_optimizer_progress(0)
    _require_optimizer_progress(1)


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


def test_policy_prompt_renders_eight_query_tokens() -> None:
    messages, _ = build_nimloth_policy_messages(
        ["obs0.png", "obs1.png"],
        "Find the chair.",
        ["moveahead"],
        history_window=4,
        latent_token_count=8,
    )
    expected = "<|latent_state|>" + "".join(
        f"<|latent_state_{index}|>" for index in range(1, 8)
    )
    assistant_texts = [
        part["text"]
        for message in messages
        if message["role"] == "assistant"
        for part in message["content"]
    ]
    assert len(assistant_texts) == 2
    assert all(expected in text for text in assistant_texts)


def test_policy_image_paths_are_materialized_as_rgb(tmp_path: Path) -> None:
    path = tmp_path / "observation.png"
    Image.new("RGBA", (3, 2), (10, 20, 30, 40)).save(path)

    result = materialize_policy_images([str(path)])

    assert len(result) == 1
    assert isinstance(result[0], Image.Image)
    assert result[0].mode == "RGB"
    assert result[0].size == (3, 2)
    assert result[0].getpixel((0, 0)) == (10, 20, 30)
    # The returned copy must remain readable after the source file is closed.
    result[0].load()


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


def test_distributed_wrapper_preserves_env_timeout() -> None:
    from nimloth.training.rl.distributed_rollout import DistributedEnvRolloutCollector

    collector = EnvRolloutCollector(
        None,
        None,
        "http://env",
        None,
        eval_sets=("base_train",),
        split="train",
        env_timeout=37,
    )
    wrapped = DistributedEnvRolloutCollector.from_collector(collector)
    assert wrapped._env_timeout == 37


def test_collector_can_reset_fixed_heldout_seed_cursor() -> None:
    collector = EnvRolloutCollector(
        None,
        None,
        "http://env",
        None,
        seed_offset=1,
        eval_sets=("base",),
        split="validation",
    )
    collector._ep_counter = 99
    collector.reset_seed_cursor()
    assert collector._ep_counter == 1


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


def test_k8_hidden_encoding_extracts_full_query_block(monkeypatch, tmp_path: Path) -> None:
    from nimloth.latent.extraction import latent_state_tokens
    from nimloth.training.rl.trainer import encode_trajectory_hiddens

    latent_names = latent_state_tokens(8)
    token_id_map = {name: 101 + index for index, name in enumerate(latent_names)}
    input_ids = torch.tensor([[0, *[token_id_map[name] for name in latent_names], 9]])
    hidden = torch.arange(input_ids.numel() * 2, dtype=torch.float32).reshape(1, -1, 2)

    def fake_batch(items, processor, max_length, *, latent_token_count, **kwargs):
        assert latent_token_count == 8
        return {"input_ids": input_ids}

    class FakeModel:
        def __call__(self, **kwargs):
            return SimpleNamespace(hidden_states=(hidden,))

    monkeypatch.setattr(
        "nimloth.training.common.qwen_batch.build_qwen_batch", fake_batch
    )
    trajectory = RolloutTrajectory(
        record_id="k8",
        image_paths=[str(tmp_path / "a.png"), str(tmp_path / "b.png")],
        action_indices=[0],
        action_names=["moveahead"],
        action_log_probs=[[float(-torch.log(torch.tensor(8.0)))] * 8],
        nav_instruction="Move.",
        split="train",
        messages=[{"role": "system", "content": "Navigate."}],
    )
    states = encode_trajectory_hiddens(
        trajectory,
        FakeModel(),
        object(),
        token_id_map,
        torch.device("cpu"),
        latent_token_count=8,
    )
    assert len(states) == 2
    assert states[0].shape == (8, 2)
    assert torch.equal(states[0], hidden[0, 1:9])


def test_state_projector_loader_infers_k8_checkpoint_width(tmp_path: Path) -> None:
    from nimloth.training.rl.cli import load_state_projector_for_rl
    from nimloth.wm.state_proj import StateProjector

    expected = StateProjector(
        qwen_hidden_dim=3,
        lewm_emb_dim=4,
        projector_hidden_dim=5,
        latent_token_count=8,
    )
    checkpoint = tmp_path / "state_proj.pt"
    torch.save(expected.state_dict(), checkpoint)

    loaded = load_state_projector_for_rl(
        checkpoint,
        qwen_hidden_dim=3,
        lewm_emb_dim=4,
        latent_token_count=8,
    )
    assert loaded.input_dim == 24
    assert loaded.latent_token_count == 8
    assert all(
        torch.equal(expected.state_dict()[key], loaded.state_dict()[key])
        for key in expected.state_dict()
    )
    with pytest.raises(ValueError, match="input dim"):
        load_state_projector_for_rl(
            checkpoint,
            qwen_hidden_dim=3,
            lewm_emb_dim=4,
            latent_token_count=1,
        )


def test_value_head_loader_honors_checkpoint_hidden_width(tmp_path: Path) -> None:
    from nimloth.wm.value_head import ValueHead

    checkpoint = tmp_path / "value"
    expected = ValueHead(emb_dim=12, hidden_dim=5)
    expected.save_checkpoint(checkpoint)
    loaded = ValueHead.load_checkpoint(checkpoint, emb_dim=12)
    assert loaded.net[0].weight.shape == (5, 12)
    assert all(
        torch.equal(expected.state_dict()[key], loaded.state_dict()[key])
        for key in expected.state_dict()
    )


def test_validation_summary_requires_all_fixed_episodes() -> None:
    from nimloth.training.rl.trainer import summarize_validation_trajectories

    trajectories = [
        RolloutTrajectory(record_id="v1", success=True, reward=10.0, action_indices=[0]),
        RolloutTrajectory(record_id="v2", success=False, reward=0.0, action_indices=[0, 1]),
    ]
    metrics = summarize_validation_trajectories(trajectories, expected_episodes=2)
    assert metrics == {
        "val_success_rate": 0.5,
        "val_avg_reward": 5.0,
        "val_avg_steps": 1.5,
        "val_num_episodes": 2.0,
    }
    with pytest.raises(RuntimeError, match="expected 3"):
        summarize_validation_trajectories(trajectories, expected_episodes=3)


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
