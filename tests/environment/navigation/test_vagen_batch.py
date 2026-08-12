from __future__ import annotations

import asyncio
import sys
from types import ModuleType

import pytest

from nimloth.environment.navigation.vagen_batch import VAGENBatchEnvClient


class _FakeAsyncClient:
    instances: list["_FakeAsyncClient"] = []

    def __init__(self, config):  # type: ignore[no-untyped-def]
        self.config = dict(config)
        self.seed: int | None = None
        self.closed = False
        self.loop_ids: list[int] = []
        self.instances.append(self)

    async def reset(self, seed: int):
        self.loop_ids.append(id(asyncio.get_running_loop()))
        self.seed = seed
        return (
            {
                "obs_str": f"Human Instruction: seed {seed}\n<image>",
                "multi_modal_input": {"<image>": []},
            },
            {"seed": seed},
        )

    async def system_prompt(self):
        self.loop_ids.append(id(asyncio.get_running_loop()))
        if self.seed is None:
            raise RuntimeError("system_prompt called before reset")
        return {"obs_str": f"prompt {self.seed}"}

    async def step(self, action_str: str):
        self.loop_ids.append(id(asyncio.get_running_loop()))
        return (
            {"obs_str": f"after {action_str} <image>"},
            float(self.seed or 0),
            True,
            {"action": action_str},
        )

    async def close(self):
        self.loop_ids.append(id(asyncio.get_running_loop()))
        self.closed = True


@pytest.fixture(autouse=True)
def _fake_vagen_client(monkeypatch):  # type: ignore[no-untyped-def]
    _FakeAsyncClient.instances.clear()
    envs_remote = ModuleType("vagen.envs_remote")
    envs_remote.GymImageEnvClient = _FakeAsyncClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vagen.envs_remote", envs_remote)


def test_batch_adapter_preserves_identity_and_upstream_lifecycle() -> None:
    client = VAGENBatchEnvClient(base_url="http://env", timeout=500)
    client.create_environments_batch(
        {
            "a": {"prompt_format": "nimloth", "latent_token_count": 16},
            "b": {"prompt_format": "nimloth", "latent_token_count": 16},
        }
    )

    reset = client.reset_batch({"a": 3, "b": 5})
    prompts = client.get_system_prompts_batch(["a", "b"])
    steps = client.step_batch({"a": "action-a", "b": "action-b"})
    client.close_batch(["a", "b"])

    assert list(reset) == ["a", "b"]
    assert prompts == {"a": "prompt 3", "b": "prompt 5"}
    assert steps["a"][1:] == (3.0, True, {"action": "action-a"})
    assert steps["b"][1:] == (5.0, True, {"action": "action-b"})
    assert all(instance.config["base_urls"] == ["http://env"] for instance in _FakeAsyncClient.instances)
    assert all(instance.config["timeout"] == 500 for instance in _FakeAsyncClient.instances)
    assert all(instance.closed for instance in _FakeAsyncClient.instances)
    assert all(len(set(instance.loop_ids)) == 1 for instance in _FakeAsyncClient.instances)


def test_batch_adapter_rejects_unknown_or_duplicate_identity() -> None:
    client = VAGENBatchEnvClient(base_url="http://env", timeout=1)
    client.create_environments_batch({"a": {}})
    with pytest.raises(ValueError, match="already exists"):
        client.create_environments_batch({"a": {}})
    with pytest.raises(KeyError, match="unknown"):
        client.reset_batch({"missing": 0})


def test_sync_batch_calls_work_from_thread_with_running_event_loop() -> None:
    client = VAGENBatchEnvClient(base_url="http://env", timeout=1)
    client.create_environments_batch({"a": {}})

    async def outer():
        reset = client.reset_batch({"a": 7})
        prompt = client.get_system_prompts_batch(["a"])
        client.close_batch(["a"])
        return reset, prompt

    reset, prompt = asyncio.run(outer())
    assert reset["a"][1] == {"seed": 7}
    assert prompt == {"a": "prompt 7"}
