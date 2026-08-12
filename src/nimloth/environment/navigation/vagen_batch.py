"""上游 VAGEN 单-session async client 的同步 batch 适配。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from concurrent.futures import Future
from threading import Thread
from typing import Any, TypeVar

_T = TypeVar("_T")


class VAGENBatchEnvClient:
    """为每个 environment identity 持有独立的上游 VAGEN session。"""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        retries: int = 6,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("VAGEN base_url must be non-empty")
        if timeout <= 0:
            raise ValueError("VAGEN timeout must be positive")
        if retries < 0:
            raise ValueError("VAGEN retries must be non-negative")
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._retries = int(retries)
        self._clients: dict[str, Any] = {}
        self._loop = asyncio.new_event_loop()
        self._loop_thread = Thread(
            target=self._run_loop,
            name="nimloth-vagen-env-client",
            daemon=True,
        )
        self._loop_thread.start()

    def create_environments_batch(
        self,
        ids2configs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        from vagen.envs_remote import GymImageEnvClient

        new_clients: dict[str, Any] = {}
        for env_id, config in ids2configs.items():
            if not isinstance(env_id, str) or not env_id:
                raise ValueError("environment id must be a non-empty string")
            if env_id in self._clients or env_id in new_clients:
                raise ValueError(f"environment already exists: {env_id}")
            if not isinstance(config, Mapping):
                raise ValueError(f"environment config must be a mapping: {env_id}")
            new_clients[env_id] = GymImageEnvClient(
                {
                    **dict(config),
                    "base_urls": [self._base_url],
                    "timeout": self._timeout,
                    "retries": self._retries,
                }
            )
        self._clients.update(new_clients)

    def reset_batch(
        self,
        ids2seeds: Mapping[str, int],
    ) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        clients = self._selected_clients(ids2seeds)

        async def reset_all():
            return await asyncio.gather(
                *(client.reset(int(ids2seeds[env_id])) for env_id, client in clients)
            )

        rows = self._run(reset_all())
        return {
            env_id: row
            for (env_id, _client), row in zip(clients, rows, strict=True)
        }

    def get_system_prompts_batch(
        self,
        env_ids: Sequence[str],
    ) -> dict[str, str]:
        clients = self._selected_clients(env_ids)

        async def prompt_all():
            return await asyncio.gather(
                *(client.system_prompt() for _env_id, client in clients)
            )

        rows = self._run(prompt_all())
        result: dict[str, str] = {}
        for (env_id, _client), row in zip(clients, rows, strict=True):
            text = row.get("obs_str") if isinstance(row, Mapping) else None
            result[env_id] = text if isinstance(text, str) else ""
        return result

    def step_batch(
        self,
        ids2actions: Mapping[str, str],
    ) -> dict[str, tuple[dict[str, Any], float, bool, dict[str, Any]]]:
        clients = self._selected_clients(ids2actions)

        async def step_all():
            return await asyncio.gather(
                *(
                    client.step(str(ids2actions[env_id]))
                    for env_id, client in clients
                )
            )

        rows = self._run(step_all())
        return {
            env_id: row
            for (env_id, _client), row in zip(clients, rows, strict=True)
        }

    def close_batch(self, env_ids: Sequence[str] | None = None) -> None:
        selected_ids = list(self._clients) if env_ids is None else list(env_ids)
        clients = self._selected_clients(selected_ids)

        async def close_all():
            return await asyncio.gather(
                *(client.close() for _env_id, client in clients),
                return_exceptions=True,
            )

        results = self._run(close_all())
        for env_id, _client in clients:
            self._clients.pop(env_id, None)
        failures = [value for value in results if isinstance(value, BaseException)]
        if not self._clients:
            self._stop_loop()
        if failures:
            raise RuntimeError(
                f"failed to close {len(failures)} VAGEN environment session(s)"
            ) from failures[0]

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, awaitable: Awaitable[_T]) -> _T:
        if not self._loop_thread.is_alive():
            if hasattr(awaitable, "close"):
                awaitable.close()  # type: ignore[attr-defined]
            raise RuntimeError("VAGEN environment client loop is closed")
        future: Future[_T] = asyncio.run_coroutine_threadsafe(
            awaitable,
            self._loop,
        )
        return future.result()

    def _stop_loop(self) -> None:
        if not self._loop_thread.is_alive():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join()
        self._loop.close()

    def _selected_clients(
        self,
        values: Mapping[str, Any] | Sequence[str],
    ) -> list[tuple[str, Any]]:
        env_ids = list(values) if not isinstance(values, Mapping) else list(values.keys())
        missing = [env_id for env_id in env_ids if env_id not in self._clients]
        if missing:
            raise KeyError(f"unknown VAGEN environment ids: {missing}")
        return [(env_id, self._clients[env_id]) for env_id in env_ids]


__all__ = ["VAGENBatchEnvClient"]
