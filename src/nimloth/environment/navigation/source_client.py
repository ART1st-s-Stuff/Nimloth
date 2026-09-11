"""Transport for the original VAGEN BatchEnvironmentServer API."""
from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "__pil_image__" in value:
            raw = base64.b64decode(value["__pil_image__"])
            with Image.open(io.BytesIO(raw)) as image:
                return image.convert("RGB")
        if "__numpy_array__" in value:
            try:
                import numpy as np
            except ImportError as error:  # pragma: no cover - runtime dependency
                raise RuntimeError("numpy observation decoding requires numpy") from error
            array = value["__numpy_array__"]
            return np.array(array["data"], dtype=array["dtype"]).reshape(
                array["shape"]
            )
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


class LegacyVAGENBatchClient:
    """Minimal client for the evidence-verified legacy batch endpoints."""

    def __init__(self, base_url: str, *, timeout: float = 500.0) -> None:
        if not base_url.strip():
            raise ValueError("source environment URL must be non-empty")
        if timeout < 500:
            raise ValueError("source environment timeout must be at least 500 seconds")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self._env_ids: set[str] = set()

    def _request(self, endpoint: str, *, method: str = "POST", data: Any = None) -> Any:
        try:
            import requests
        except ImportError as error:  # pragma: no cover - runtime dependency
            raise RuntimeError("legacy VAGEN client requires requests") from error
        url = f"{self.base_url}/{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=self.timeout)
        else:
            response = requests.post(url, json=data, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def check_server_health(self) -> dict[str, Any]:
        value = self._request("health", method="GET")
        if not isinstance(value, dict):
            raise TypeError("source server health response is not a mapping")
        return value

    def create_environments_batch(self, ids2configs: dict[str, dict[str, Any]]) -> None:
        duplicate = self._env_ids & set(ids2configs)
        if duplicate:
            raise ValueError(f"source environments already exist: {sorted(duplicate)}")
        value = self._request("environments", data={"ids2configs": ids2configs})
        if value.get("success") is not True:
            raise RuntimeError(f"source environment creation failed: {value!r}")
        self._env_ids.update(ids2configs)

    def reset_batch(self, ids2seeds: dict[str, int]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        value = self._request("batch/reset", data={"ids2seeds": ids2seeds})
        rows = value.get("results", {})
        return {
            env_id: (_decode_value(row[0]), _decode_value(row[1]))
            for env_id, row in rows.items()
        }

    def get_system_prompts_batch(self, env_ids: list[str]) -> dict[str, str]:
        value = self._request("batch/system_prompt", data={"env_ids": env_ids})
        return {str(key): str(item) for key, item in value.get("system_prompts", {}).items()}

    def step_batch(self, ids2actions: dict[str, str]) -> dict[str, tuple[dict[str, Any], float, bool, dict[str, Any]]]:
        value = self._request("batch/step", data={"ids2actions": ids2actions})
        rows = value.get("results", {})
        return {
            env_id: (
                _decode_value(row[0]),
                float(row[1]),
                bool(row[2]),
                _decode_value(row[3]),
            )
            for env_id, row in rows.items()
        }

    def close_batch(self, env_ids: list[str] | None = None) -> None:
        selected = sorted(self._env_ids if env_ids is None else set(env_ids))
        if not selected:
            return
        self._request("batch/close", data={"env_ids": selected})
        self._env_ids.difference_update(selected)
