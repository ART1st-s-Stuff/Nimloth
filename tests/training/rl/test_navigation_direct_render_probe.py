from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from nimloth.environment.navigation.direct_render_probe import (
    probe_navigation_render,
)


class _Controller:
    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame
        self.stopped = False

    def reset(self, *, scene: str) -> SimpleNamespace:
        assert scene == "FloorPlan1"
        return SimpleNamespace(frame=self._frame)

    def stop(self) -> None:
        self.stopped = True


def test_probe_accepts_non_uniform_frame_and_records_mapping(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    mapping_dir = tmp_path / ".ai2thor"
    mapping_dir.mkdir()
    (mapping_dir / "cuda-vulkan-mapping.json").write_text(
        json.dumps({"0": 6})
    )
    frame = np.zeros((255, 255, 3), dtype=np.uint8)
    frame[:, 1:, 1] = 127
    controller = _Controller(frame)

    result = probe_navigation_render(lambda **_: controller)

    assert result.image_dynamic_range == 127
    assert result.gpu_device == 0
    assert result.cuda_visible_devices == "6"
    assert result.cuda_vulkan_mapping == {"0": 6}
    assert controller.stopped


def test_probe_rejects_uniform_frame_and_stops_controller(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    mapping_dir = tmp_path / ".ai2thor"
    mapping_dir.mkdir()
    (mapping_dir / "cuda-vulkan-mapping.json").write_text("{\"0\": 0}")
    controller = _Controller(np.zeros((255, 255, 3), dtype=np.uint8))

    with pytest.raises(RuntimeError, match="uniform image"):
        probe_navigation_render(lambda **_: controller)

    assert controller.stopped


def test_probe_requires_selected_ordinal_in_mapping(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    mapping_dir = tmp_path / ".ai2thor"
    mapping_dir.mkdir()
    (mapping_dir / "cuda-vulkan-mapping.json").write_text("{\"0\": 1}")
    frame = np.zeros((255, 255, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(255, dtype=np.uint8)[:, None]
    controller = _Controller(frame)

    with pytest.raises(RuntimeError, match="no GPU ordinal 1"):
        probe_navigation_render(lambda **_: controller, gpu_device=1)

    assert controller.stopped
