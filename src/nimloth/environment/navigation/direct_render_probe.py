"""直接验证一张已分配GPU能否产生有效AI2-THOR画面。"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from nimloth.environment.navigation.vagen import (
    navigation_image_dynamic_range,
    validate_navigation_image,
)


@dataclass(frozen=True)
class DirectRenderProbeResult:
    """一次Controller create/reset/stop门禁的可审计结果。"""

    scene: str
    elapsed_seconds: float
    image_width: int
    image_height: int
    image_dynamic_range: int
    gpu_device: int
    cuda_visible_devices: str
    cuda_vulkan_mapping: dict[str, int]


def probe_navigation_render(
    controller_factory: Callable[..., Any],
    *,
    scene: str = "FloorPlan1",
    gpu_device: int = 0,
) -> DirectRenderProbeResult:
    """创建Controller并要求reset返回非纯色RGB frame。"""

    if gpu_device < 0:
        raise ValueError("gpu_device must be non-negative")
    started_at = time.monotonic()
    controller = controller_factory(
        agentMode="default",
        gridSize=0.1,
        visibilityDistance=10,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=255,
        height=255,
        fieldOfView=100,
        gpu_device=gpu_device,
        server_timeout=60,
        server_start_timeout=60,
    )
    try:
        event = controller.reset(scene=scene)
        frame = getattr(event, "frame", None)
        if frame is None:
            raise RuntimeError("AI2-THOR render probe returned no frame")
        image = validate_navigation_image(Image.fromarray(frame))
    finally:
        controller.stop()

    mapping_path = Path.home() / ".ai2thor" / "cuda-vulkan-mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f"invalid CUDA/Vulkan mapping: {mapping!r}")
    if str(gpu_device) not in mapping:
        raise RuntimeError(
            f"CUDA/Vulkan mapping has no GPU ordinal {gpu_device}: {mapping!r}"
        )
    return DirectRenderProbeResult(
        scene=scene,
        elapsed_seconds=round(time.monotonic() - started_at, 3),
        image_width=image.width,
        image_height=image.height,
        image_dynamic_range=navigation_image_dynamic_range(image),
        gpu_device=gpu_device,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        cuda_vulkan_mapping={str(key): int(value) for key, value in mapping.items()},
    )


def main() -> int:
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    parser = argparse.ArgumentParser(
        description="Validate one allocated AI2-THOR GPU ordinal",
    )
    parser.add_argument("--gpu-device", type=int, required=True)
    args = parser.parse_args()
    result = probe_navigation_render(
        lambda **kwargs: Controller(platform=CloudRendering, **kwargs),
        gpu_device=args.gpu_device,
    )
    print(json.dumps({"status": "AI2THOR_RENDER_OK", **asdict(result)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DirectRenderProbeResult", "probe_navigation_render"]
