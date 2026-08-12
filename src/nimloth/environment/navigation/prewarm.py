"""在加载 rollout 模型前验证真实 VAGEN navigation 生命周期。"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from nimloth.environment.navigation.vagen import (
    NAVIGATION_REQUEST_TIMEOUT_SECONDS,
    navigation_image_dynamic_range,
    navigation_environment_config,
    observation_image,
    observation_text,
)


@dataclass(frozen=True)
class NavigationPrewarmResult:
    """一次 create -> reset -> close 门禁的可审计结果。"""

    env_id: str
    eval_set: str
    seed: int
    elapsed_seconds: float
    observation_chars: int
    image_width: int
    image_height: int
    image_dynamic_range: int


def prewarm_navigation_client(
    client: Any,
    *,
    eval_set: str,
    seed: int,
    env_id: str,
) -> NavigationPrewarmResult:
    """要求 navigation 服务完成真实创建、prompt、reset 和资源释放。"""

    started_at = time.monotonic()
    client.create_environments_batch(
        {env_id: navigation_environment_config(eval_set, latent_token_count=16)}
    )
    try:
        raw_observation, _ = client.reset_batch({env_id: seed})[env_id]
        prompts = client.get_system_prompts_batch([env_id])
        prompt = prompts.get(env_id)
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError("navigation prewarm returned an empty system prompt")
        text = observation_text(raw_observation)
        image = observation_image(raw_observation)
        return NavigationPrewarmResult(
            env_id=env_id,
            eval_set=eval_set,
            seed=int(seed),
            elapsed_seconds=round(time.monotonic() - started_at, 3),
            observation_chars=len(text),
            image_width=image.width,
            image_height=image.height,
            image_dynamic_range=navigation_image_dynamic_range(image),
        )
    finally:
        client.close_batch([env_id])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prewarm and validate one VAGEN navigation environment",
    )
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=NAVIGATION_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument("--env-id", default="nimloth-navigation-prewarm")
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= NAVIGATION_REQUEST_TIMEOUT_SECONDS:
        raise ValueError(
            "timeout-seconds must be between 1 and "
            f"{NAVIGATION_REQUEST_TIMEOUT_SECONDS}"
        )

    from nimloth.environment.navigation.vagen_batch import VAGENBatchEnvClient

    result = prewarm_navigation_client(
        VAGENBatchEnvClient(
            base_url=args.env_url,
            timeout=args.timeout_seconds,
        ),
        eval_set=args.eval_set,
        seed=args.seed,
        env_id=args.env_id,
    )
    print(json.dumps({"status": "ENV_PREWARM_OK", **asdict(result)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NavigationPrewarmResult",
    "prewarm_navigation_client",
]
