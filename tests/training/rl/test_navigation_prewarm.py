from __future__ import annotations

from PIL import Image
import pytest

from nimloth.environment.navigation.prewarm import prewarm_navigation_client


class _FakeNavigationClient:
    def __init__(
        self,
        *,
        empty_prompt: bool = False,
        uniform_image: bool = False,
    ) -> None:
        self.empty_prompt = empty_prompt
        self.uniform_image = uniform_image
        self.calls: list[tuple[str, object]] = []

    def create_environments_batch(self, configs):
        self.calls.append(("create", configs))

    def get_system_prompts_batch(self, env_ids):
        self.calls.append(("prompt", env_ids))
        return {env_ids[0]: "" if self.empty_prompt else "navigation prompt"}

    def reset_batch(self, seeds):
        self.calls.append(("reset", seeds))
        env_id = next(iter(seeds))
        image = Image.new("RGB", (8, 6))
        if not self.uniform_image:
            image.putpixel((0, 0), (255, 128, 64))
        return {
            env_id: (
                {
                    "obs_str": "Human Instruction: move closer\n<image>",
                    "image": image,
                },
                {},
            )
        }

    def close_batch(self, env_ids):
        self.calls.append(("close", env_ids))


def test_navigation_prewarm_validates_real_lifecycle() -> None:
    client = _FakeNavigationClient()

    result = prewarm_navigation_client(
        client,
        eval_set="base_train",
        seed=3,
        env_id="prewarm",
    )

    assert result.env_id == "prewarm"
    assert result.eval_set == "base_train"
    assert result.seed == 3
    assert result.observation_chars > 0
    assert (result.image_width, result.image_height) == (8, 6)
    assert result.image_dynamic_range == 255
    assert [name for name, _ in client.calls] == [
        "create",
        "reset",
        "prompt",
        "close",
    ]
    create_config = client.calls[0][1]
    assert create_config["prewarm"]["eval_set"] == "base_train"
    assert create_config["prewarm"]["latent_token_count"] == 16


def test_navigation_prewarm_closes_environment_after_validation_failure() -> None:
    client = _FakeNavigationClient(empty_prompt=True)

    with pytest.raises(RuntimeError, match="empty system prompt"):
        prewarm_navigation_client(
            client,
            eval_set="base_train",
            seed=1,
            env_id="prewarm",
        )

    assert [name for name, _ in client.calls] == [
        "create",
        "reset",
        "prompt",
        "close",
    ]


def test_navigation_prewarm_rejects_uniform_render_and_closes() -> None:
    client = _FakeNavigationClient(uniform_image=True)

    with pytest.raises(RuntimeError, match="uniform image"):
        prewarm_navigation_client(
            client,
            eval_set="base",
            seed=1,
            env_id="prewarm",
        )

    assert [name for name, _ in client.calls] == [
        "create",
        "reset",
        "prompt",
        "close",
    ]
