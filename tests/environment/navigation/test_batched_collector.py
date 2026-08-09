from __future__ import annotations

from pathlib import Path

from PIL import Image

from nimloth.agent import PolicyDecision, PolicyState, PolicyTokenTrace
from nimloth.environment.navigation.collector import (
    VAGENBatchedNavigationRolloutCollector,
)


def _observation(env_id: str, step: int) -> dict:
    image = Image.new("RGB", (2, 1))
    image.putpixel((0, 0), (0, 0, 0))
    image.putpixel((1, 0), (255, step, 1))
    return {
        "obs_str": f"Human Instruction: task {env_id}\nstep {step} <image>",
        "image": image,
    }


class _Client:
    def __init__(self) -> None:
        self.created: list[tuple[str, ...]] = []
        self.step_batch_sizes: list[int] = []
        self.closed: list[tuple[str, ...]] = []
        self.steps: dict[str, int] = {}

    def create_environments_batch(self, ids2configs):  # type: ignore[no-untyped-def]
        self.created.append(tuple(ids2configs))
        self.steps.update({env_id: 0 for env_id in ids2configs})

    def get_system_prompts_batch(self, env_ids):  # type: ignore[no-untyped-def]
        return {env_id: "Navigate." for env_id in env_ids}

    def reset_batch(self, ids2seeds):  # type: ignore[no-untyped-def]
        return {
            env_id: (_observation(env_id, 0), {"seed": seed})
            for env_id, seed in ids2seeds.items()
        }

    def step_batch(self, ids2actions):  # type: ignore[no-untyped-def]
        self.step_batch_sizes.append(len(ids2actions))
        results = {}
        for env_id, response in ids2actions.items():
            assert response.startswith("<think>")
            self.steps[env_id] += 1
            limit = 1 if env_id.endswith("000001") else 2
            done = self.steps[env_id] >= limit
            results[env_id] = (
                _observation(env_id, self.steps[env_id]),
                10.0 if done else 0.0,
                done,
                {
                    "last_action_success": True,
                    "task_success": done,
                },
            )
        return results

    def close_batch(self, env_ids):  # type: ignore[no-untyped-def]
        self.closed.append(tuple(env_ids))


class _BatchPolicy:
    prompt_mode = "response"
    credit_assignment = "turn"

    def __init__(self) -> None:
        self.action_batch_sizes: list[int] = []
        self.terminal_batch_sizes: list[int] = []

    def reset_episode(self) -> None:
        pass

    def select_actions(self, prompts):  # type: ignore[no-untyped-def]
        self.action_batch_sizes.append(len(prompts))
        return tuple(
            PolicyDecision(
                action_index=0,
                action_log_probs=(0.0, *([float("-inf")] * 7)),
                response=(
                    "<think>real batch cot</think><|latent_state|>"
                    "<|action_start|><|action_(0)|><|action_end|>"
                ),
                token_trace=PolicyTokenTrace(
                    token_ids=(5, 10, 20),
                    old_log_probs=(-0.2, 0.0, None),
                    loss_mask=(True, True, False),
                    token_roles=("reasoning", "action", "injected"),
                    action_token_ids=tuple(range(10, 18)),
                    reasoning_text="real batch cot",
                    finish_reason="stop",
                ),
            )
            for _ in prompts
        )

    def generate_states(self, prompts):  # type: ignore[no-untyped-def]
        self.terminal_batch_sizes.append(len(prompts))
        return tuple(
            PolicyState(
                assistant_prefix=(
                    "<think>real terminal cot</think><|latent_state|>"
                    "<|action_start|>"
                )
            )
            for _ in prompts
        )


def test_batched_collector_steps_only_active_envs_and_persists_prefix(
    tmp_path: Path,
) -> None:
    policy = _BatchPolicy()
    client = _Client()
    collector = VAGENBatchedNavigationRolloutCollector(
        policy=policy,  # type: ignore[arg-type]
        env_url="http://unused",
        client=client,
        seed_offset=1,
        temperature=0.7,
        top_p=0.95,
        eval_sets=("base_train",),
        split="train",
        latent_token_count=1,
        max_episode_attempts=1,
    )

    trajectories = collector.collect(
        num_episodes=2,
        max_steps_per_episode=2,
        output_dir=tmp_path,
    )

    assert [item.record_id for item in trajectories] == ["rl_000001", "rl_000002"]
    assert [item.num_steps for item in trajectories] == [1, 2]
    assert [item.instruction for item in trajectories] == [
        "task rl_000001",
        "task rl_000002",
    ]
    assert policy.action_batch_sizes == [2, 1]
    assert policy.terminal_batch_sizes == [1, 1]
    assert client.step_batch_sizes == [2, 1]
    assert client.created == [("rl_000001", "rl_000002")]
    assert client.closed == [("rl_000001",), ("rl_000002",)]
    assert (tmp_path / "trajectories.jsonl").is_file()
    assert len(list((tmp_path / "images").glob("*.png"))) == 5
