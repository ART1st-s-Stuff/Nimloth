"""RL rollout schema and dataset-split safety tests."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from experiments.training.rl.rollout_env import (
    parse_args,
    summarize_eval_set_rollouts,
    validate_split,
    validate_trajectories,
)
from nimloth.agent import AgentTranscript, NimlothPromptTemplate
from nimloth.backbone.qwen25vl.policy import validate_agent_policy_protocol
from nimloth.environment.navigation import collector as collector_module
from nimloth.environment.navigation.collector import VAGENNavigationRolloutCollector
from nimloth.environment.navigation.vagen import (
    navigation_environment_config,
    vagen_eval_nimloth_observation_text,
)
from nimloth.rollout import RolloutTrajectory, save_trajectories
from nimloth.rollout.record_format import STEP_REWARD_PROVENANCE
from nimloth.rollout.transitions import discounted_action_value_targets


def _trajectory() -> RolloutTrajectory:
    prompt = NimlothPromptTemplate(latent_token_count=1, action_count=8)
    system_prompt = "Follow the navigation instruction."
    observation_texts = (
        "Human Instruction: Move near the couch.\n<image>",
        "Feedback: Action completed.\n<image>",
    )
    image_paths = ("before.png", "after.png")
    response = prompt.assistant_response(0, thought="Move toward the couch.")
    policy_messages = prompt.build_response_policy_prompt(
        AgentTranscript(
            system_prompt=system_prompt,
            observation_texts=observation_texts[:1],
            observation_images=image_paths[:1],
            action_indices=(),
        ),
    ).unbound_messages()
    full_transcript = AgentTranscript(
        system_prompt=system_prompt,
        observation_texts=observation_texts,
        observation_images=image_paths,
        action_indices=(0,),
        assistant_responses=(response,),
    )
    return RolloutTrajectory(
        record_id="train-1",
        image_paths=list(image_paths),
        action_indices=[0],
        action_names=["moveahead"],
        action_log_probs=[[-math.log(8.0)] * 8],
        instruction="Move near the couch.",
        reward_provenance=STEP_REWARD_PROVENANCE,
        rewards=[0.0],
        terminated=True,
        split="train",
        system_prompt=system_prompt,
        observation_texts=list(observation_texts),
        assistant_responses=[response],
        terminal_assistant_prefix=prompt.assistant_prefix(
            thought="Terminal observation."
        ),
        state_latent_hiddens=[[[0.0, 1.0]], [[1.0, 2.0]]],
        policy_credit_assignment="turn",
        policy_messages=[policy_messages],
        policy_token_ids=[[100, 102, 103]],
        policy_token_log_probs=[[-0.2, -math.log(8.0), None]],
        policy_loss_masks=[[True, True, False]],
        policy_token_roles=[["reasoning", "action", "injected"]],
        policy_action_token_ids=[[102, 202, 203, 204, 205, 206, 207, 208]],
        policy_reasoning_texts=["Move toward the couch."],
        policy_finish_reasons=["stop"],
        policy_reasoning_truncated=[False],
        prompt_template_spec=prompt.spec,
    )


def _turn_trajectory() -> RolloutTrajectory:
    return _trajectory()


def test_rl_policy_protocol_requires_positive_k_inject() -> None:
    assert validate_agent_policy_protocol(SimpleNamespace(
        nimloth_latent_token_count=1,
        nimloth_latent_query_mode="inject",
    )) == 1
    assert validate_agent_policy_protocol(SimpleNamespace(
        nimloth_latent_token_count=16,
        nimloth_latent_query_mode="inject",
    )) == 16
    with pytest.raises(ValueError, match="positive-k inject"):
        validate_agent_policy_protocol(SimpleNamespace(
            nimloth_latent_token_count=1,
            nimloth_latent_query_mode="generate",
        ))
    with pytest.raises(ValueError, match="positive-k inject"):
        validate_agent_policy_protocol(SimpleNamespace(
            nimloth_latent_token_count=0,
            nimloth_latent_query_mode="inject",
        ))


def test_training_split_requires_training_dataset() -> None:
    validate_split("base_train", "train")
    with pytest.raises(ValueError, match="refusing to label eval dataset"):
        validate_split("base", "train")
    with pytest.raises(ValueError, match="must use --split train"):
        validate_split("base_train", "eval")


def test_rollout_cli_accepts_multiple_training_datasets() -> None:
    args = parse_args(
        [
            "--model", "model",
            "--env-url", "http://env",
            "--output-dir", "output",
            "--eval-sets", "base_train", "common_sense_train",
            "--split", "train",
        ]
    )

    assert args.eval_set is None
    assert args.eval_sets == ["base_train", "common_sense_train"]
    assert args.max_pixels is None


def test_vagen_eval_navigation_profile_is_an_explicit_rollout_setting() -> None:
    args = parse_args(
        [
            "--model", "model",
            "--env-url", "http://env",
            "--output-dir", "output",
            "--eval-set", "base",
            "--split", "eval",
            "--navigation-profile", "vagen_eval",
        ]
    )

    assert args.navigation_profile == "vagen_eval"
    current = navigation_environment_config("base")["env_config"]
    historical = navigation_environment_config("base", profile="vagen_eval")[
        "env_config"
    ]
    assert current["step_length"] == 0.5
    assert current["success_threshold"] == 1.5
    assert historical["step_length"] == 0.3
    assert historical["success_threshold"] == 1.0
    assert historical["format_reward"] == 0.01
    assert historical["per_turn_format_reward"] == 0.01
    assert historical["success_reward"] == 1.0

    with pytest.raises(ValueError, match="unknown navigation profile"):
        navigation_environment_config("base", profile="approximate")


def test_vagen_eval_nimloth_observation_uses_source_prompt_wording() -> None:
    initial = vagen_eval_nimloth_observation_text(
        {
            "obs_str": (
                "[Initial Observation]:\n<image>\n"
                "Human Instruction: go to the couch\n"
                "Decide your next action.\ncurrent format text"
            )
        },
        initial=True,
    )
    later = vagen_eval_nimloth_observation_text(
        {
            "obs_str": (
                "After your answer, the extracted valid action is ['move_forward'].\n"
                "<image>\nDecide your next action."
            )
        },
        initial=False,
    )

    assert "Human Instruction: go to the couch" in initial
    assert "Decide your next action(s)." in initial
    assert "<|action_start|><|action_(idx)|><|action_end|>" in initial
    assert "current format text" not in initial
    assert later.startswith("After your action,")
    assert later.endswith("Decide your next action(s).")


def test_env_collector_enforces_training_dataset() -> None:
    VAGENNavigationRolloutCollector(
        None,
        "http://env",
        eval_sets=("base_train",),
        split="train",
    )
    with pytest.raises(ValueError, match=r"requires \*_train datasets"):
        VAGENNavigationRolloutCollector(
            None,
            "http://env",
            eval_sets=("base",),
            split="train",
        )


def test_eval_collector_can_assign_the_same_seed_range_per_dataset() -> None:
    collector = VAGENNavigationRolloutCollector(
        None,
        "http://env",
        eval_sets=("base", "common_sense"),
        split="eval",
        seed_offset=5,
        seed_per_eval_set=True,
    )

    assert [collector._next_episode_identity(index) for index in range(4)] == [
        ("rl_base_000005", "base", 5),
        ("rl_common_sense_000005", "common_sense", 5),
        ("rl_base_000006", "base", 6),
        ("rl_common_sense_000006", "common_sense", 6),
    ]


def test_eval_collector_resumes_contiguous_persisted_seed_prefix(tmp_path) -> None:
    records = [_trajectory(), _trajectory()]
    records[0].record_id = "rl_base_train_000005"
    records[1].record_id = "rl_base_train_000006"
    save_trajectories(records, tmp_path)
    collector = VAGENNavigationRolloutCollector(
        None,
        "http://env",
        eval_sets=("base_train",),
        split="train",
        seed_offset=5,
        seed_per_eval_set=True,
    )

    restored = collector._load_resume_prefix(tmp_path, num_episodes=4)

    assert [record.record_id for record in restored] == [
        "rl_base_train_000005",
        "rl_base_train_000006",
    ]
    assert collector._next_episode_identity(2) == (
        "rl_base_train_000007",
        "base_train",
        7,
    )


def test_eval_collector_rejects_noncontiguous_resume_prefix(tmp_path) -> None:
    record = _trajectory()
    record.record_id = "rl_base_train_000006"
    save_trajectories([record], tmp_path)
    collector = VAGENNavigationRolloutCollector(
        None,
        "http://env",
        eval_sets=("base_train",),
        split="train",
        seed_offset=5,
        seed_per_eval_set=True,
    )

    with pytest.raises(ValueError, match="contiguous requested seed prefix"):
        collector._load_resume_prefix(tmp_path, num_episodes=4)


def _install_collector_attempt_fakes(monkeypatch, *, failures: int):
    attempts: list[tuple[str, str, int]] = []

    class FakeRuntime:
        def __init__(self, **_kwargs) -> None:
            pass

        def terminal_state(self):
            return None

    class FakeSession:
        def __init__(self, *, episode_id, eval_set, **_kwargs) -> None:
            self.episode_id = episode_id
            self.eval_set = eval_set

    class FakeRunner:
        def __init__(self, _runtime) -> None:
            pass

        def run(self, session, *, seed, max_steps):
            del max_steps
            attempts.append((session.episode_id, session.eval_set, seed))
            if len(attempts) <= failures:
                raise RuntimeError("synthetic trajectory failure")
            observations = (
                SimpleNamespace(text="Human Instruction: move near the couch"),
                SimpleNamespace(text="Feedback: done"),
            )
            return SimpleNamespace(actions=(object(),), observations=observations)

    def fake_trajectory(_episode, *, record_id, **_kwargs):
        trajectory = _trajectory()
        trajectory.record_id = record_id
        return trajectory

    monkeypatch.setattr(collector_module, "AgentRuntime", FakeRuntime)
    monkeypatch.setattr(collector_module, "VAGENNavigationSession", FakeSession)
    monkeypatch.setattr(collector_module, "EpisodeRunner", FakeRunner)
    monkeypatch.setattr(
        collector_module,
        "trajectory_from_agent_episode",
        fake_trajectory,
    )
    return attempts


def test_env_collector_retries_same_episode_identity(tmp_path, monkeypatch) -> None:
    attempts = _install_collector_attempt_fakes(monkeypatch, failures=1)
    collector = VAGENNavigationRolloutCollector(
        object(),  # type: ignore[arg-type]
        "http://env",
        eval_sets=("base_train",),
        split="train",
        seed_offset=5,
    )
    collector._client = object()
    monkeypatch.setattr(
        collector,
        "_save_images",
        lambda *_args: ["before.png", "after.png"],
    )

    trajectories = collector.collect(
        num_episodes=1,
        max_episode_attempts=2,
        output_dir=tmp_path,
    )

    assert attempts == [
        ("rl_000005", "base_train", 5),
        ("rl_000005", "base_train", 5),
    ]
    assert [trajectory.record_id for trajectory in trajectories] == ["rl_000005"]


def test_env_collector_fails_after_bounded_same_identity_attempts(
    tmp_path,
    monkeypatch,
) -> None:
    attempts = _install_collector_attempt_fakes(monkeypatch, failures=2)
    collector = VAGENNavigationRolloutCollector(
        object(),  # type: ignore[arg-type]
        "http://env",
        eval_sets=("base_train",),
        split="train",
        seed_offset=5,
    )
    collector._client = object()

    with pytest.raises(RuntimeError, match="id=rl_000005.*seed=5.*attempts=2"):
        collector.collect(
            num_episodes=1,
            max_episode_attempts=2,
            output_dir=tmp_path,
        )

    assert attempts == [
        ("rl_000005", "base_train", 5),
        ("rl_000005", "base_train", 5),
    ]


def test_rollout_metrics_report_overall_and_each_eval_set() -> None:
    records = [_trajectory() for _ in range(4)]
    records[0].success = True
    records[2].success = True
    records[0].reward = 10.0
    records[2].reward = 10.0

    metrics = summarize_eval_set_rollouts(
        records,
        ("base", "common_sense"),
    )

    assert metrics["overall"]["success_rate"] == 0.5
    assert metrics["by_eval_set"]["base"]["success_rate"] == 1.0
    assert metrics["by_eval_set"]["common_sense"]["success_rate"] == 0.0


def test_complete_trajectory_schema_passes() -> None:
    validate_trajectories([_trajectory()])


def test_rollout_batch_rejects_missing_trajectory() -> None:
    with pytest.raises(RuntimeError, match="incomplete trajectory batch"):
        validate_trajectories([_trajectory()], expected_count=2)


def test_missing_final_observation_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.image_paths.pop()
    with pytest.raises(RuntimeError, match="images=1 but actions=1"):
        validate_trajectories([trajectory])


def test_missing_action_log_probs_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.action_log_probs.clear()
    with pytest.raises(RuntimeError, match="action_log_probs=0 but actions=1"):
        validate_trajectories([trajectory])


def test_non_normalized_action_log_probs_are_rejected() -> None:
    trajectory = _trajectory()
    trajectory.action_log_probs[0] = [-2.0] * 8
    with pytest.raises(RuntimeError, match="invalid action probabilities"):
        validate_trajectories([trajectory])


def test_missing_policy_prompt_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.policy_messages.clear()
    with pytest.raises(RuntimeError, match="policy_messages=0 but actions=1"):
        validate_trajectories([trajectory])


def test_missing_prompt_template_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.prompt_template_spec = None
    with pytest.raises(RuntimeError, match="no prompt_template_spec"):
        validate_trajectories([trajectory])


def test_policy_prompt_must_match_structured_transcript() -> None:
    trajectory = _trajectory()
    trajectory.policy_messages[0][-1]["content"] = "different prompt"
    with pytest.raises(RuntimeError, match="does not match the shared Agent template"):
        validate_trajectories([trajectory])


def test_turn_credit_roundtrip_separates_behavior_and_state_prompts() -> None:
    trajectory = _turn_trajectory()

    validate_trajectories([trajectory])
    restored = RolloutTrajectory.from_record(trajectory.to_record())

    assert restored.build_policy_prompt(0).messages[-1]["content"] == "<think>"
    state_prefix = restored.build_state_prompt(0).messages[-1]["content"]
    assert state_prefix.endswith("<|latent_state|><|action_start|>")
    assert "Move toward the couch." in state_prefix
    assert "<|action_(0)|>" not in state_prefix
    terminal_prompt = restored.build_state_prompt(1)
    assert terminal_prompt.messages[-1]["content"] == (
        "<think>Terminal observation.</think><|latent_state|><|action_start|>"
    )
    assert any(
        "Move toward the couch." in str(message["content"])
        for message in terminal_prompt.messages
    )
    assert restored.policy_token_trace(0) == trajectory.policy_token_trace(0)
    assert restored.state_latent_hiddens == trajectory.state_latent_hiddens


def test_state_latent_hidden_must_cover_every_observation() -> None:
    trajectory = _trajectory()
    trajectory.state_latent_hiddens.pop()

    with pytest.raises(RuntimeError, match="state_latent_hiddens=1 but states=2"):
        validate_trajectories([trajectory])


def test_state_latent_hidden_must_match_prompt_latent_count() -> None:
    trajectory = _trajectory()
    trajectory.state_latent_hiddens[0].append([2.0, 3.0])

    with pytest.raises(RuntimeError, match="has 2 latent rows, expected 1"):
        validate_trajectories([trajectory])


def test_reference_log_probs_roundtrip_only_on_selected_reasoning() -> None:
    trajectory = _turn_trajectory()
    trajectory.policy_reference_token_log_probs = [[-0.7, None, None]]

    validate_trajectories([trajectory])
    restored = RolloutTrajectory.from_record(trajectory.to_record())

    trace = restored.policy_token_trace(0)
    assert trace is not None
    assert trace.selected_reference_log_probs == (-0.7,)


def test_turn_trace_action_token_must_match_action_index() -> None:
    trajectory = _turn_trajectory()
    trajectory.policy_token_ids[0][1] = 202

    with pytest.raises(RuntimeError, match="does not match action_index"):
        validate_trajectories([trajectory])


def test_turn_trace_action_log_prob_must_match_behavior_distribution() -> None:
    trajectory = _turn_trajectory()
    trajectory.policy_token_log_probs[0][1] = -0.3

    with pytest.raises(RuntimeError, match="does not match action_log_probs"):
        validate_trajectories([trajectory])


def test_turn_response_must_match_reasoning_and_action_trace() -> None:
    trajectory = _turn_trajectory()
    trajectory.assistant_responses[0] = trajectory.assistant_responses[0].replace(
        "action_(0)",
        "action_(1)",
    )

    with pytest.raises(RuntimeError, match="assistant response does not match"):
        validate_trajectories([trajectory])


def test_reasoning_truncation_metadata_must_be_consistent() -> None:
    trajectory = _turn_trajectory()
    trajectory.policy_reasoning_truncated[0] = True

    with pytest.raises(RuntimeError, match="truncation must match finish_reason"):
        validate_trajectories([trajectory])


def test_step_rewards_and_episode_status_roundtrip() -> None:
    trajectory = _trajectory()
    trajectory.reward = 1.5
    trajectory.rewards = [1.5]
    trajectory.terminated = True

    validate_trajectories([trajectory])
    restored = RolloutTrajectory.from_record(trajectory.to_record())

    assert restored.rewards == [1.5]
    assert restored.terminated is True
    assert restored.truncated is False


def test_token_credit_requires_step_reward_provenance() -> None:
    trajectory = _trajectory()
    trajectory.policy_credit_assignment = "token"
    trajectory.reward_provenance = "trajectory_terminal_reward"
    trajectory.rewards = []
    trajectory.terminated = False
    with pytest.raises(RuntimeError, match="token credit requires step rewards"):
        validate_trajectories([trajectory])

    trajectory.reward_provenance = "step_rewards"
    trajectory.rewards = [0.0]
    trajectory.truncated = True
    trajectory.terminated = False
    validate_trajectories([trajectory])


def test_step_reward_returns_preserve_intermediate_rewards() -> None:
    record = {
        "action_indices": [0, 1, 2],
        "reward": 2.5,
        "reward_provenance": STEP_REWARD_PROVENANCE,
        "rewards": [1.0, -0.5, 2.0],
        "terminated": True,
        "truncated": False,
    }

    assert discounted_action_value_targets(record, gamma=0.5) == [1.25, 0.5, 2.0]


def test_truncated_return_requires_explicit_bootstrap() -> None:
    record = {
        "action_indices": [0, 1],
        "reward": 0.0,
        "reward_provenance": STEP_REWARD_PROVENANCE,
        "rewards": [-0.1, 0.1],
        "terminated": False,
        "truncated": True,
    }

    with pytest.raises(ValueError, match="explicit value bootstrap"):
        discounted_action_value_targets(record, gamma=0.9)
    assert discounted_action_value_targets(
        record,
        gamma=0.9,
        truncated_bootstrap=2.0,
    ) == pytest.approx([1.61, 1.9])
