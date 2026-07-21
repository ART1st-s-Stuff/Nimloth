"""Rollout trajectory 的跨字段完整性校验。"""

from __future__ import annotations

from nimloth.agent import create_prompt_template, validate_action_log_probs
from nimloth.environment import get_action_space
from nimloth.rollout.schema import RolloutTrajectory


def validate_rollout_trajectory(trajectory: RolloutTrajectory) -> None:
    """在写盘或训练前校验一条结构化 Agent trajectory。"""

    prefix = f"trajectory {trajectory.record_id}"
    if len(trajectory.image_paths) != trajectory.num_steps + 1:
        raise ValueError(
            f"{prefix}: images={len(trajectory.image_paths)} "
            f"but actions={trajectory.num_steps}"
        )
    if len(trajectory.observation_texts) != trajectory.num_steps + 1:
        raise ValueError(
            f"{prefix}: observations={len(trajectory.observation_texts)} "
            f"but actions={trajectory.num_steps}"
        )
    if len(trajectory.action_names) != trajectory.num_steps:
        raise ValueError(
            f"{prefix}: action_names={len(trajectory.action_names)} "
            f"but actions={trajectory.num_steps}"
        )

    action_space = get_action_space(
        trajectory.action_space_id,
        trajectory.action_space_version,
    )
    expected_names = [
        action_space.key_for(index) for index in trajectory.action_indices
    ]
    if trajectory.action_names != expected_names:
        raise ValueError(f"{prefix}: action names do not match action indices")
    _validate_behavior_probabilities(trajectory, action_count=len(action_space))

    if len(trajectory.policy_messages) != trajectory.num_steps:
        raise ValueError(
            f"{prefix}: policy_messages={len(trajectory.policy_messages)} "
            f"but actions={trajectory.num_steps}"
        )
    if not trajectory.system_prompt:
        raise ValueError(f"{prefix} has no system prompt")
    _validate_prompt_contract(trajectory, action_count=len(action_space))

    if not trajectory.instruction:
        raise ValueError(f"{prefix} has no task instruction")
    if trajectory.sampling_temperature < 0.0:
        raise ValueError(f"{prefix} has a negative sampling temperature")
    if not 0.0 < trajectory.sampling_top_p <= 1.0:
        raise ValueError(f"{prefix} has sampling_top_p outside (0, 1]")


def _validate_behavior_probabilities(
    trajectory: RolloutTrajectory,
    *,
    action_count: int,
) -> None:
    prefix = f"trajectory {trajectory.record_id}"
    if len(trajectory.action_log_probs) != trajectory.num_steps:
        raise ValueError(
            f"{prefix}: action_log_probs={len(trajectory.action_log_probs)} "
            f"but actions={trajectory.num_steps}"
        )
    for step, (action_index, log_probs) in enumerate(
        zip(trajectory.action_indices, trajectory.action_log_probs, strict=True)
    ):
        try:
            validate_action_log_probs(
                action_index,
                log_probs,
                action_count=action_count,
            )
        except ValueError as error:
            raise ValueError(
                f"{prefix} step {step} has invalid action probabilities: {error}"
            ) from error


def _validate_prompt_contract(
    trajectory: RolloutTrajectory,
    *,
    action_count: int,
) -> None:
    prefix = f"trajectory {trajectory.record_id}"
    prompt_spec = trajectory.resolved_prompt_template_spec()
    if trajectory.prompt_version != prompt_spec.version:
        raise ValueError(
            f"{prefix} prompt_version {trajectory.prompt_version!r} does not "
            f"match template version {prompt_spec.version!r}"
        )
    # 创建模板本身会验证 identifier、version 与 config。
    create_prompt_template(prompt_spec, action_count=action_count)
    if trajectory.latent_token_count != trajectory.resolved_latent_token_count():
        raise ValueError(
            f"{prefix} latent_token_count {trajectory.latent_token_count} does "
            "not match the prompt template"
        )
    for step, policy_messages in enumerate(trajectory.policy_messages):
        expected_messages = trajectory.build_policy_messages(step, bind_images=False)
        if policy_messages != expected_messages:
            raise ValueError(
                f"{prefix} step {step} policy prompt does not match the "
                "shared Agent template"
            )
    expected_completed = trajectory.build_completed_messages(bind_images=False)
    if trajectory.messages != expected_completed:
        raise ValueError(
            f"{prefix} completed messages do not match the shared Agent template"
        )
