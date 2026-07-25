"""Rollout trajectory 的跨字段完整性校验。"""

from __future__ import annotations

import math

from nimloth.agent import create_prompt_template, validate_action_log_probs
from nimloth.environment import get_action_space
from nimloth.latent import LatentActionTokens, latent_state_tokens
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
    _validate_reward_provenance(trajectory)

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
    _validate_token_provenance(trajectory, action_count=len(action_space))

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


def _validate_reward_provenance(trajectory: RolloutTrajectory) -> None:
    """校验 fresh rollout 的逐步 reward 与 episode 结束语义。"""

    prefix = f"trajectory {trajectory.record_id}"
    if trajectory.rewards:
        if len(trajectory.rewards) != trajectory.num_steps:
            raise ValueError(
                f"{prefix}: rewards={len(trajectory.rewards)} "
                f"but actions={trajectory.num_steps}"
            )
        if trajectory.terminated == trajectory.truncated:
            raise ValueError(
                f"{prefix} with step rewards must be exactly one of "
                "terminated or truncated"
            )
        if not math.isclose(
            trajectory.reward,
            sum(trajectory.rewards),
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"{prefix} aggregate reward does not equal the step rewards"
            )
    elif trajectory.terminated or trajectory.truncated:
        raise ValueError(f"{prefix} has episode status without step rewards")
    elif trajectory.policy_credit_assignment == "token":
        raise ValueError(f"{prefix} token credit requires step rewards and status")


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


def _validate_token_provenance(
    trajectory: RolloutTrajectory,
    *,
    action_count: int,
) -> None:
    prefix = f"trajectory {trajectory.record_id}"
    if trajectory.policy_credit_assignment not in {"action", "turn", "token"}:
        raise ValueError(
            f"{prefix} has unsupported policy_credit_assignment "
            f"{trajectory.policy_credit_assignment!r}"
        )
    if trajectory.assistant_responses and (
        len(trajectory.assistant_responses) != trajectory.num_steps
    ):
        raise ValueError(
            f"{prefix}: assistant_responses={len(trajectory.assistant_responses)} "
            f"but actions={trajectory.num_steps}"
        )
    if len(trajectory.assistant_responses) != trajectory.num_steps:
        raise ValueError(
            f"{prefix} requires a real assistant response for every action"
        )
    try:
        for step in range(trajectory.num_steps + 1):
            trajectory._state_assistant_prefix(step)
    except ValueError as error:
        raise ValueError(f"{prefix} has invalid CoT state data: {error}") from error
    trace_fields = (
        trajectory.policy_token_ids,
        trajectory.policy_token_log_probs,
        trajectory.policy_loss_masks,
        trajectory.policy_token_roles,
        trajectory.policy_action_token_ids,
        trajectory.policy_reasoning_texts,
        trajectory.policy_finish_reasons,
        trajectory.policy_reasoning_truncated,
    )
    populated = [bool(field) for field in trace_fields]
    if any(populated) and not all(populated):
        raise ValueError(f"{prefix} has incomplete policy token trace fields")
    if trajectory.policy_credit_assignment in {"turn", "token"} and not all(populated):
        raise ValueError(
            f"{prefix} {trajectory.policy_credit_assignment} credit requires "
            "policy token traces"
        )
    if not any(populated):
        return
    if not all(len(field) == trajectory.num_steps for field in trace_fields):
        raise ValueError(f"{prefix} policy token trace count does not match actions")
    for step in range(trajectory.num_steps):
        try:
            trace = trajectory.policy_token_trace(step)
        except ValueError as error:
            raise ValueError(f"{prefix} step {step} has invalid token trace: {error}") from error
        assert trace is not None
        if len(trace.action_token_ids) != action_count:
            raise ValueError(
                f"{prefix} step {step} action token mapping has "
                f"{len(trace.action_token_ids)} entries, expected {action_count}"
            )
        action_index = trajectory.action_indices[step]
        action_position = trace.token_roles.index("action")
        expected_action_token_id = trace.action_token_ids[action_index]
        if trace.token_ids[action_position] != expected_action_token_id:
            raise ValueError(
                f"{prefix} step {step} token trace action does not match action_index"
            )
        old_action_log_prob = trace.old_log_probs[action_position]
        expected_old_log_prob = trajectory.action_log_probs[step][action_index]
        if old_action_log_prob is None or not math.isclose(
            old_action_log_prob,
            expected_old_log_prob,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"{prefix} step {step} token trace action log-prob does not match "
                "action_log_probs"
            )
        selected_roles = [
            role
            for role, selected in zip(
                trace.token_roles,
                trace.loss_mask,
                strict=True,
            )
            if selected
        ]
        if trajectory.policy_credit_assignment == "action":
            if selected_roles != ["action"]:
                raise ValueError(
                    f"{prefix} step {step} action credit must select only action token"
                )
        elif "reasoning" not in selected_roles:
            raise ValueError(
                f"{prefix} step {step} {trajectory.policy_credit_assignment} "
                "credit has no reasoning token"
            )
        if trajectory.policy_credit_assignment in {"turn", "token"}:
            if not trajectory.assistant_responses:
                raise ValueError(f"{prefix} turn credit requires assistant responses")
            tokens = LatentActionTokens()
            expected_response = (
                f"<think>{trace.reasoning_text}</think>"
                f"{''.join(latent_state_tokens(trajectory.resolved_latent_token_count(), tokens))}"
                f"{tokens.action_start}{tokens.action_tokens[action_index]}"
                f"{tokens.action_end}"
            )
            if trajectory.assistant_responses[step] != expected_response:
                raise ValueError(
                    f"{prefix} step {step} assistant response does not match "
                    "token trace reasoning/action"
                )


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
