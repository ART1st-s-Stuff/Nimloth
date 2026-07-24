"""把 Agent episode 转换为可持久化 rollout trajectory。"""

from __future__ import annotations

from nimloth.agent import AgentEpisode, create_prompt_template
from nimloth.environment import get_action_space
from nimloth.rollout.schema import RolloutTrajectory


def trajectory_from_agent_episode(
    episode: AgentEpisode,
    *,
    record_id: str,
    image_paths: list[str],
    instruction: str,
    split: str,
    sampling_temperature: float,
    sampling_top_p: float,
    terminal_assistant_prefix: str | None = None,
) -> RolloutTrajectory:
    """只消费 AgentEpisode，不再从 collector 拼装 prompt 细节。"""

    if len(image_paths) != len(episode.observations):
        raise ValueError(
            "saved image count must match Agent episode observations: "
            f"{len(image_paths)} != {len(episode.observations)}"
        )
    action_space = get_action_space(
        episode.action_space_id,
        episode.action_space_version,
    )
    template = create_prompt_template(
        episode.prompt_template,
        action_count=len(action_space),
    )
    completed_prompt = template.build_supervised_prompt(episode.transcript)
    for action in episode.actions:
        if action.policy_prompt.template != episode.prompt_template:
            raise ValueError("Agent episode mixes prompt template specifications")
    traces = [action.token_trace for action in episode.actions]
    has_traces = [trace is not None for trace in traces]
    if any(has_traces) and not all(has_traces):
        raise ValueError("Agent episode mixes traced and untraced policy actions")
    credit_assignments = {
        "turn"
        if trace is not None
        and any(
            role == "reasoning" and selected
            for role, selected in zip(
                trace.token_roles,
                trace.loss_mask,
                strict=True,
            )
        )
        else "action"
        for trace in traces
    }
    if len(credit_assignments) != 1:
        raise ValueError("Agent episode mixes PPO credit assignment modes")
    credit_assignment = credit_assignments.pop()
    if not terminal_assistant_prefix:
        raise ValueError(
            "trajectory conversion requires a separately generated terminal CoT prefix"
        )

    return RolloutTrajectory(
        record_id=record_id,
        image_paths=image_paths,
        action_indices=[action.action_index for action in episode.actions],
        action_names=[action.action_key for action in episode.actions],
        action_log_probs=[
            list(action.action_log_probs) for action in episode.actions
        ],
        instruction=instruction,
        # 成功语义由具体 environment session 判定，公共 rollout 不猜 reward 阈值。
        success=episode.success,
        reward=episode.reward,
        split=split,
        messages=completed_prompt.unbound_messages(),
        system_prompt=episode.system_prompt,
        observation_texts=[
            observation.text for observation in episode.observations
        ],
        policy_messages=[
            action.policy_prompt.unbound_messages()
            for action in episode.actions
        ],
        assistant_responses=[action.response for action in episode.actions],
        terminal_assistant_prefix=terminal_assistant_prefix,
        policy_credit_assignment=credit_assignment,
        policy_token_ids=[
            list(trace.token_ids) for trace in traces if trace is not None
        ],
        policy_token_log_probs=[
            list(trace.old_log_probs) for trace in traces if trace is not None
        ],
        policy_loss_masks=[
            list(trace.loss_mask) for trace in traces if trace is not None
        ],
        policy_token_roles=[
            list(trace.token_roles) for trace in traces if trace is not None
        ],
        policy_action_token_ids=[
            list(trace.action_token_ids) for trace in traces if trace is not None
        ],
        policy_reasoning_texts=[
            trace.reasoning_text for trace in traces if trace is not None
        ],
        policy_finish_reasons=[
            trace.finish_reason for trace in traces if trace is not None
        ],
        policy_reasoning_truncated=[
            trace.reasoning_truncated for trace in traces if trace is not None
        ],
        prompt_template_spec=episode.prompt_template,
        prompt_version=episode.prompt_template.version,
        latent_token_count=int(
            episode.prompt_template.config.get("latent_token_count", 1)
        ),
        sampling_temperature=sampling_temperature,
        sampling_top_p=sampling_top_p,
        action_space_id=episode.action_space_id,
        action_space_version=episode.action_space_version,
    )
