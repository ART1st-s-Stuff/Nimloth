"""把 Agent episode 转换为可持久化 rollout trajectory。"""

from __future__ import annotations

from nimloth.agent import AgentEpisode, PolicyState, create_prompt_template
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
    terminal_state: PolicyState,
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
    trace_rows = [
        (step, action.token_trace)
        for step, action in enumerate(episode.actions)
        if action.token_trace is not None
    ]
    planner_rows = [
        (step, action.planner_trace)
        for step, action in enumerate(episode.actions)
        if action.planner_trace is not None
    ]
    trace_steps = [step for step, _trace in trace_rows]
    planner_steps = [step for step, _trace in planner_rows]
    if planner_rows and trace_steps != planner_steps:
        raise ValueError("planner anchor must carry its Qwen token provenance")
    if trace_rows and len(trace_rows) != len(episode.actions) and not planner_rows:
        raise ValueError("direct policy episode mixes traced and untraced actions")
    credit_assignments = {action.credit_assignment for action in episode.actions}
    if len(credit_assignments) != 1:
        raise ValueError("Agent episode mixes PPO credit assignment modes")
    credit_assignment = credit_assignments.pop()
    if credit_assignment in {"turn", "token"} and len(trace_rows) != len(
        episode.actions
    ):
        raise ValueError(f"{credit_assignment} credit requires token traces")
    cached_state_rows = [
        (step, action.state_latent_hidden)
        for step, action in enumerate(episode.actions)
        if action.state_latent_hidden is not None
    ]
    if cached_state_rows and [step for step, _state in cached_state_rows] != trace_steps:
        raise ValueError("Qwen state cache must align with policy anchor steps")
    if bool(cached_state_rows) != (terminal_state.latent_hidden is not None):
        raise ValueError("terminal Qwen state cache does not match action anchors")
    state_anchor_steps = [step for step, _state in cached_state_rows]
    cached_states = [state for _step, state in cached_state_rows]
    if terminal_state.latent_hidden is not None:
        state_anchor_steps.append(len(episode.actions))
        cached_states.append(terminal_state.latent_hidden)

    world_model_states = [
        action.world_model_state for action in episode.actions
    ]
    has_world_model_states = [state is not None for state in world_model_states]
    if any(has_world_model_states) and not all(has_world_model_states):
        raise ValueError("planner episode requires a state for every executed action")
    if any(has_world_model_states) != (terminal_state.world_model_state is not None):
        raise ValueError("terminal world-model state does not match planner trajectory")
    if any(has_world_model_states):
        world_model_states.append(terminal_state.world_model_state)

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
        rewards=list(episode.rewards),
        terminated=episode.done,
        truncated=not episode.done,
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
        terminal_assistant_prefix=terminal_state.assistant_prefix,
        state_anchor_steps=state_anchor_steps,
        state_latent_hiddens=[
            state.detach().cpu().float().tolist()
            for state in cached_states
            if state is not None
        ],
        world_model_states=[
            state.detach().cpu().float().tolist()
            for state in world_model_states
            if state is not None
        ],
        policy_credit_assignment=credit_assignment,
        policy_step_indices=trace_steps,
        policy_token_ids=[
            list(trace.token_ids) for _step, trace in trace_rows if trace is not None
        ],
        policy_token_log_probs=[
            list(trace.old_log_probs)
            for _step, trace in trace_rows
            if trace is not None
        ],
        policy_loss_masks=[
            list(trace.loss_mask) for _step, trace in trace_rows if trace is not None
        ],
        policy_token_roles=[
            list(trace.token_roles) for _step, trace in trace_rows if trace is not None
        ],
        policy_action_token_ids=[
            list(trace.action_token_ids)
            for _step, trace in trace_rows
            if trace is not None
        ],
        policy_reasoning_texts=[
            trace.reasoning_text for _step, trace in trace_rows if trace is not None
        ],
        policy_finish_reasons=[
            trace.finish_reason for _step, trace in trace_rows if trace is not None
        ],
        policy_reasoning_truncated=[
            trace.reasoning_truncated
            for _step, trace in trace_rows
            if trace is not None
        ],
        planner_policy_traces=[
            trace for _step, trace in planner_rows if trace is not None
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
