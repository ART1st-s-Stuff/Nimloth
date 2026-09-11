"""公共 Agent transcript、模板、policy、runtime 与 episode runner。"""

from nimloth.agent.policy import (
    ActionLogProbReplay,
    AgentPolicy,
    PlannerPolicyTrace,
    PolicyDecision,
    PolicyReplayInput,
    PolicyReplayOutput,
    PolicyState,
    PolicyStateTokenBudgetExceeded,
    PolicyTokenTrace,
    behavior_log_probs,
    categorical_entropy_from_log_probs,
    sample_policy_decision,
    validate_action_log_probs,
)
from nimloth.agent.registry import create_prompt_template
from nimloth.agent.runner import AgentEpisode, EpisodeRunner
from nimloth.agent.runtime import AgentAction, AgentRuntime
from nimloth.agent.serialization import prompt_template_spec_from_record
from nimloth.agent.template import (
    AgentPrompt,
    AgentPromptTemplate,
    PromptTemplateSpec,
    bind_image_placeholders,
)
from nimloth.agent.templates import (
    NIMLOTH_PROMPT_TEMPLATE_ID,
    PROMPT_VERSION,
    NimlothPromptTemplate,
)
from nimloth.agent.transcript import AgentTranscript

__all__ = [
    "PROMPT_VERSION",
    "Agent",
    "AgentAction",
    "ActionLogProbReplay",
    "AgentEpisode",
    "AgentPolicy",
    "AgentPrompt",
    "AgentPromptTemplate",
    "AgentOutput",
    "AgentRolloutOutput",
    "AgentRuntime",
    "AgentStateOutput",
    "AgentTranscript",
    "EpisodeRunner",
    "NIMLOTH_PROMPT_TEMPLATE_ID",
    "NimlothPromptTemplate",
    "PolicyDecision",
    "PolicyReplayInput",
    "PolicyReplayOutput",
    "PolicyState",
    "PolicyStateTokenBudgetExceeded",
    "PlannerPolicyTrace",
    "PolicyTokenTrace",
    "PlanningPolicy",
    "PromptTemplateSpec",
    "WorldModelPlan",
    "WorldModelPlanner",
    "behavior_log_probs",
    "bind_image_placeholders",
    "categorical_entropy_from_log_probs",
    "create_prompt_template",
    "prompt_template_spec_from_record",
    "sample_policy_decision",
    "validate_action_log_probs",
]


# Prompt/direct-policy users do not need the optional world-model implementation.
# Preserve public model/planner imports, loading them only when requested.
def __getattr__(name: str):
    from importlib import import_module

    modules = {
        "Agent": "model", "AgentOutput": "model",
        "AgentRolloutOutput": "model", "AgentStateOutput": "model",
        "PlanningPolicy": "planning", "WorldModelPlan": "planning",
        "WorldModelPlanner": "planning",
    }
    module = modules.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
