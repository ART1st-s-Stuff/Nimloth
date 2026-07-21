# Agent

`nimloth.agent` owns the model-independent navigation Agent contract. SFT2,
RL rollout, PPO replay, and future evaluation code must share this contract
instead of constructing their own prompts.

| Module | Responsibility |
|--------|----------------|
| `prompt.py` | Structured transcript, action vocabulary, prompt version, assistant action format, and ordered image binding |
| `runtime.py` | Stateful `NavigationAgent`, policy protocol, action result, and behavior-probability validation |

The Qwen implementation of the policy protocol lives in
`nimloth.backbone.qwen25vl.policy`; environment orchestration stays in the
training or evaluation caller.

## Episode contract

The environment is authoritative for text:

- `system_prompt` is the prompt returned by the environment.
- Each observation keeps its original `obs_str`, including the `<image>` slot.
- Images are stored separately, in observation order.
- `action_indices[t]` is the action after observation `t`.
- A completed trajectory has one more observation/image than actions.

`NimlothAgentPrompt` turns that structured data into either:

- a policy query ending at `<|action_start|>`, used by online rollout, PPO
  replay, and WM state encoding; or
- completed assistant action turns, used by SFT2 and trajectory serialization.

Both forms share the same assistant prefix and are tagged with
`PROMPT_VERSION`. Image objects or paths are bound only immediately before the
Qwen processor is called.

## Runtime

```python
from nimloth.agent import NavigationAgent, NimlothAgentPrompt
from nimloth.backbone.qwen25vl.policy import QwenNavigationPolicy

policy = QwenNavigationPolicy(
    model=model,
    processor=processor,
    device=device,
    temperature=0.7,
    top_p=0.95,
)
agent = NavigationAgent(policy=policy, prompt=NimlothAgentPrompt())
agent.reset(system_prompt=environment_system_prompt)

agent.observe(text=observation["obs_str"], image=observation_image)
action = agent.act()
next_observation = environment.step(action.response)
```

`NavigationAgent` owns episode history and performs the actual policy call.
Qwen-specific tokenization and logits remain behind `QwenNavigationPolicy`, so
the Agent runtime can be tested with a small fake policy.

The former `WMAgent` slow/fast-path prototype was removed because no in-repo
training, evaluation, rollout, CLI, or experiment path called it. A future
planner-backed Agent should be introduced only together with a real caller and
an explicit state/prompt contract.
