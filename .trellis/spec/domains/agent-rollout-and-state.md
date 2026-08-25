# Agent, Rollout, and State

## Agent/runtime boundary

`nimloth.agent` owns `Agent(nn.Module)`, prompt templates, behavior/planning policy, transcript, and episode runtime. Environment supplies observation/action/reward/success semantics. Rollout owns persisted trajectory schema and provenance; training consumes that contract rather than rebuilding environment behavior.

## Real-state path

For each real environment step, preserve:

```text
observation -> actual assistant response/CoT -> prompt/token trace
            -> same-forward latent hidden -> projected state
            -> selected/executed action -> reward/termination provenance
```

Terminal observation receives a separately generated and persisted real CoT/state and executes no draft action. Follow [`../governance/cot-and-state.md`](../governance/cot-and-state.md).

## Planning

Planning starts from the real Qwen state, simulates candidate actions in WM latent space, selects by the configured search contract, and executes only the first selected action. The next environment observation triggers a fresh Qwen state and replanning. Candidate tail states do not receive invented Qwen CoT/hidden. Keep planner behavior/search trace distinct from direct-Qwen PPO policy replay.

## Rollout persistence

Use the current versioned trajectory schema and validation. Preserve prompt template, action-space mapping, actual assistant response, behavior sampling/token log-probs/masks, per-step reward, `terminated` versus `truncated`, success provenance, and required planner state/search trace. Migration declares unavailable source semantics and fails closed; it does not invent CoT, hidden state, reward, or transitions.

Windows remain inside one trajectory and preserve original order/provenance. Ordinary trajectories re-encode current prompts during training rather than storing arbitrary stale hidden/KV state. Any exception is an explicit audited schema contract documented by the rollout owner.

Sources: [`agent/README.md`](../../../src/nimloth/agent/README.md), [`rollout/README.md`](../../../src/nimloth/rollout/README.md), [`training/rl/README.md`](../../../src/nimloth/training/rl/README.md).
