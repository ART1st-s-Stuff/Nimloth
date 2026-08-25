# CoT and State Semantics

These are hard project semantics. They apply to training, evaluation, planning, rollout, persistence, and deployment.

## No invented fixed CoT

- Never invent or fill a fixed CoT. CoT is model-generated or dataset-recorded content, not a prompt-template constant.
- Unless the human explicitly requests fixed text, default thoughts, stand-in thoughts, “canonical thoughts,” and synthetic filler must not enter state.
- Do not silently repair missing CoT with a plausible sentence.

## Observation-aligned real CoT

- A CoT-conditioned state uses the real CoT corresponding to that same observation.
- For an ordinary observation, state reads the actual assistant response produced/recorded for that turn.
- For a terminal observation, generate and persist the actual terminal CoT, but do not execute the draft action associated with that terminal response.
- Intermediate planner states that did not run Qwen do not receive invented CoT.

## Terminal generation gate

Before generating terminal CoT, the checkpoint, sampling parameters, and generation boundary must be explicit. If any is unspecified, stop and ask the human. Do not choose defaults merely to continue an experiment.

## Cross-module evidence

- [`src/nimloth/agent/README.md`](../../../src/nimloth/agent/README.md) owns Agent prompt/runtime behavior.
- [`src/nimloth/rollout/README.md`](../../../src/nimloth/rollout/README.md) owns persisted trajectory and terminal-state requirements.
- [`src/nimloth/training/rl/README.md`](../../../src/nimloth/training/rl/README.md) owns current RL replay/training semantics.
- Known error [`E0045_do_not_invent_fixed_cot.md`](../../../ai_rules/known_errors/E0045_do_not_invent_fixed_cot.md) records the confirmed failure pattern.

## Verification

For every touched state path, trace observation → actual assistant response/terminal generation → persisted record → state encoder. Tests that merely assert a constant thought string are not evidence of compliance.
