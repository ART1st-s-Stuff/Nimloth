# Terminology and Ownership

Use the exact project terms defined in the root [`README.md`](../../../README.md). When discussing a parameter, include its full configuration field. In particular, keep these distinct:

- environment step, model turn, WM prediction horizon, planner horizon, PPO iteration, and episode length;
- action key, stable action index, Nimloth action token, and tokenizer token ID;
- prompt history, Backbone hidden, projected WM state, online/target state, and predicted state;
- `train.history_size`, `train.prediction_horizon`, `predictor.history_size`, `agent.planning.horizon`, and `rl.max_steps_per_episode`;
- training-rollout success, held-out success, average reward, single-step reward, and static dataset statistics;
- initialization checkpoint, component checkpoint, merged policy artifact, and full resume checkpoint.

## Ownership map

- Agent prompt/model/planning/runtime: [`src/nimloth/agent/README.md`](../../../src/nimloth/agent/README.md)
- Backbone/Qwen integration: [`src/nimloth/backbone/README.md`](../../../src/nimloth/backbone/README.md)
- Configuration schemas: [`src/nimloth/config/README.md`](../../../src/nimloth/config/README.md)
- Environment semantics: [`src/nimloth/environment/navigation/README.md`](../../../src/nimloth/environment/navigation/README.md)
- Trajectory schema/storage/windows: [`src/nimloth/rollout/README.md`](../../../src/nimloth/rollout/README.md)
- State transition/value modules: [`src/nimloth/wm/README.md`](../../../src/nimloth/wm/README.md)
- Stage objectives/runtime/checkpoint: [`src/nimloth/training/README.md`](../../../src/nimloth/training/README.md)
- Reconstruction/evaluation: [`src/nimloth/recon/README.md`](../../../src/nimloth/recon/README.md), [`src/nimloth/eval/README.md`](../../../src/nimloth/eval/README.md)

If ownership appears to require a new cross-module mechanism, stop and update the reviewed design rather than inserting it into a convenient module.
