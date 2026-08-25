# World Model and Training

## Model ownership

`nimloth.wm` owns StateProjector, state predictor, ValueHead, grid variants, sequence SIGReg, and latent rollout primitives. Agent owns search. SFT2/RL own stage-specific data assembly, objectives, stop-gradient/train/freeze policy, optimizer, validation, and checkpoint behavior.

ValueHead is action value `Q(s,a)` for outgoing actions from the input decision state. Do not pair an executed incoming action with its successor as though it were the successor's outgoing action. SFT2 prediction horizon and RL planning horizon are separate contracts.

## Training contract

For each change or experiment, explicitly trace:

- source/target states and time/batch/sequence axes;
- recorded action alignment and no cross-trajectory windows;
- each objective and its exact module/parameter recipients;
- gradient path into or around Backbone, StateProjector, predictor, ValueHead, actor/token critic, and any teacher/EMA;
- trainable/frozen parameter sets and optimizer groups;
- checkpoint component ownership, structure metadata, optimizer/scheduler/RNG/data position, and resume boundary;
- distributed rank/model-parallel topology and collective order.

Do not flatten a multistep WM objective into independent prompts, pre-encode states when gradients must reach Backbone, duplicate an existing algorithm under an “optional loss,” or replace framework synchronization with an approximate manual mechanism.

## Stage sources

- [`wm/README.md`](../../../src/nimloth/wm/README.md)
- [`training/sft2/README.md`](../../../src/nimloth/training/sft2/README.md)
- [`training/rl/README.md`](../../../src/nimloth/training/rl/README.md)
- [`training/common/README.md`](../../../src/nimloth/training/common/README.md)
- [`config/sft2/README.md`](../../../src/nimloth/config/sft2/README.md)
- [`config/rl/README.md`](../../../src/nimloth/config/rl/README.md)

A configuration printout, import test, or fake-only unit test does not prove the real model/runtime composition. Verify production constructors and at least the appropriate focused integration path.
