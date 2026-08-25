# Structure and Module Indexes

Reusable Python belongs under `src/nimloth/`; reusable YAML/JSON under `configs/`; thin reusable operational entry points under `experiments/`; tests under the matching `tests/` ownership boundary. Do not solve module ownership by splitting files mechanically or by moving reusable logic into one-off launchers. Backward compatibility with old data is not mandatory unless the reviewed task requires it; do not contort a clear design merely to preserve an unapproved legacy format.

Each source module must have a README index. Read and update the owner when its public boundary changes:

| Area | Module index |
|---|---|
| package overview | [`src/nimloth/README.md`](../../../src/nimloth/README.md) |
| Agent, prompt, planning, runtime | [`agent/README.md`](../../../src/nimloth/agent/README.md) |
| Backbone | [`backbone/README.md`](../../../src/nimloth/backbone/README.md), [`qwen25vl/README.md`](../../../src/nimloth/backbone/qwen25vl/README.md) |
| Typed configuration | [`config/README.md`](../../../src/nimloth/config/README.md), [`agent`](../../../src/nimloth/config/agent/README.md), [`rollout`](../../../src/nimloth/config/rollout/README.md), [`sft2`](../../../src/nimloth/config/sft2/README.md), [`rl`](../../../src/nimloth/config/rl/README.md) |
| Environment | [`environment/navigation/README.md`](../../../src/nimloth/environment/navigation/README.md) |
| Latent extraction | [`latent/README.md`](../../../src/nimloth/latent/README.md) |
| Rollout schema/storage/windows | [`rollout/README.md`](../../../src/nimloth/rollout/README.md) |
| World model and heads | [`wm/README.md`](../../../src/nimloth/wm/README.md) |
| Training | [`training/README.md`](../../../src/nimloth/training/README.md), [`common`](../../../src/nimloth/training/common/README.md), [`sft2`](../../../src/nimloth/training/sft2/README.md), [`rl`](../../../src/nimloth/training/rl/README.md), [`reconstruction`](../../../src/nimloth/training/reconstruction/README.md) |
| Reconstruction | [`recon/README.md`](../../../src/nimloth/recon/README.md), [`cfm`](../../../src/nimloth/recon/cfm/README.md), [`rcdm`](../../../src/nimloth/recon/rcdm/README.md) |
| Evaluation and utilities | [`eval/README.md`](../../../src/nimloth/eval/README.md), [`util/README.md`](../../../src/nimloth/util/README.md), [`util/cache/README.md`](../../../src/nimloth/util/cache/README.md) |

Tests mirror these boundaries under `tests/{agent,backbone,config,environment,rollout,wm,recon,eval,training}`. Add tests to the owner rather than a generic bucket.
