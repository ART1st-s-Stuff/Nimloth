# SFT2 library

SFT2 aligns Qwen latent states with a one-step world-model predictor and an
action-value head. This package owns phase-specific orchestration; reusable
Qwen and world-model concepts stay outside it.

| Module | Responsibility |
|--------|----------------|
| `nimloth.config.sft2`, `cli.py` | Strict SFT2 YAML schema and CLI adapter |
| `components.py` | Model/head construction, placement, DDP, EMA, optimizer |
| `data/` | Transition batch protocol, samplers, and loaders |
| `engine.py` | Shared train/validation forward contract |
| `loop.py` | Resumable micro-batch loop, validation, and checkpoint policy |
| `step.py` | One-step WM and value computation |
| `objectives.py` | Tensor-level losses and schedules |
| `evaluate.py` | Validation loop and distributed metric aggregation |
| `checkpoint.py` | SFT2 artifact set, resume state, save manager |
| `utils.py` | Small runtime helpers shared by training and validation |
| `diagnosis/` | Non-production packed/KV equivalence investigations |

预处理 cache、profiling、CSV、W&B 和分布式工具由 `nimloth.util` 负责，
不属于 SFT2 的训练语义。

For structured rollout records, transcript and action-prompt construction are
owned by `nimloth.agent`. SFT2 expands each action into a supervised current
prefix and a policy-query next prefix using the trajectory's registered
`AgentPromptTemplate`. Legacy JSONL
records without `system_prompt`/`observation_texts` remain readable through
their stored `messages`, but new data should use the structured Agent schema.

Dependency direction:

```text
agent (transcript/prompt/action contract)
  + wm (transition/model concepts)
  + backbone/qwen25vl (Qwen adapters)
        -> training/sft2 (phase orchestration)
              -> experiments/training/sft2 (thin entry points)
```

Outer reconstruction/evaluation modules consume `nimloth.wm` transition types
and `nimloth.backbone.qwen25vl` adapters directly. They must not import SFT2's
private data implementation.

Training and validation call the same `SFT2StepRunner.forward`; validation only
changes mode/gradient/EMA policy. Checkpoint selection uses model-derived
`val_wm_mse`. Static rollout labels are available from `nimloth.wm.statistics`
for dataset inspection only.
