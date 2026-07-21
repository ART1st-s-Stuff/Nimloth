# Training module

Unified training logic for Nimloth phases. See `ai_tasks/sft2_phase2_plan.md`.

| Package / subpackage | Purpose |
|----------------------|---------|
| `nimloth.backbone/qwen25vl/` | Qwen2.5-VL batching, latent extraction, tuning, vision EMA |
| `nimloth.wm/` | WM models, transition data, and dataset statistics |
| `nimloth.recon/` | Post-hoc CFM and RCDM reconstruction models |
| `nimloth.eval/` | Model-dependent offline evaluation and reconstruction diagnostics |
| `nimloth.model` | 完整 `NimlothModel(llm, wm)` 模型组合 |
| `nimloth.agent/` | Agent prompt、transcript、policy 和 environment runner |
| `nimloth.rollout/` | 跨训练阶段使用的 rollout schema、collector 和存储 |
| `nimloth.config/` | Agent、rollout、SFT2 和 RL 配置 |
| `nimloth.util/` | dist、schedule、profiling、cache、metrics、W&B |
| `training/sft2/` | SFT2 数据、梯度策略、训练循环、验证和 checkpoint |
| `training/rl/` | RL loss、rollout iteration、验证和 checkpoint |
| `training/sft2/diagnosis/` | Packed/KV trajectory equivalence diagnostics |
| `training/reconstruction/` | post-hoc diagnostic image decoder training; freezes Qwen/WM |

`experiments/training/sft2/train.py` is a thin wrapper around `nimloth.training.sft2.trainer`.
