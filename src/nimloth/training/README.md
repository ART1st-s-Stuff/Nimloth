# Training module

Unified training logic for Nimloth phases. See `ai_tasks/sft2_phase2_plan.md`.

| Package / subpackage | Purpose |
|----------------------|---------|
| `nimloth.backbone/qwen25vl/` | Qwen2.5-VL batching, latent extraction, tuning, vision EMA |
| `nimloth.wm/` | WM models, transition data, and dataset statistics |
| `nimloth.recon/` | Post-hoc CFM and RCDM reconstruction models |
| `nimloth.eval/` | Model-dependent offline evaluation and reconstruction diagnostics |
| `training/common/` | dist, schedules, metrics, wandb |
| `training/phase0_vagen/` | Phase 0 hooks |
| `training/phase1_sft/` | Phase 1 SFT |
| `training/sft2/` | SFT2 configuration, components, data plane, shared step engine, objectives, evaluation, and checkpoints |
| `training/sft2/diagnosis/` | Packed/KV trajectory equivalence diagnostics |
| `training/reconstruction/` | post-hoc diagnostic image decoder training; freezes Qwen/WM |

`experiments/training/sft2/train.py` is a thin wrapper around `nimloth.training.sft2.trainer`.
