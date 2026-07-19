# Training module

Unified training logic for Nimloth phases. See `ai_tasks/sft2_phase2_plan.md`.

| Package / subpackage | Purpose |
|----------------------|---------|
| `nimloth.backbone/` | Qwen2.5-VL tuning、vision EMA、冻结 DINOv3 teacher |
| `nimloth.wm/` | WM models, transition data/collate |
| `nimloth.eval/` | Offline rollout metrics |
| `training/common/` | dist, qwen_batch, schedules, metrics, wandb |
| `training/phase0_vagen/` | Phase 0 hooks |
| `training/phase1_sft/` | Phase 1 SFT |
| `training/sft2/` | trainer, step, checkpoint, evaluate, loss, cli；可把 projected current query state 直接对齐当前 RGB 的 DINOv3 CLS 特征 |
| `training/reconstruction/` | post-hoc diagnostic image decoder training; freezes Qwen/WM |

`experiments/training/sft2/train.py` is a thin wrapper around `nimloth.training.sft2.trainer`.
