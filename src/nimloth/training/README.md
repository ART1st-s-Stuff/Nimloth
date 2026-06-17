# Training module

Unified training logic for Nimloth phases. See `ai_tasks/sft2_phase2_plan.md`.

| Subpackage | Phase | Purpose |
|------------|-------|---------|
| `common/` | — | Qwen tuning, LR schedules, metrics, wandb |
| `phase0_vagen/` | 0 | VAGEN defaults loaders, rollout post-hooks |
| `phase1_sft/` | 1 | SFT1 LM CE, checkpoints |
| `sft2/` | 2 | WM predictor, Value head, combined losses |
