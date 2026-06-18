# Training module

Unified training logic for Nimloth phases. See `ai_tasks/sft2_phase2_plan.md`.

| Subpackage | Phase | Purpose |
|------------|-------|---------|
| `common/` | — | `dist`, `qwen_batch`, `qwen_tuning`, schedules, metrics, wandb |
| `phase0_vagen/` | 0 | VAGEN defaults loaders, rollout post-hooks |
| `phase1_sft/` | 1 | SFT1 LM CE, checkpoints |
| `sft2/` | 2 | `trainer`, `step`, `checkpoint`, `evaluate`, `loss`, `cli` |

World-model modules live in `nimloth.wm/` (see `src/nimloth/wm/README.md`).

`experiments/training/sft2/train.py` is a thin wrapper around `nimloth.training.sft2.trainer`.
