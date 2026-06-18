# Training experiments (by phase)

**实验规则入口：** [`ai_rules/03_experiments_and_data.md`](../../ai_rules/03_experiments_and_data.md) · [`experiments/README.md`](../README.md)

| Directory | Phase | Description |
|-----------|-------|-------------|
| `phase0_vagen/` | 0 | VAGEN rollout & RL collection |
| `phase1_sft/` | 1 | Format SFT (SFT1) |
| `sft2/` | 2 | WM + Value alignment (SFT2) |

SFT2 `train.py` 为薄入口，调用 `nimloth.training.sft2.trainer`；WM 在 `wm/`；Qwen 调参在 `backbone/`；离线 eval 在 `eval/`。

SFT1 and VAGEN rollout scripts remain in `experiments/navigation_baseline/`.
See `ai_tasks/sft2_phase2_plan.md` for the migration map.
