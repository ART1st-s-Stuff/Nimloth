# Training experiments (by phase)

**实验规则入口：** [`ai_rules/03_experiments_and_data.md`](../../ai_rules/03_experiments_and_data.md) · [`experiments/README.md`](../README.md)

| Directory | Phase | Description |
|-----------|-------|-------------|
| `baseline/` | 0 | **VAGEN navigation RL baseline** (canonical) |
| `sft1/` | 1 | **Format SFT (SFT1)** (canonical) |
| `phase0_vagen/` | 0 | Planned: additional rollout collection helpers |
| `sft2/` | 2 | WM + Value alignment (SFT2) |

SFT2 `train.py` 为薄入口，调用 `nimloth.training.sft2.trainer`；WM 在 `wm/`；Qwen 调参在 `backbone/`；离线 eval 在 `eval/`。

VAGEN baseline → `experiments/training/baseline/`；SFT1 → `experiments/training/sft1/`。
`navigation_baseline/` 为遗留目录（runs 数据暂留），勿新增脚本。
