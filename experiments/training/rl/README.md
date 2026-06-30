# RL 训练实验

从 SFT2 checkpoint warm-start，在线 RL 训练 WM predictor + value head。

## 脚本

| 文件 | 用途 |
|------|------|
| `smoke_test.slurm` | 单 GPU smoke test：加载 SFT2 checkpoint，synthetic data 跑 1 步训练 |

## 输出

```
outputs/experiments/training/rl/<date>/<name>/
├── README.md
├── train_step_log.csv
├── best/                  # best checkpoint (state_proj, predictor, value_head, optimizer)
├── iter_NNNN/             # periodic checkpoints
├── rollouts/              # per-iteration trajectory JSONL
└── final/                 # final checkpoint
```

## 入口

```bash
# Smoke test (单 GPU，synthetic data)
sbatch experiments/training/rl/smoke_test.slurm

# 完整在线 RL（后续，需要 VAGEN env server）
sbatch experiments/training/rl/train_online.slurm
```
