# Evaluation (`nimloth.eval`)

离线评估指标与脚本所用库代码（不含 VAGEN 在线 rollout）。

| 文件 | 内容 |
|------|------|
| `rollout.py` | Nimloth jsonl 轨迹级成功率等 |
| `reconstruction.py` | WM reconstruction diagnostic：oracle / predicted / copy / shuffled-action 对比 |
| `rcdm_reconstruction.py` | 从 SFT2 true / WM-predicted latent state 采样 RCDM 可视化 |

实验入口示例：`experiments/training/sft2/eval_val_rollout_success.py`。
