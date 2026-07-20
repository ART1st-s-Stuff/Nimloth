# Evaluation (`nimloth.eval`)

依赖模型输出的离线评估与重建诊断（不含 VAGEN 在线 rollout）。

| 文件 | 内容 |
|------|------|
| `reconstruction.py` | WM reconstruction diagnostic：oracle / predicted / copy / shuffled-action 对比 |
| `rcdm_reconstruction.py` | 从 SFT2 true / WM-predicted latent state 采样 RCDM 可视化 |

静态数据集统计位于 `nimloth.wm.statistics`，不能作为模型 checkpoint 指标。
