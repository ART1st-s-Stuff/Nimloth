# Evaluation (`nimloth.eval`)

依赖模型输出的离线评估、重建诊断，以及真实 rollout 证据的离线浏览产物。

| 文件 | 内容 |
|------|------|
| `reconstruction.py` | WM reconstruction diagnostic：oracle / predicted / copy / shuffled-action 对比 |
| `rcdm_reconstruction.py` | 从 SFT2 true / WM-predicted latent state 采样 RCDM 可视化 |
| `rollout_browser/` | 将 VAGEN/SFT behavior-time rollout 证据原子归档为可筛选的离线 HTML |

静态数据集统计位于 `nimloth.wm.statistics`，不能作为模型 checkpoint 指标。
