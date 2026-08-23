# Evaluation (`nimloth.eval`)

依赖模型输出的离线评估、重建诊断，以及真实 rollout 证据的离线浏览产物。

| 文件 | 内容 |
|------|------|
| `reconstruction.py` | WM reconstruction diagnostic：oracle / predicted / copy / shuffled-action 对比 |
| `rcdm_reconstruction.py` | 从 SFT2 true / WM-predicted latent state 采样 RCDM 可视化 |
| `rollout_browser/` | 将 VAGEN/SFT behavior-time rollout 证据原子归档为可筛选的离线 HTML |
| `sft_checkpoint_state_matrix.py` | 在pre-RL validation上只读交叉比较SFT1/ID74 backbone、projector与vision EMA，并审计冻结ID74 WM/ValueHead兼容性 |
| `deployed_actor_sft1_goal_audit.py` | 用真实归档instruction/CoT只读检查ID176 actor与SFT1 projector的视觉兼容性和target-object检索 |

静态数据集统计位于 `nimloth.wm.statistics`，不能作为模型 checkpoint 指标。
