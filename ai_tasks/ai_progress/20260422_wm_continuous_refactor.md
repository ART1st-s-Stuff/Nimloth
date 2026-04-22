# WM连续动作与训练范式重构进度（2026-04-22）

## 当前目标
- 完成AI2THOR连续动作采样改造，并将WM训练升级为Transformer + 逆动力学双范式。

## 已完成事项
- 数据标签升级为连续动作三元组：`move_ahead_distance`、`delta_yaw`、`delta_pitch`。
- 采集配置新增连续动作范围、OU过程、防撞参数、NavMesh rollout参数，并接入Hydra配置映射。
- AI2THOR适配器新增连续动作执行、中心深度估计、可达点读取接口。
- 采集器新增OU趋势采样、防撞策略、失败动作强制旋转、NavMesh混合rollout逻辑。
- 训练数据集升级为`K`帧历史序列样本输出：`z_history`、`action_history`、`z_next`、`gt_action`。
- WM主干改为Transformer时序建模；新增逆动力学模型模块。
- 训练流程支持`unsupervised`与`semi_supervised`两种模式，并支持损失权重与梯度裁剪配置。
- 通过`py_compile`完成改动文件语法检查；`ReadLints`未发现新增lints问题。

## 阻塞问题
- 未执行AI2THOR在线采集和全量训练回归，当前仅完成本地轻量级链路验证。

## 下一步计划
- 在具备AI2THOR运行环境的机器上执行采集冒烟与短训练验证（两种training_mode各1轮）。
- 按实验结果微调防撞概率、NavMesh占比和半监督损失权重。
