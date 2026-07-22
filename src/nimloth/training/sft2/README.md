# SFT2

SFT2 包只保留阶段算法、训练生命周期、数据与 checkpoint。文件按可独立阅读的
执行阶段组织，不再把一次 batch 横向拆成 components/objective/schedule。

SFT2 是离线初始化阶段：它只消费 VAGEN 已生成的 trajectory，不运行
`AgentRuntime`，不执行 environment action，也不使用 RL 的多步 WM planner。
它产出的 StateProjector、WM predictor 和 ValueHead checkpoint 用作 RL 在线规划的
warm start。SFT2 与 RL 的 `history_size` 含义和 checkpoint 形状必须一致；
RL 只在真实 rollout 时用独立的 planning horizon 自回归预测多个未来 step。

| 文件 | 职责 |
|------|------|
| `algorithm.py` | 纯单批计算：Agent/target forward、全部 loss 和 WM 权重策略 |
| `trainer.py` | 按执行顺序加载 Agent、设置 DDP/EMA/optimizer 并启动训练 |
| `batch.py` | SFT2 action/return/next-state 对齐与 terminal mask 装配 |
| `data/` | dataset、sampler 与 DataLoader |
| `runtime.py` | Agent/target 执行视图、梯度更新与 Qwen LR 运行期 |
| `loop.py` | epoch/microbatch 驱动、resume cursor 和 validation 边界 |
| `evaluate.py` | validation 与分布式指标聚合 |
| `reporting.py` | CSV、W&B 与 epoch 摘要 |
| `checkpoint.py` | SFT2 artifact、恢复状态与保存触发策略 |
| `diagnosis/` | 不进入生产训练的 packed/KV 等价性诊断 |

`SFT2Algorithm` 是普通 Python 算法对象，不注册参数，也不处理 processor、cache、
DDP、optimizer、EMA 或 checkpoint。`SFT2ModelRuntime` 统一持有训练侧 Agent、
target-state 梯度路径与 Backbone EMA；不等长分布式验证只解除这个 runtime 的
模型包装。
公共 Backbone input builder 只负责 prompt 到张量的转换；`SFT2BatchAssembler`
负责把 DataLoader 的扁平行恢复成 `SFT2Batch(B,H)`，并分别装配在线 current、
在线 final state 与冻结 Backbone 的 next-state target。

SIGReg 只在训练阶段计算。每个窗口用同一个在线 Backbone 编码真实状态
`s_0 ... s_H`，`SequenceSIGReg` 接收 `(B,H+1,D)`；EMA/target state 不进入这个
统计量。`B<2` 时保留其他目标并记录该批跳过 SIGReg；验证集不计算 SIGReg。
`TrajectoryWindowBatchSampler` 明确定义窗口，`batch_size` 表示窗口数 `B`。
