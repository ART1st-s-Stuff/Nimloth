# SFT3：世界模型与价值头训练

本目录迁入原 `training/sft2` 的完整实现；公共 `SFT2*` 类型、函数和
checkpoint 标识保留历史名称，避免目录迁移改变恢复及下游加载合同。
这里的 SFT3 不等于新 SFT2 的 query/DINO 对齐阶段。

训练入口为 `python -m nimloth.training.sft.stage3 --config <原 WM/value 配置>`，
调用 `trainer.main()` / `train_sft2()`。继续使用 `nimloth.config.sft2` 的已验证配置。
迁移未改变默认配置、模型结构、损失权重、优化器或 checkpoint schema。

## 阅读顺序

1. `trainer.py` 构建 Agent、模型包装、EMA、优化器和数据，再启动 `loop.py`。
2. `data/factory.py` 选择 sampler；`data/samplers.py` 定义真实轨迹的采样单位。
3. `batch.py` 对齐起点、执行动作、完整 episode 的 MC return 和后继观测；
   `dino_grid.py` 装配冻结 teacher cache 的空间 target。
4. `algorithm.py` 组织主损失及独立 SIGReg 的计算/反传顺序；`sigreg.py` 负责跨 rank 的有效样本汇聚、可微通信和同步随机投影；`runtime.py` 管理目标编码、反传和更新。
5. `loop.py` 驱动 microbatch、恢复游标和验证；`reporting.py` 汇总指标；
   `checkpoint.py` 保存模型、优化器、EMA 与各 rank 的历史缓存。

## 多步窗口与旧单步模式

`prediction_horizon=T>1` 使用 `FutureRolloutBatchSampler`，要求 `history_size=1`。
每个窗口含同一 episode 的 T 个连续执行动作及 T+1 个真实观测，只编码起点
作为在线递推输入。`SFT2Algorithm._rollout_step` 调用 Agent 的真实多步前向：
在动作前状态评分 outgoing Q，再预测后继状态。WM/DINO 对齐全部 T 个后继状态，
value 对齐 T 个实际执行动作的 MC return；这些 return 在完整 episode 上计算，
不在窗口尾截断。CE 只来自窗口起点。采样采用固定长度滑窗，短于 T 的片段不产生
窗口，不补造后继状态，也不跨越 episode 或缺失后继观测的位置。

`prediction_horizon=1` 保留 `OnlineHistoryBatchSampler` 及 H 步历史兼容模式。
此模式的统计单位是当前 transition，历史仅提供因果上下文；旧历史 state 从
rank-local `history_cache.py` 读取 detached tensor。这与 T 个未来预测步不同。

目标状态编码沿用冻结 backbone（可使用已有 backbone EMA）和共享 projector 的
无梯度分支。主损失反传完成后才编码相邻在线状态做 SIGReg；起点 detach，梯度只进入
新状态侧。跨 rank 的有效样本统计、padding 零权重及随机投影同步保持原实现。

## 验证、诊断和兼容

`evaluate.py` 负责离线 loss 验证，`mcts_evaluation.py` 提供 MCTS checkpoint 及 value 语义校验；真实环境评估统一由 [`../evaluation/`](../evaluation/README.md) 提供。

专项 canary、动作头修复、特征定位审计和 packed/KV 研究原型位于 `experiments/training/sft/diagnosis/`。这些工具可调用训练组件，训练组件不依赖实验工具。

旧 `nimloth.training.sft2` Python 包已移除，代码调用者应使用 `nimloth.training.sft.stage3`。既有 `SFT2*` 类型名、配置字段和 `decision_state_executed_action_mc_v3` 仍描述相同 WM/value 目标，不会被新 Query 对齐阶段静默接受。历史产物不改写；通过实际 checkpoint 加载与恢复测试校验迁移。
