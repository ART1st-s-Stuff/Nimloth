# SFT2

SFT2 包只保留阶段算法、训练生命周期、数据与 checkpoint。文件按可独立阅读的
执行阶段组织，不再把一次 batch 横向拆成 components/objective/schedule。

SFT2 是离线初始化阶段：它只消费 VAGEN 已生成的 trajectory，不运行
`AgentRuntime`，不执行 environment action，也不使用 RL 的多步 WM planner。
它产出的 StateProjector、WM predictor 和 ValueHead checkpoint 用作 RL 在线规划的
warm start。SFT2 与 RL 的 `history_size` 含义和 checkpoint 形状必须一致；
RL 只在真实 rollout 时用独立的 planning horizon 自回归预测多个未来 step。
输入JSONL必须先迁移为当前versioned trajectory格式；训练dataset不解析旧`messages`
布局或字段别名。preprocess cache只接受`dedup_sharded_v2`，v1/逐transition cache需从
迁移后的JSONL重新构建，因为旧cache缺少当前terminal next-state encoding。

| 文件 | 职责 |
|------|------|
| `algorithm.py` | 唯一的 CE/WM/value/SIGReg 核心流程，以及可配置附加 loss 契约 |
| `dino_grid.py` | 只把 cached 4x4 DINO target 装入 SFT2 batch，不实现训练流程 |
| `trainer.py` | 按执行顺序加载 Agent、设置 DDP/视觉 Backbone EMA/optimizer 并启动训练 |
| `batch.py` | SFT2 action/return/next-state 对齐与 terminal mask 装配 |
| `data/` | dataset、sampler 与 DataLoader |
| `history_cache.py` | rank-local 在线 detached state cache 与 checkpoint 状态 |
| `runtime.py` | Agent、固定下一状态监督值、梯度更新与 Qwen LR 运行期 |
| `loop.py` | epoch/microbatch 驱动、resume cursor 和 validation 边界 |
| `evaluate.py` | validation 与分布式指标聚合 |
| `reporting.py` | CSV、W&B 与 epoch 摘要 |
| `checkpoint.py` | SFT2 artifact、恢复状态与保存触发策略 |
| `mcts_evaluation.py` | pre-RL MCTS评估所需的完整checkpoint、H/K和value语义门禁 |
| `diagnosis/` | 不进入生产训练的 packed/KV 等价性诊断 |

`SFT2Algorithm` 是普通 Python 算法对象，不注册参数，也不处理 processor、DDP、
optimizer、EMA 或 checkpoint。`SFT2ModelRuntime` 统一持有训练侧 Agent、在线历史
cache、固定下一状态监督值与 Backbone EMA；不等长分布式验证只解除这个 runtime
的模型包装。
公共 Backbone input builder 只负责 prompt 到张量的转换；`SFT2BatchAssembler`
负责把 DataLoader 的扁平行恢复成 `SFT2Batch(B,T)`（`1<=T<=H`），并分别装配在线 current、
在线 final state 与冻结 Backbone 的 next-state target。

一个样本的统计单位是当前 transition，不是窗口内的每个位置。CE、WM、value 与
SIGReg 每个 current step 各计算一次；T 只提供最长 H 的真实因果上下文。sampler 把
完整 trajectory lane 固定给一个 rank 并按时间顺序推进。每个 state 只在它作为
current step 时执行一次在线 Qwen，随后以 detached CPU tensor 写入 cache；未来窗口
直接读取该 state，不重算历史 Qwen，也不把梯度传回更老时间点。episode 开头使用
T=1..H-1 的短上下文，因此每个拥有真实 next state 的 transition 每 epoch 恰好作为
一次 current step。cache 每个 epoch/validation phase 清空；epoch 内 checkpoint 会
保存每个 rank 的 cache，使恢复后的 sampler cursor 能继续命中历史。

SIGReg 只在训练阶段计算，每个 step 只接收在线 `(s_t,s_{t+1})`；
WM target state 不进入这个统计量。每个 microbatch 先对唯一一次 CE/WM/value
执行 backward 并释放 `s_t` 的 Qwen 图，再编码在线 `s_{t+1}`。SIGReg 的 `s_t`
输入强制 detach，只有 `s_{t+1}` 接收该正则的梯度。各 rank 只 all-gather 小型
state，并用 valid mask 排除 sampler padding；SIGReg 的 `B` 是该 microbatch 的
全局有效 current-step 数。在线 next state 使用可微 gather，把梯度送回来源 rank；
所有 rank 使用相同随机投影。只有 global `B<2` 才跳过 SIGReg。验证集不计算
SIGReg。配置中的 `batch_size` 仍表示每个 rank 的 current step 数。

DINO-grid 使用 k=16 query slots 和 row-major 4x4 teacher tokens，但不拥有第二套
SFT2 algorithm。SFT1 用 DINO target 预训练 `SharedSlotProjector`，SFT2 加载后继续
训练同一个 projector；它的输出就是 WM state，不再串接第二层 state encoder，也没有
WM EMA target encoder。`SFT2Algorithm` 始终执行同一套 current/target、CE、WM、value
与 SIGReg；DINO-grid batch 显式携带 `dino_grid_target`，算法直接比较 predicted next
state 与 next-image cached target 的 MSE，再乘 `lambda_dino`。这两个 WM loss
由 `training/common/world_model.py` 与 RL 共用；teacher cache 由
`backbone/dino_grid.py` 校验，temporal-spatial predictor 由 `wm/grid.py` 拥有。
SFT2 的下一状态监督分支同时冻结 Backbone 与共享 projector。projector 只在 state
作为 current state 时训练；监督值不能反向靠近 predicted state。这里没有第二套
target 参数或 WM EMA。

SFT2 的 value 目标是完整 episode 上先计算、再切到 current step 的 Monte
Carlo return。`training/common/value.py` 与 RL 共用同一个基础objective；当前SFT2
只回归实际执行动作的MC return，rank loss已关闭且未执行action slot没有直接loss。
T步rollout中，WM/DINO监督`[hat{s}_{t+1},...,hat{s}_{t+T}]`，value则监督标准
outgoing Q的决策state序列`[s_t,hat{s}_{t+1},...,hat{s}_{t+T-1}]`与
`[a_t,...,a_{t+T-1}]`。两条序列相差一个时间位置，禁止把incoming action配到
它产生的successor state。

## Pre-RL MCTS rollout evaluation

`experiments/training/sft2/eval_mcts_rollout.py`只接受epoch-complete的完整HF SFT2
checkpoint。入口从`training_state.pt.training_invariants`读取`history_size`和
`prediction_horizon=K`，要求DINO-grid、`history_size=1`和
`decision_state_executed_action_mc_v3`，随后每个真实environment step重新生成
Qwen CoT/state并执行K-step UCT-MCTS。leaf evaluation严格使用
`Q(predicted_state_{K-1}, final_simulated_action)`；各深度的MC return不相加。
评估不会进入RL optimizer，也不会生成fresh-consumption manifest。
