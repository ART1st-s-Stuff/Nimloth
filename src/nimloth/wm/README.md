# World Model

`nimloth.wm` 只定义可训练世界模型及 latent action sequence 模拟能力。
在线搜索策略属于 `nimloth.agent.planning`。

| 文件 | 职责 |
|------|------|
| `model.py` | `WorldModel`：组合组件并声明 optimizer、broadcast、DDP 和阶段冻结接口 |
| `factory.py` | checkpoint loader registry；训练阶段只依赖这个公共恢复入口 |
| `state_proj.py` | backbone hidden → WM state |
| `predictor.py` | latent 下一状态预测与自回归 sequence 模拟 |
| `grid.py` | k16 shared-slot state、EMA target、H-step temporal-spatial predictor 与 DINO decoder |
| `grid_factory.py` | grid checkpoint 识别、artifact 完整性、组件重建和 RL 冻结策略 |
| `sigreg.py` | SFT2/RL 共用的 ``(B,T,D)`` sequence SIGReg |
| `value_head.py` | 每个离散动作的 value |
| `lewm.py`、`_vendor_lewm.py` | LeWM 配置和最小核心算子 |
| `reconstruction.py` | post-hoc reconstruction 诊断模型 |

`WorldModel.forward()` 只做神经网络计算。各训练阶段保留自己的 stop-gradient、
ranking 和 loss 权重策略；SFT2 与 RL 都让 `SequenceSIGReg` 消费真实的
`H+1` 状态序列。RL 的未来规划长度由 Agent planning horizon 单独控制。

`GridWorldModel` 保留同一个公共 state/predict/value 接口，并通过多态组件接口
声明额外 decoder/EMA 的同步和冻结语义。DINO teacher identity、target cache 和
loss 仍分别属于 backbone cache 与 SFT2 objective；公共 SFT2/RL trainer 和
checkpoint 不识别 `GridWorldModel` 或 DINO artifact 文件名。
