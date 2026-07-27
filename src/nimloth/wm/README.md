# World Model

`nimloth.wm` 只定义可训练世界模型及 latent action sequence 模拟能力。
在线搜索策略属于 `nimloth.agent.planning`。

| 文件 | 职责 |
|------|------|
| `model.py` | `WorldModel`：组合 StateProjector、WMPredictor、ValueHead |
| `state_proj.py` | backbone hidden → WM state |
| `predictor.py` | latent 下一状态预测与自回归 sequence 模拟 |
| `grid.py` | 可在 SFT2 继续训练的 k16 SFT1 projector 与 H-step temporal-spatial predictor |
| `sigreg.py` | SFT2/RL 共用的 ``(B,T,D)`` sequence SIGReg |
| `value_head.py` | 每个离散动作的 value |
| `lewm.py`、`_vendor_lewm.py` | LeWM 配置和最小核心算子 |
| `reconstruction.py` | post-hoc reconstruction 诊断模型 |

`WorldModel.forward()` 只做神经网络计算。各训练阶段保留自己的 stop-gradient、
ranking 和 loss 权重策略；SFT2 与 RL 都让 `SequenceSIGReg` 消费真实的
`H+1` 状态序列。RL 的未来规划长度由 Agent planning horizon 单独控制。

`GridWorldModel` 保留同一个公共 state/predict/value 接口。它直接把 SFT1
`SharedSlotProjector` 的输出作为 grid state，并在 SFT2 继续训练该 projector；
DINO teacher identity、target cache 和 direct predicted-state loss 仍分别属于
backbone cache 与 SFT2 objective。WM 本身不维护 EMA 参数。
