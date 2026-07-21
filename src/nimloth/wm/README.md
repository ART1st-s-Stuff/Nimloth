# World Model

`nimloth.wm` 只定义可训练世界模型及规划能力。

| 文件 | 职责 |
|------|------|
| `model.py` | `WorldModel`：组合 StateProjector、WMPredictor、ValueHead |
| `state_proj.py` | backbone hidden → WM state |
| `predictor.py` | latent 下一状态预测 |
| `sigreg.py` | 可供单步训练阶段复用的 ``(T=2,B,D)`` SIGReg 输入契约 |
| `value_head.py` | 每个离散动作的 value |
| `lewm.py`、`_vendor_lewm.py` | LeWM 配置和最小核心算子 |
| `planning.py` | WM 规划 |
| `reconstruction.py` | post-hoc reconstruction 诊断模型 |

`WorldModel.forward()` 只做神经网络计算。各训练阶段保留自己的 stop-gradient、
ranking 和 loss 权重策略；当前 SFT2 的单步 SIGReg 时间轴由
`OneStepSIGReg` 负责。RL 的多步契约不能直接套用这一模块。
