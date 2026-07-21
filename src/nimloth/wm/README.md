# World Model

`nimloth.wm` 只定义可训练世界模型及规划能力。

| 文件 | 职责 |
|------|------|
| `model.py` | `WorldModel`：组合 StateProjector、WMPredictor、ValueHead |
| `state_proj.py` | backbone hidden → WM state |
| `predictor.py` | latent 下一状态预测 |
| `value_head.py` | 每个离散动作的 value |
| `lewm.py`、`_vendor_lewm.py` | LeWM 配置和最小核心算子 |
| `planning.py` | WM 规划 |
| `reconstruction.py` | post-hoc reconstruction 诊断模型 |

`WorldModel.forward()` 只做神经网络计算。SFT2 与 RL 的 loss 分别属于各自的
`objective.py`，因为两个阶段的 stop-gradient、ranking 和权重策略不同。
