# SFT2

SFT2 包只保留阶段专用的目标函数、训练循环、评估和 checkpoint 策略。

| 文件 | 职责 |
|------|------|
| `algorithm.py` | current Agent → target state → objective 的三步编排 |
| `objective.py` | LM、WM、SIGReg、value loss 与指标 |
| `schedule.py` | WM loss 权重调度 |
| `components.py` | Agent、batch builder、target runtime、DDP 和 optimizer 装配 |
| `data/` | dataset、sampler 与 DataLoader |
| `loop.py` | 微批、backward、optimizer、EMA、验证和保存时机 |
| `evaluate.py` | validation 与分布式指标聚合 |
| `checkpoint.py` | SFT2 artifact 与恢复状态 |
| `diagnosis/` | 不进入生产训练的 packed/KV 等价性诊断 |

`algorithm.py` 不导入 Qwen，也不处理 processor、cache、EMA、DDP、optimizer 或
checkpoint。Qwen batch 在进入算法前被转换成 `AgentBatch`；terminal transition
通过 mask 参与统一调用结构，不再需要 `_compute_wm` 或 DDP dummy-loss 分支。
