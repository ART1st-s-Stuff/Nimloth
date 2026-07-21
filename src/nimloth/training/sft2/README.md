# SFT2

SFT2 包只保留阶段算法、训练生命周期、数据与 checkpoint。文件按可独立阅读的
执行阶段组织，不再把一次 batch 横向拆成 components/objective/schedule。

| 文件 | 职责 |
|------|------|
| `algorithm.py` | 完整单批计算：Agent/target forward、全部 loss 和 WM 权重策略 |
| `trainer.py` | 按执行顺序加载 Agent、设置 DDP/EMA/optimizer 并启动训练 |
| `data/` | dataset、sampler 与 DataLoader |
| `loop.py` | 微批、backward、optimizer、EMA、验证和保存时机 |
| `evaluate.py` | validation 与分布式指标聚合 |
| `checkpoint.py` | SFT2 artifact 与恢复状态 |
| `diagnosis/` | 不进入生产训练的 packed/KV 等价性诊断 |

`algorithm.py` 不导入 Qwen，也不处理 processor、cache、DDP、EMA 或 checkpoint。
Qwen batch 在进入算法前被转换成 `AgentBatch`；terminal transition
通过 mask 参与统一调用结构，不再需要 `_compute_wm` 或 DDP dummy-loss 分支。
