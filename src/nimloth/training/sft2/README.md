# SFT2

SFT2 包只保留阶段算法、训练生命周期、数据与 checkpoint。文件按可独立阅读的
执行阶段组织，不再把一次 batch 横向拆成 components/objective/schedule。

| 文件 | 职责 |
|------|------|
| `algorithm.py` | 纯单批计算：Agent/target forward、全部 loss 和 WM 权重策略 |
| `trainer.py` | 按执行顺序加载 Agent、设置 DDP/EMA/optimizer 并启动训练 |
| `data/` | dataset、sampler 与 DataLoader |
| `runtime.py` | Agent/target 执行视图、梯度更新与 Qwen LR 运行期 |
| `loop.py` | epoch/microbatch 驱动、resume cursor 和 validation 边界 |
| `evaluate.py` | validation 与分布式指标聚合 |
| `reporting.py` | CSV、W&B 与 epoch 摘要 |
| `checkpoint.py` | SFT2 artifact、恢复状态与保存触发策略 |
| `diagnosis/` | 不进入生产训练的 packed/KV 等价性诊断 |

`SFT2Algorithm` 是普通 Python 算法对象，不注册参数，也不处理 processor、cache、
DDP、optimizer、EMA 或 checkpoint。`SFT2ModelRuntime` 是在线 Agent 与 target
路径的完整执行契约；不等长分布式验证只解除这个 runtime 的模型包装。
Qwen batch 在进入算法前被转换成公共 `TransitionBatch`；terminal transition
通过 mask 参与统一调用结构，不再需要 `_compute_wm` 或 DDP dummy-loss 分支。

SIGReg 只在训练阶段计算。每个有效 transition 提供 ``[s_t, s_{t+1}]``，公共
`OneStepSIGReg` 固定构造 `(T=2,B,D)`；`B` 是 microbatch 中有下一状态的
transition 数。`B<2` 时保留其他目标并记录该批跳过 SIGReg；验证集不计算
SIGReg。trajectory sampler 只决定哪些 transition 共享 microbatch，不再定义 `T`。
