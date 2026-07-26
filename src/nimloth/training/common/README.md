# Training common

本目录只放 SFT2 与 RL 语义完全相同的训练目标。当前公共目标是：

- `value.py`：ValueHead 对实际执行动作的 Monte Carlo return 回归，以及可选的动作排序约束。

阶段特有的 batch、模型前向、loss 权重和反传顺序仍由各自的 algorithm 管理。
