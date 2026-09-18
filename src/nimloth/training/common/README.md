# Training common

本目录只放 SFT2 与 RL 语义完全相同的训练组件。当前公共组件是：

- `value.py`：ValueHead 对实际执行动作的 Monte Carlo return 回归，以及可选的动作排序约束。
- `activation_offload.py`：用 PyTorch `save_on_cpu` 在反传前暂存计算图张量；不改变阶段目标或参数位置。

阶段特有的 batch、模型前向、loss 权重和反传顺序仍由各自的 algorithm 管理。
