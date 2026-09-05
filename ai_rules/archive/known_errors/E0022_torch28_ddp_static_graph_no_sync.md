# E0022：PyTorch 2.8 DDP static graph 不能配合 `no_sync()` 首轮累积

错误：SFT2 使用 `DDP(static_graph=True)`，同时在 gradient accumulation 的非同步 micro-batch 外包 `model.no_sync()`。服务器 PyTorch 2.8 会在第一次 backward 触发 reducer 的 `expect_autograd_hooks_ INTERNAL ASSERT FAILED`，无法完成 step1。

原因：这是 PyTorch DDP 已确认的 upstream regression；`no_sync` 跳过 `prepare_for_backward()`，但 static-graph sink 仍在 backward 调用 `finalize_backward()`。

正确做法：在当前固定运行时保留 repeated-forward/checkpointing 所需的 static graph，但每个 accumulation micro-batch 都同步梯度，不进入 `no_sync()`。逐 micro-batch all-reduce 与先本地累积再 all-reduce 在数学上线性等价，只增加通信开销。升级到包含 upstream 修复的 PyTorch 后，必须重新做多卡数值 gate 才能恢复 `no_sync()`。
