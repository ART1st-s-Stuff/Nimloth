# E0022：PyTorch 2.8 DDP static graph 不能配合 `no_sync()` 首轮累积

错误：SFT2 使用 `DDP(static_graph=True)`，同时在 gradient accumulation 的非同步 micro-batch 外包 `model.no_sync()`。服务器 PyTorch 2.8 会在第一次 backward 触发 reducer 的 `expect_autograd_hooks_ INTERNAL ASSERT FAILED`，无法完成 step1。

原因：这是 PyTorch DDP 已确认的 upstream regression；`no_sync` 跳过 `prepare_for_backward()`，但 static-graph sink 仍在 backward 调用 `finalize_backward()`。

正确做法：禁止把 `static_graph=True` 与 `no_sync()` 组合使用。若DDP模块本身需要repeated-forward/checkpointing的static graph，则每个micro-batch同步。pair-parallel路径已把Qwen移出DDP；其StateProjector/WM/value auxiliary图不需要static graph，因此可令aux DDP `static_graph=False`，在前7个micro使用`no_sync()`并只在GA8边界同步。该优化必须先通过真实PyTorch2.8多rank、每micro多forward、混合aux-device smoke。升级PyTorch后也必须重新做多卡数值gate。
