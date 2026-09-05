# E0058：不要用手写梯度同步绕过未定位的 DDP 问题

错误：RL 的多设备 DDP 出现 collective 顺序分叉后，直接移除了 DDP wrapper，并在
`OptimizationRuntime` 中按 optimizer 参数顺序逐个调用 `all_reduce`。这绕开了当时的
死锁，却自行承担了参数一致性、缺失梯度、通信顺序和 reducer 生命周期等框架职责。

原因：当时只确认多个 DDP reducer 在同一次 loss backward 中发生了 collective 序列
分叉，没有先把完整 loss 收敛到一个官方 DDP forward/reducer 边界，就把框架问题误当成
了重写同步机制的理由。

正确做法：多设备 Qwen、WorldModel 和 TokenValueHead 共同产生一个 RL loss 时，把它们
注册到同一个 training-step `nn.Module`，并只用一次官方
`DistributedDataParallel(device_ids=None)` 包装该完整 forward。删除 optimizer 内的手工
梯度通信；在真实多 rank GPU 门禁通过前，只能称为待验证修复，不能宣称分布式语义正确。
