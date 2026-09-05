# E0038：归因 OOM 前核实 sampler 的实际 microbatch

## 错误

看到配置 `batch_size=2` 和首步 OOM 后，直接判断降低为 per-rank batch 1 会降低
该步显存，没有核实图片预算 sampler 是否已把实际 microbatch 限制为一个 window。

## 后果

batch1 smoke 使用了与失败 attempt 基本相同的单-window forward，再次在 Qwen
全词表 FP32 causal-LM CE 阶段 OOM，浪费一次 8-GPU 初始化时间。

## 正确做法

对 trajectory/image-budget 训练，先核实实际 window count、时间长度、prefix 图片数
和序列长度。`history_size=H` 时，一个 window 本身含 H 个连续状态；若单 window
已 OOM，调整配置 batch size 或 grad accumulation 都不会降低该 forward 的峰值，
必须处理单-window 计算图或明确改变输入/损失语义。
