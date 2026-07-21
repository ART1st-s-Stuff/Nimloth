# E0033：把 SIGReg 的时间轴误当成整条 trajectory

## 已确认错误

SFT2 曾把每条变长 trajectory 单独整理为 `(T_i, 1, D)`，逐条调用 SIGReg。
发现 `B=1` 退化后，又错误地计划用多条等长 trajectory 窗口构造动态 `T`。
两种实现都没有先对齐当前 WM 的单步训练契约。

## 错误原因

LeWM 的 `T` 由 `history_size + prediction_offset` 决定，不是 rollout 的完整长度。
本次修复范围内的 SFT2 是一状态预测下一状态，因此每个 transition 已经提供
`[s_t, s_{t+1}]`，固定 `T=2`；microbatch 中的 transition 样本才组成 `B`。

## 正确做法

- 单步 WM 的一次 SIGReg 输入固定为 `(2, B, D)`。
- 第 0 个时间位置是所有有效 transition 的 current state，第 1 个位置是对应的
  next state；不能对每条 trajectory 分别用 `B=1` 调用。
- `B<2` 时明确跳过并记录，普通 gradient accumulation 不能恢复一次 `B>1` 的统计。
- 如果未来真正实现 `history_size>1`，必须先实现连续上下文训练，再按
  `T=history_size+prediction_offset` 重新定义输入；不能只调大配置值。
