# E0033：把 SIGReg 的时间轴误当成整条 trajectory

## 已确认错误

SFT2 曾把每条变长 trajectory 单独整理为 `(T_i, 1, D)`，逐条调用 SIGReg。
发现 `B=1` 退化后，又一度把所有训练强制解释为 `history_size=1`，固定构造
`T=2`。前者混淆了时间轴和样本轴，后者掩盖了 LeWM 连续上下文训练缺失的问题。

## 错误原因

LeWM 的 `T` 由 `history_size + prediction_offset` 决定，不是 rollout 的完整长度，
也不能为了适配单 transition 接口而退化为 2。当前一步偏移下，SFT2/RL 都应使用
`T=H+1` 的真实状态窗口；microbatch 中的窗口样本组成 `B`。

## 正确做法

- SFT2/RL 都先构造 H 个连续动作和 H+1 个真实状态，再向 SIGReg 传入
  `(B,H+1,D)`；模块内部才转为 LeWM 的 `(H+1,B,D)`。
- SIGReg 的全部状态必须来自同一个在线 encoder。EMA/target encoder 只服务于
  WM target，不能补充 SIGReg 的最后一个状态。
- `B<2` 时明确跳过并记录，普通 gradient accumulation 不能恢复一次 `B>1` 的统计。
- sampler 必须产出固定长度连续窗口；不能只调大 predictor 配置值，也不能用重复
  状态、zero action 或不相关 transition 凑满上下文。
