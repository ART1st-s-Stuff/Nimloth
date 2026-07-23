# E0039：禁止在 history forward 中隐藏重复的逐步 loss

## 错误

把重叠的 H-step windows 展平成 `B*H` 行后，在 `Agent.forward_sequence()` 内对
所有行计算 CE，并在 SFT2 algorithm 中对 H 个位置同时计算 WM/value loss。这样
`algorithm.py` 表面只有一个 `lm_loss`，实际却让同一 transition 随窗口重叠最多
重复 H 次，无法从 objective 定义处审查真实统计权重。

## 后果

- CE 不再是标准 SFT 的“一条 transition 每 epoch 一次”；轨迹内部 step 被额外加权。
- WM/value 同样按窗口覆盖次数重复，训练单位从 step 偷换成 window-position。
- H 增大时计算、显存和梯度权重一起隐式增加。

## 正确做法

1. sampler 必须让每个拥有真实 next state 的 transition 恰好作为一次 current step；
   H 只决定该 step 最多能看到多少真实历史。
2. episode 开头使用 T=1..H-1 的真实短上下文；禁止丢弃这些 step 或伪造 padding。
3. `algorithm.py` 必须显式使用 current action/value/next target，只计算一次
   CE、WM、value 和 SIGReg。
4. 历史 Backbone/state 路径在进入 WM predictor 前 detach；只有 current row
   拥有 CE 和 Backbone 梯度。ValueHead 读取当前 `s_t`，更早历史由构造
   `s_t` 的累计 Agent prompt 提供。
5. 测试必须统计每个 transition 的 current-step ownership，并验证旧 history 的
   Backbone 梯度为零。
