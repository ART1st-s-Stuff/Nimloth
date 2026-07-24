# E0035：不要在确定梯度策略前预编码 Backbone state

## 已确认错误

RL 曾在采样训练窗口前把整条 trajectory 的 Qwen hidden 统一放在 `no_grad` 下编码并
保存。这样即使 LLM/vision 参数已经解冻，WM、value 和 SIGReg loss 也不可能更新
Backbone；训练参数是否可更新被一次隐藏的预处理决定。

## 错误原因

把参数的 tune mode、PPO actor 是否启用、表征目标是否回传 Backbone 混成了同一个
开关。Qwen adapter 还承担了 RL window/return/detach，导致具体 Backbone 反向依赖
训练阶段语义。

## 正确做法

- rollout 只保存原始 trajectory，并先按连续边界采样 `H+1` 个状态的窗口。
- 训练 runtime 在采样后执行 Backbone forward；是否使用 `no_grad` 由显式的表征
  梯度模式决定。
- WM target 只 detach 右移后的 next-state view，不能提前 detach 整段 state。
- tune mode 只决定哪些 Backbone 参数可训练；`actor.enabled` 只决定是否计算 PPO；
  表征 loss 是否进入 Backbone 使用独立配置。
- Qwen adapter 只处理 Qwen 输入、forward、policy replay、tuning 和 artifact，不定义
  SFT2/RL 的 window、return、target 对齐或梯度策略。
