# E0037：SFT2 与 RL 必须共享 LeWM history_size 语义

## 已确认错误

重构过程中曾把 SFT2 固定为 `history_size=1`，同时让 RL 使用可变
`history_size`，并试图通过 checkpoint 位置参数扩展或其他兼容逻辑跨越二者。
这会让 SFT2 warm-start 的 predictor 从未学过 RL 实际使用的因果上下文。

## 错误原因

把 LeWM 的上下文长度和 RL 想向未来模拟的步数混成了一个参数。前者决定训练
窗口、predictor 结构和 checkpoint 形状；后者只决定一次真实环境动作之前要在
latent 空间自回归多少步。

## 正确做法

- `history_size=H` 在 SFT2/RL 中都表示 predictor 可见的最大因果上下文长度。
- SFT2 的一个样本使用不超过 H 个连续真实 state/action 作为 context；当前实现只对
  窗口末端 current step 计算 CE、下一状态和 ValueHead 主监督，较老 state 来自它们
  先前作为 current step 时建立的 detached history cache。
- RL 的一个采样窗口包含 H 个连续 transition 和 H+1 个真实状态，并在同一次更新中
  对 H 个因果位置计算下一状态、action value 和可选 PPO 目标。
- warm-start checkpoint 的 `history_size` 必须与当前配置严格相等，不做形状扩展。
- SFT2 只消费 VAGEN 离线轨迹，不调用 planner 或 environment。
- RL 的未来预测长度使用独立的 `agent.planning.horizon=P`；planner 从最近真实
  history 出发，自回归模拟 P 步，最终只执行所选序列的第一个动作。
