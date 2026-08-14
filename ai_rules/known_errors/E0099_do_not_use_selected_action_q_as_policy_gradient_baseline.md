# E0099：禁止把 selected-action Q 直接当作 policy-gradient baseline

## 已确认错误

在解释首版 joint PPO actor advantage 选项时，agent 曾提出
`A = discounted_return - frozen_Q(s, executed_action)`。这是 action-dependent baseline；若
critic 接近真实 `Q(s,a)`，该残差对给定动作的期望接近 0，并不等同于 PPO 所需的
`Q(s,a)-V(s)` 或 `return-V(s)` advantage。直接这样使用会改变并可能破坏 policy-gradient
目标。

## 正确做法

- PPO baseline 必须是 rollout-time、stop-gradient 的 state value，例如
  `V_frozen(s)=Σ_a π_guided,behavior(a|s) Q_frozen(s,a)`。
- 使用真实 discounted return 时，首版可定义
  `A_t = G_t - V_frozen(s_t)`；若使用 GAE，则也必须从同一 rollout-time frozen scalar
  value 序列构造，禁止混入更新后的 current critic。
- `Q_frozen(s, executed_action)`可以作为 critic 对已执行动作的预测、校准指标或
  `Q-V` advantage 的 Q 项，但不能单独作为与动作无关的 baseline。
- 任何 advantage 定义都必须有独立公式测试，验证 current critic 更新不会改变历史
  rollout advantage。
