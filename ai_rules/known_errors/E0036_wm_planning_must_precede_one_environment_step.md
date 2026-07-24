# E0036：WM 多步规划必须发生在一次真实 environment step 之前

## 已确认错误

RL 配置需要多步 WM planning 时，在线 rollout 曾仍直接用 Qwen policy 选择动作
并立刻调用 environment；另一个 `wm/planning.py` 只有测试调用，没有接入 Agent。
该 planner 每层只传一个 state，
还会让 `history_size>1` 的 predictor 在实际搜索中退化为单步上下文。

## 错误原因

没有把“latent 空间模拟候选动作”和“真实 environment 执行动作”定义成两个不同
契约。训练 replay、在线 planning 和 environment 生命周期分散后，测试中的 planner
被误认为真实运行路径。

## 正确做法

- 每个真实 observation 只运行一次 Qwen/StateProjector，得到 planning 根状态。
- WM 可以在 latent 空间模拟多个完整 action sequence，但不得持有 environment 或
  调用 `step()`。
- planner 只返回最终选中 sequence 的首动作；`EpisodeRunner` 是唯一调用
  `EnvironmentSession.step()` 的位置，每轮只执行一次。
- SFT2 只从 VAGEN trajectory 学习初始 WM/value 参数，不调用 planner 或
  environment runner。
- SFT2 与 RL 的 `history_size` 都表示 LeWM 因果上下文长度，checkpoint 必须
  严格匹配。RL 要自回归预测多个未来 step 时，应使用独立的
  planning horizon，不得混入 `history_size` 语义。
- beam expansion 必须保留或重放完整候选 history，不能把多步 predictor 当单步模块。
- 如果 planner 决定了真实动作，rollout 必须记录 planner 的真实 behavior
  distribution；在 planner replay 实现前不得复用 Qwen PPO probability。
