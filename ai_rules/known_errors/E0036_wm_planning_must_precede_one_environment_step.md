# E0036：不要在已选中的 WM segment 内重新运行 Qwen

## 已确认错误

早期实现一度让planner每个environment step都重新运行Qwen，只执行多步计划的首动作。
这会把`planning.horizon`降格成每步重复搜索，并让history里的每个state都变成Qwen实际
输出，违背了当前明确要求：一次Qwen锚点后应实际模拟并消费整段WM预测state/action。

## 错误原因

把“planner评估一个未来序列”和“policy真正执行一个已选序列”混在了一起；同时擅自
假定每个history state都必须来自Qwen，而没有确认预测state是否应被保留到后续上下文。

## 正确做法

- Qwen只在segment锚点运行。锚点真实state校正上一段的预测终点。
- planner返回完整选中sequence及对应预测state；policy逐步消费它们，直到整段执行完、
  environment提前结束，或terminal observation到来。
- segment内部不运行Qwen，不生成或填充CoT。`EpisodeRunner`仍是唯一调用
  `EnvironmentSession.step()`的位置，每次只执行当前已选动作。
- SFT2 只从 VAGEN trajectory 学习初始 WM/value 参数，不调用 planner 或
  environment runner。
- SFT2 与 RL 的 `history_size` 都表示 LeWM 因果上下文长度，checkpoint 必须
  严格匹配。RL 要自回归预测多个未来 step 时，应使用独立的
  planning horizon，不得混入 `history_size` 语义。
- search expansion必须保留或重放完整候选history，不能把多步predictor当单步模块。
- 如果planner决定了真实动作，rollout必须记录WM actor的真实behavior distribution；
  Qwen只能做distillation。未来PPO必须先把行为所有权切换到Qwen，并保证Qwen采样动作
  就是environment实际执行动作。
