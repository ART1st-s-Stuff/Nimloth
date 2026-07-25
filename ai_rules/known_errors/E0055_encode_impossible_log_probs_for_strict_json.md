# E0055：严格JSON必须显式编码不可能动作的log-prob

## 已确认错误

greedy planner的teacher/behavior分布会用`-inf`表示未选动作。ID102完成四条真实rollout后，
`planner_policy_traces`把这些值直接交给`json.dumps(..., allow_nan=False)`，导致保存失败并留下
零字节`trajectories.jsonl`。

## 正确做法

- 所有动作分布字段共享同一严格JSON契约：写盘时`-inf -> null`，读取时`null -> -inf`。
- 对top-level behavior和nested planner的Qwen/teacher/behavior分布都执行该转换；不得改用
  非标准`Infinity`字面量。
- 保存前先校验并完成全部序列化，再原子替换目标JSONL，禁止以空文件冒充有效trajectory。
- 对greedy planner的严格JSON round-trip和失败写入不覆盖旧文件保留回归测试。
