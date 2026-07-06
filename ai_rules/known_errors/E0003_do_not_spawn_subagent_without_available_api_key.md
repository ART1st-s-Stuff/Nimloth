# E0003: Do not spawn a subagent when its model/API key is unavailable

## 错误

在需要继续监控 retry15 rollout 时，开了一个 subagent，但该 subagent 环境没有可用 API Key，导致 subagent stall，不能可靠完成任务。

## 原因

没有先确认当前任务是否真的需要 subagent，以及 subagent 所需模型/API Key 是否可用。

## 正确做法

- 对简单监控/检查任务，优先由当前 agent 直接执行，不要默认开 subagent。
- 只有在明确需要并行、且确认 subagent 运行环境可用时，才启动 subagent。
- 如果误开了不可用 subagent，应立即中断，并由当前 agent接管任务。

## 触发场景

用户指出：“你刚才开了一个没有API Key的subagent。”
