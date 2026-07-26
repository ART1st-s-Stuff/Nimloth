# Agent

`nimloth.agent` 同时定义神经网络 Agent 的公共契约，以及与 environment 交互的
episode runtime。两者名称和职责明确分开：

| 模块 | 职责 |
|------|------|
| `model.py` | `Agent(nn.Module)`：组合 `Backbone` 与 `WorldModel` |
| `transcript.py` | 按时间保存 observation、图片、动作和真实 assistant response |
| `template.py`、`templates/` | prompt 契约与具体模板 |
| `policy.py` | 行为分布、action objective 与 policy replay 公共协议 |
| `planning.py` | Qwen 锚点校正、WM 多步搜索与整段动作执行 |
| `runtime.py` | `AgentRuntime`：维护一个 episode 的 transcript |
| `runner.py` | `EpisodeRunner`：驱动 `EnvironmentSession` |

## 模型契约

`Agent.encode_state(BackboneBatch)` 是 Qwen→StateProjector 的公共状态入口；
planning只在segment锚点和terminal observation调用它。`Agent.forward(..., action_indices)`
在此基础上继续计算 WM prediction 与 action values，供 SFT2 离线训练使用。
SFT2 不调用 planning policy 或 episode runner。processor、
cache、EMA、optimizer、checkpoint 与 environment 状态均不进入 `Agent.state_dict()`。
rollout transition 的 batch 契约属于 `nimloth.rollout`，不属于 Agent 模型接口。
SFT2 的 target-state 梯度与 EMA 策略属于 `training.sft2.runtime`。
Policy replay 接收 `PolicyReplayInput(AgentPrompt, action, sampling config,
token trace)`。`PolicyTokenTrace` 区分 reasoning、action 与 injected token；
`ActionTrainingTrace` 另外声明环境动作由 Qwen 还是 WM actor 产生，以及当前目标是
distillation 还是 PPO。PPO 只有在 Qwen 实际采样并执行该动作时才合法。

`build_response_policy_prompt()` 属于 behavior 生成入口，只预填 `<think>`。每个
Qwen 锚点和 terminal observation 都保存模型实际生成的 CoT 与同一次 forward 的
latent hidden。禁止用固定 thought 补齐没有运行 Qwen 的中间 state。

## Planning 契约

`PlanningPolicy` 把 episode 分成若干 WM segment。segment 开始时，Qwen 对当前真实
observation 生成 CoT 和 state；该真实 state 会校正上一段保存的预测终点。planner
随后从最多 `history_size` 个真实/预测混合 state 出发，产生长度为
`planning.horizon` 的动作和预测 state。`EpisodeRunner` 逐步执行整段动作；段内不再次
运行 Qwen，也不会伪造 CoT。若环境提前结束，尚未执行的计划尾部直接丢弃。

当前环境动作由 greedy WM/ValueHead actor 决定。Qwen action head 只在 segment 锚点上
模仿该确定性动作；planner action 不进入 PPO。接口保留了未来的 action PPO provenance，
但在切换为“Qwen 采样的动作就是环境实际执行动作”之前会明确拒绝启用。

当前 WM 没有 reward/done head，因此search明确使用叶节点最大 action-value
作为启发式 score，不逐步累加 Q-value，也不声称该 score 是预测 return。

## Episode 契约

```python
from nimloth.agent import AgentRuntime, EpisodeRunner

runtime = AgentRuntime(
    policy=policy,
    action_space=session.action_space,
    prompt_template=prompt_template,
)
episode = EpisodeRunner(runtime).run(session, seed=42, max_steps=20)
```

environment 提供 system prompt、observation、动作空间、reward 和 success；
`moveahead` 等环境语义不会硬编码在 Agent prompt 中。
