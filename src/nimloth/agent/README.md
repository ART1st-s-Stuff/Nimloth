# Agent

`nimloth.agent` 同时定义神经网络 Agent 的公共契约，以及与 environment 交互的
episode runtime。两者名称和职责明确分开：

| 模块 | 职责 |
|------|------|
| `model.py` | `Agent(nn.Module)`：组合 `Backbone` 与 `WorldModel` |
| `transcript.py` | 按时间保存 observation、图片、动作和真实 assistant response |
| `template.py`、`templates/` | prompt 契约与具体模板 |
| `policy.py` | 行为分布、planner search trace 与直接 Qwen PPO replay 协议 |
| `planning.py` | 每步真实 Qwen state、WM 多步搜索与首动作执行 |
| `runtime.py` | `AgentRuntime`：维护一个 episode 的 transcript |
| `runner.py` | `EpisodeRunner`：驱动 `EnvironmentSession` |

## 模型契约

`Agent.encode_state(BackboneBatch)` 是 Qwen→StateProjector 的公共状态入口；
planning在每个真实environment step和terminal observation调用它。`Agent.forward(..., action_indices)`
在此基础上依次计算 WM predicted next state 及其 action values，供 SFT2 离线训练使用。
`Agent.forward_action_rollout()`从一个真实state出发，严格使用调用方提供的
recorded action sequence递归产生`T`个预测state及各自action values。
SFT2 不调用 planning policy 或 episode runner。processor、
cache、EMA、optimizer、checkpoint 与 environment 状态均不进入 `Agent.state_dict()`。
rollout transition 的 batch 契约属于 `nimloth.rollout`，不属于 Agent 模型接口。
SFT2 的固定下一状态监督值与 Backbone EMA 策略属于 `training.sft2.runtime`。
Policy replay 接收 `PolicyReplayInput(AgentPrompt, action, sampling config,
token trace)`。`PolicyTokenTrace` 区分 reasoning、action 与 injected token；
planner路线不进入policy replay；只有直接由Qwen拥有并执行环境动作的路线可以做PPO。

`build_response_policy_prompt()` 属于 behavior 生成入口，只预填 `<think>`。每个
Qwen 锚点和 terminal observation 都保存模型实际生成的 CoT 与同一次 forward 的
latent hidden。禁止用固定 thought 补齐没有运行 Qwen 的中间 state。

## Planning 契约

`PlanningPolicy` 在每个environment step都让Qwen对当前真实observation生成CoT和
state，再从最多`history_size`个真实state/action出发搜索长度为
`planning.horizon`的候选动作序列。greedy/exhaustive/beam按候选叶值选择；MCTS
按UCT visit count和backed-up leaf mean选择根动作。所有模式只把选中的首动作交给
environment。候选尾部只用于搜索，永远不直接执行；下一步拿到真实
observation后重新运行Qwen并重新规划。

环境动作由WM+ValueHead actor决定。planner保存确定性的行为分布和search trace，
但不保存Qwen action prior，不对Qwen action logits做蒸馏或PPO。Qwen只通过
WM/ValueHead的可微state路径接收梯度。

当前 WM 没有 reward/done head，因此search只使用叶节点action-value启发式，
不逐步累加MC-return prediction。greedy/exhaustive/beam保留leaf max；H=1/K-step
SFT2 MCTS使用训练时实际受监督的
`Q_tilde(predicted_state_K, final_simulated_action)`作为leaf evaluation。MCTS trace
额外保存candidate/root visits、backed-up mean、simulation数和UCT exploration常数。

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
