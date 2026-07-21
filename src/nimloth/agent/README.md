# Agent

`nimloth.agent` 同时定义神经网络 Agent 的公共契约，以及与 environment 交互的
episode runtime。两者名称和职责明确分开：

| 模块 | 职责 |
|------|------|
| `model.py` | `Agent(nn.Module)`：组合 `Backbone` 与 `WorldModel` |
| `target.py` | target backbone stop-gradient 与 target StateProjector 梯度边界 |
| `transcript.py` | 按时间保存 observation、图片和动作 |
| `template.py`、`templates/` | prompt 契约与具体模板 |
| `policy.py` | 行为 policy 与 PPO replay 协议 |
| `runtime.py` | `AgentRuntime`：维护一个 episode 的 transcript |
| `runner.py` | `EpisodeRunner`：驱动 `EnvironmentSession` |

## 模型契约

`Agent.forward(BackboneBatch, action_indices)` 是 SFT2 当前状态的真实模型入口，
返回 hidden、state、WM prediction、action values 和可选 LM loss。processor、
cache、EMA、optimizer、checkpoint 与 environment 状态均不进入 `Agent.state_dict()`。
rollout transition 的 batch 契约属于 `nimloth.rollout`，不属于 Agent 模型接口。

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
