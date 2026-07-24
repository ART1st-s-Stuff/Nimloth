# Agent

`nimloth.agent` 同时定义神经网络 Agent 的公共契约，以及与 environment 交互的
episode runtime。两者名称和职责明确分开：

| 模块 | 职责 |
|------|------|
| `model.py` | `Agent(nn.Module)`：组合 `Backbone` 与 `WorldModel` |
| `transcript.py` | 按时间保存 observation、图片、动作和真实 assistant response |
| `template.py`、`templates/` | prompt 契约与具体模板 |
| `policy.py` | 行为分布、policy 与 PPO replay 公共协议 |
| `planning.py` | Qwen 单次状态编码、WM 多步搜索与首动作决策 |
| `runtime.py` | `AgentRuntime`：维护一个 episode 的 transcript |
| `runner.py` | `EpisodeRunner`：驱动 `EnvironmentSession` |

## 模型契约

`Agent.encode_state(BackboneBatch)` 是 Qwen→StateProjector 的公共状态入口；
planning 每个真实 observation 只调用它一次。`Agent.forward(..., action_indices)`
在此基础上继续计算 WM prediction 与 action values，供 SFT2 离线训练使用。
SFT2 不调用 planning policy 或 episode runner。processor、
cache、EMA、optimizer、checkpoint 与 environment 状态均不进入 `Agent.state_dict()`。
rollout transition 的 batch 契约属于 `nimloth.rollout`，不属于 Agent 模型接口。
SFT2 的 target-state 梯度与 EMA 策略属于 `training.sft2.runtime`。
PPO replay 接收 `PolicyReplayInput(AgentPrompt, action, sampling config, token
trace)`。`PolicyTokenTrace` 区分 reasoning、action 与 injected token；只有
`loss_mask=True` 的真实 behavior token 能进入 PPO，因此具体 Backbone 不需要依赖
rollout transition 类型。

`build_response_policy_prompt()` 属于 behavior 生成入口，只预填 `<think>` 并保留
实际采样 CoT 供 PPO replay。旧 `build_state_prompt()` 无法为未执行 observation
生成真实 CoT，当前明确拒绝调用；RL 必须补齐 current/terminal CoT 持久化后才能恢复
state replay 与 `PlanningPolicy`。禁止用固定 thought 代替该 TODO。

## Planning 契约

`PlanningPolicy` 的 WM 搜索器仍保留，但在线 policy 当前 fail-fast：它尚未在每个
observation 上先生成并持久化真实 CoT。完成该边界后，预期每个真实 environment step
只编码一次当前 prompt，随后
`WorldModelPlanner` 在 latent 空间模拟候选 action sequence，并只把选中序列的首
动作交给 `EpisodeRunner`。planner 不持有 environment；唯一真实动作执行点仍是
`EpisodeRunner` 中的 `session.step()`。

当前 WM 没有 reward/done head，因此 beam search 明确使用叶节点最大 action-value
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
