# Agent

`nimloth.agent` 是训练阶段无关的 Agent 核心。在线采集、评估、SFT2 数据展开和
RL policy 重放都使用这里定义的 transcript 与 prompt 契约。

| 模块 | 职责 |
|------|------|
| `transcript.py` | 按时间保存 observation、图片和动作 index |
| `template.py` | `AgentPrompt`、可持久化模板描述和模板协议 |
| `templates/` | 具体 prompt 模板；当前实现是 Nimloth latent-action 模板 |
| `registry.py` | 按模板标识和版本重建模板 |
| `policy.py` | 模型无关 policy 协议、决策和行为概率校验 |
| `runtime.py` | 维护 transcript 并执行 policy 的 `Agent` |
| `runner.py` | 驱动 `EnvironmentSession` 的 `EpisodeRunner` |
| `serialization.py` | 新旧 trajectory 的模板字段迁移 |

## 职责边界

- environment 提供 `system_prompt`、observation、动作空间、reward 和 success；
  `moveahead` 一类环境指令不写在 Agent 模板中。
- Agent 模板只把 transcript 编码成 policy prompt 或监督 prompt。
- policy 只接收结构化 `AgentPrompt`；Qwen 的实现位于
  `nimloth.backbone.qwen25vl.policy`。
- runner 只负责一次 episode 的交互和资源关闭，不负责 JSONL。
- rollout 通过 `AgentEpisode` 持久化运行结果，不重新拼 prompt。

## 调用方式

```python
from nimloth.agent import Agent, EpisodeRunner, NimlothPromptTemplate

agent = Agent(
    policy=policy,
    action_space=session.action_space,
    prompt_template=NimlothPromptTemplate(
        latent_token_count=1,
        action_count=len(session.action_space),
    ),
)
episode = EpisodeRunner(agent).run(session, seed=42, max_steps=20)
```

`EpisodeRunner` 会把 session 返回的 `system_prompt` 原样交给 Agent。每次 policy
调用都保留实际使用的未绑定 prompt；图片只在模型调用前按 `<image>` 顺序绑定。
完成的 episode 比动作多一个 observation，用于构造最后一个 `s_{t+1}`。

旧 `WMAgent` 原型已删除，因为仓库内没有训练、评估、rollout 或 CLI 调用它。
新的 planner Agent 只有在存在真实入口，并明确 transcript/policy 契约后才应加入。
