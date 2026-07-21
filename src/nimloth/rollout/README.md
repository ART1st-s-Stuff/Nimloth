# Rollout

`nimloth.rollout` 独立于具体优化阶段，负责 trajectory 数据。Agent 执行、
评估、SFT2 和 RL 统一使用这套 schema。

- `schema.py`：统一 trajectory 记录、序列化和 prompt 重建。
- `validation.py`：落盘和训练前的跨字段完整性校验。
- `from_agent.py`：把 `AgentEpisode` 转成 trajectory，不参与 environment 交互。
- `storage.py`：JSONL 持久化。
- `source.py`：trajectory source 协议和离线 JSONL source。
- `transitions.py`：trajectory 到 transition 的展开和 dataset。
- `batch.py`：backend 与阶段算法共享的 transition batch/builder 契约。

本包中的任何模块都不得导入 `nimloth.training`。

在线调用链为：

```text
EnvironmentSession -> Agent/EpisodeRunner -> AgentEpisode
                                          -> trajectory_from_agent_episode
                                          -> RolloutTrajectory/JSONL
```

prompt 模板、动作空间和行为概率都随 `AgentEpisode` 进入 trajectory。collector
只选择具体 environment、policy 和保存位置，不得复制 prompt 构造逻辑，也不得在
公共适配器中猜测某个环境的 reward/success 语义。

Qwen hidden state 编码属于模型适配，位于
`nimloth.backbone.qwen25vl.rollout`；VAGEN navigation collector 属于
`nimloth.environment.navigation.collector`。二者都不属于本包。
