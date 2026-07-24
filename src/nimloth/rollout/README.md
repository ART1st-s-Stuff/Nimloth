# Rollout

`nimloth.rollout` 独立于具体优化阶段，负责 trajectory 数据。Agent 执行、
评估、SFT2 和 RL 统一使用这套 schema。

- `schema.py`：统一 trajectory 记录、序列化和 prompt 重建。
- `validation.py`：落盘和训练前的跨字段完整性校验。
- `from_agent.py`：把 `AgentEpisode` 转成 trajectory，不参与 environment 交互。
- `storage.py`：JSONL 持久化。
- `source.py`：trajectory source 协议和离线 JSONL source。
- `transitions.py`：trajectory 到 transition 的展开和 dataset。
- `windows.py`：在原始 trajectory 上计数并采样连续 `H` 步窗口。
- `batch.py`：阶段 assembler 与算法之间的 transition batch 契约。

本包中的任何模块都不得导入 `nimloth.training`。

在线调用链为：

```text
EnvironmentSession -> Agent/EpisodeRunner -> AgentEpisode
                                          -> trajectory_from_agent_episode
                                          -> RolloutTrajectory/JSONL
```

prompt 模板、动作空间、真实 assistant response、逐 token behavior log-prob 与
loss mask 都随 `AgentEpisode` 进入 trajectory。collector
只选择具体 environment、policy 和保存位置，不得复制 prompt 构造逻辑，也不得在
公共适配器中猜测某个环境的 reward/success 语义。

窗口保留原始 Agent prompt 与 behavior provenance，不提前固化为 detached
Backbone hidden。VAGEN navigation collector 属于
`nimloth.environment.navigation.collector`，不属于本包。

turn-credit trajectory 同时保留两种可重建输入：behavior replay prompt 从
`<think>` 开始；state prompt 使用该步真实 CoT 的 latent query prefix。窗口模块只
负责保持顺序，不计算 advantage。
