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

prompt 模板、动作空间、真实 assistant response、逐 token behavior log-prob、
loss mask、逐步reward与`terminated`/`truncated`都随 `AgentEpisode` 进入trajectory。
planner rollout还保存每个动作state和terminal state同一次Qwen forward产生的
pre-StateProjector latent hidden。collector只选择具体 environment、policy 和保存位置，
不得复制 prompt 构造逻辑，也不得在公共适配器中猜测某个环境的 reward/success 语义。

窗口始终保留原始 Agent prompt 与 behavior provenance。普通trajectory不提前固化
Backbone hidden；planner trajectory可额外携带rollout captured Qwen latent hidden，
但只能在训练显式关闭representation-to-Backbone梯度时用于state路径。它是Qwen输出、
不是KV cache或投影后的WM state。VAGEN navigation collector 属于
`nimloth.environment.navigation.collector`，不属于本包。

turn/token-credit trajectory 的 behavior replay prompt 从 `<think>` 开始并保留实际采样
CoT。terminal observation额外生成并持久化真实CoT，不执行其draft action。planner
trajectory若缺少任一`T + 1` state hidden会在读取/训练前失败，不能退回固定thought或
重新把`B * (H + 1)`完整prefix一次送进Qwen。窗口模块只负责保持顺序，不计算
advantage。fresh逐步reward用于完整episode return；真正terminal从0 bootstrap，
时间上限truncation必须由训练配置显式选择bootstrap语义。
