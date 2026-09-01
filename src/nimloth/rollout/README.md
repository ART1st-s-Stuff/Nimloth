# Rollout

`nimloth.rollout` 独立于具体优化阶段，负责 trajectory 数据。Agent 执行、
评估、SFT2 和 RL 统一使用这套 schema。

- `schema.py`：统一 trajectory 记录、序列化和 prompt 重建。
- `record_format.py`：当前持久化版本、reward来源和最小结构契约。
- `migration.py`：把未版本化JSONL离线转换为当前格式并写SHA256 manifest。
- `validation.py`：落盘和训练前的跨字段完整性校验。
- `from_agent.py`：把 `AgentEpisode` 转成 trajectory，不参与 environment 交互。
- `storage.py`：JSONL 持久化。
- `source.py`：trajectory source 协议和离线 JSONL source。
- `transitions.py`：trajectory 到 transition 的展开和 dataset。
- `windows.py`：在原始 trajectory 上计数并采样连续 `H` 步窗口。
- `batch.py`：阶段 assembler 与算法之间的 transition batch 契约。

本包中的任何模块都不得导入 `nimloth.training`。

训练读取器只接受`record_format: nimloth_trajectory_v1`。旧的`messages`交替列表、
`nav_instruction`和`prompt_version`/`latent_token_count`只在迁移命令中解析：

```bash
python -m nimloth.rollout.migration \
  --source old.jsonl \
  --output migrated.jsonl \
  --missing-action-space-id navigation \
  --missing-action-space-version 1 \
  --missing-reward-provenance trajectory_terminal_reward
```

这些`--missing-*`参数是调用者对源数据语义的声明，会写入manifest。迁移器只拆分
源记录真实存在的system/user/assistant文本，不生成CoT、token trace、Qwen hidden或
WM state。旧planner记录缺少每个真实step的Qwen state和独立search trace，无法忠实
迁移到当前receding-horizon contract；必须重新采集。

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
Backbone hidden；planner trajectory额外携带每个step的rollout Qwen latent hidden和
投影state用于审计及固定next-state target。训练value路径仍从真实CoT重建完整prefix并
重新执行Qwen，不把这些hidden或vLLM KV cache接回旧graph。VAGEN navigation collector 属于
`nimloth.environment.navigation.collector`，不属于本包。

turn/token-credit trajectory 的 behavior replay prompt 从 `<think>` 开始并保留实际采样
CoT。只有版本为`vagen_step60_dual_view_conversion_v2`且通过source/audit/converted hash
校验的离线source-policy记录，才能把`policy_credit_assignment=none`与空
`action_log_probs`/`policy_messages`记录为unavailable；其`source_identity`会经标准
`RolloutTrajectory` roundtrip保留，source audit绑定原始identity/chat/action/reward/image/
checkpoint/runtime证据。step60 reconstruction 还必须使用
`source_audit.contract_version=vagen_step60_reconstruction_audit_v2`，分开记录不可读source
commit与实际reconstruction base/HEAD/tree/diff/evidence，并只接受逐步reward；旧v1 audit不得静默解释为v2。
其他记录不能借该字段绕过behavior prompt或概率校验，也不能用
one-hot behavior分布冒充缺失概率。terminal observation额外生成并持久化真实CoT，不执行其draft action。planner
trajectory若缺少任一`T + 1` state hidden会在读取/训练前失败，不能退回固定thought。
planner训练逐真实transition重算一次完整prefix并立即backward，不同时保留多个step的
Qwen graph。窗口模块只负责保持顺序，不计算
advantage。fresh逐步reward用于完整episode return；真正terminal从0 bootstrap，
时间上限truncation必须由训练配置显式选择bootstrap语义。
