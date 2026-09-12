# Navigation environment

本包定义 navigation 动作空间，并把新版 VAGEN 的 async 单-session client
适配为 Nimloth `EnvironmentSession`。轨迹记录由 `nimloth.rollout` 负责。

## 模块

- `action_space.py`：navigation 动作 key、别名和稳定 index。
- `vagen.py`：observation 解码、K16 环境配置、reward/success 与 session 生命周期。
- `vagen_batch.py`：为现有同步 collector 批量管理多个 async VAGEN sessions。
- `collector.py`：组合 `AgentRuntime`、navigation session 与公共 policy 采集 trajectory。

## 调用关系

`EpisodeRunner` 调用 `VAGENNavigationSession`；collector 只接收公共
`AgentPolicy`，负责批量选择任务、保存图片和构造统一 trajectory，不依赖具体
backbone。训练代码不直接调用 VAGEN client。

`source_client.py` 复用原 VAGEN BatchEnvironmentServer HTTP 传输；`early_evaluation.py`
拥有 VAGEN/B/Query 的真实 direct episode 生命周期、无效响应 no-op 和显式 success 读取。
它们只用于早期阶段统一评估；上文新版 async client 的 planner/collector 路径不变。

早期评估通过 `--episode-concurrency INT` 控制并行 episode 数（默认 1）。
同一轮的环境请求和模型请求均批量提交；独立保存每条轨迹的历史、seed 和原始输出。
环境启动时设置 `ENV_MAX_WORKERS` 至少等于 episode concurrency，并按可用渲染内存选择并行数。
例如现有统一评估命令追加 `--episode-concurrency 4`，环境入口使用 `ENV_MAX_WORKERS=4`。
Stage 1 固定 32 条门禁也按相同大小分批；不改变样本或门禁阈值。
并行数写入运行合同，恢复时不能改变；旧串行合同仅允许并行数 1 恢复。

`run_direct_episodes(..., identities=...)` 可接收配置集合的严格子集，用于训练内多 rank
互不重叠的评估；每个调用拥有独立输出和 UUID session。`phase_timings.jsonl`
记录每批环境操作与生成耗时，默认完整集合调用保持原语义。
