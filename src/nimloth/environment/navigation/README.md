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
