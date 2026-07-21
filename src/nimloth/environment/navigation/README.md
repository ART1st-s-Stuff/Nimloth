# Navigation environment

本包定义 navigation 动作空间，并把 VAGEN server 的单个 episode 适配为
Nimloth `EnvironmentSession`。轨迹记录由 `nimloth.rollout` 负责。

## 模块

- `action_space.py`：navigation 动作 key、别名和稳定 index。
- `vagen.py`：observation 解码、环境配置、reward/success 与 session 生命周期。

## 调用关系

`EpisodeRunner` 调用 `VAGENNavigationSession`；online rollout collector 只负责
批量选择任务、保存图片和构造统一 trajectory。训练代码不直接调用 VAGEN client。
