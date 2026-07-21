# Rollout

`nimloth.rollout` 独立于具体优化阶段，负责 trajectory 数据。Agent 执行、
评估、SFT2 和 RL 统一使用这套 schema。

- `schema.py`：统一 trajectory 记录与校验。
- `storage.py`：JSONL 持久化。
- `source.py`：trajectory source 协议和离线 JSONL source。
- `collector.py`：在线环境收集。
- `transitions.py`：trajectory 到 transition 的展开和 dataset。
- `encoding.py`：用共享 Qwen policy state 编码 rollout transition。

本包中的任何模块都不得导入 `nimloth.training`。
