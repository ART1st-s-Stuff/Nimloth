# E0013 — 父进程启动 child 前必须清除旧 attempt marker

## 错误

rollout orchestrator 重试时，旧任务留下 `external_env_4gpu/failed`。虽然 child env launcher 启动后会删除该文件，但父进程在 child 获得调度前先读到了旧 marker，导致6张 GPU 全部通过 AI2-THOR smoke 后仍立即错误退出。

## 正确做法

父进程必须在 spawn child **之前**清除 attempt-scoped `ready` / `failed`，并重置 URL/host control files。不能依赖异步 child 稍后清理；否则存在确定的启动竞态。历史 rollout shard 不在该 control 目录中，不得随 marker 一起删除。
