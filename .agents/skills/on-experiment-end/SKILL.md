---
name: on-experiment-end
description: >-
  Runs the on-experiment-end event hook after an experiment stops. Use when any
  training, evaluation, collection, calibration, rollout-train, remote long job,
  or Slurm task completes, fails, is cancelled, or is paused—even if started by
  another session.
---

# On Experiment End

## 触发条件

以下实验性任务**结束、失败、取消或暂停后**立即启用本 skill（无论是否由当前会话启动）：

- 训练、评估、采集、校准
- rollout-train
- 远程长任务、Slurm 任务
- 其他需要 GPU 或长时间运行的计算任务

## 必须执行

1. 完整阅读 `ai_rules/events/on_experiment_end.md`。
2. 逐条执行其中列出的步骤，不得跳过。
3. 若 event 文件要求更新实验文档、进度文件或评估 memory，必须在**当前对话**中完成，不得推迟。
