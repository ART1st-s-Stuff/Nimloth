---
name: on-experiment-start
description: >-
  Runs the on-experiment-start event hook before any expensive or long-running
  experiment. Use before training, evaluation, data collection, calibration,
  rollout-train, remote long jobs, Slurm tasks, or other GPU-intensive work.
---

# On Experiment Start

## 触发条件

在启动以下**具体任务之前**立即暂停当前工作，启用本 skill：

- 训练、评估、采集、校准
- rollout-train
- 远程长任务、Slurm 任务
- 其他需要 GPU 或长时间运行的计算任务

## 必须执行

1. 完整阅读 `ai_rules/events/on_experiment_start.md`。
2. 逐条执行其中列出的步骤，不得跳过。
3. 若任一关键项不清楚，停止并询问人类；不得用近似实验替代人类指定实验。
