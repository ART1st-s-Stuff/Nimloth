---
name: on-experiment-end
description: >-
  在Nimloth实验完成、失败、取消或暂停后记录实验。只要观察到上述终止状态就必须使用，即使该次运行由其他session启动。
---

# 实验结束记录

## 触发条件

任何训练、评估、收集、校准、rollout-train、远程长job、Slurm任务或其他昂贵计算一旦完成、失败、被取消或暂停，必须立即执行本skill。

## 必须执行

1. 阅读[启动/生命周期](../../../.trellis/spec/experiments/launch-and-lifecycle.md)、[输出/checkpoint证据](../../../.trellis/spec/experiments/outputs-checkpoints-and-evidence.md)和[任务/进度/memory](../../../.trellis/spec/governance/tasks-progress-and-memory.md)。
2. 更新运行README/metadata，记录状态、调度器/runtime证据、实际命令/config/commit、数据/split/checkpoint/output来源、W&B标识和train/freeze/objective边界。
3. 记录关键指标/异常、失败/取消原因、目标是否达成、有效性限制和下一步建议。
4. 记录最新checkpoint和精确恢复方法；若无法忠实恢复，必须说明原因。
5. 使用该参数设置的最新**有效**结果更新`outputs/experiments/<group>/progress.md`；禁止提升无效重试的结果。
6. 更新当前Trellis任务证据/检查清单；branch级状态发生变化时，再添加一条简短`AI_branch_progress.md`里程碑。禁止创建新的旧式进度文件。
7. 执行`on-progress`的memory评估：只有实际使用过的memory经重新核验且确实有帮助时才upvote；只添加不重复且可复用的经验；禁止运行human-only审批命令。
