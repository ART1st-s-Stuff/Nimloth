---
name: on-progress
description: >-
  Runs the on-progress event hook after substantive task progress. Use when
  finishing a verifiable subtask, fixing a critical bug, confirming an important
  design, completing an experiment phase, changing project rules, or invalidating
  an old conclusion.
---

# On Progress

## 触发条件

完成以下**具体动作**后立即暂停当前工作，启用本 skill（不是抽象判断「有没有进展」）：

- 完成一个可验证功能或子任务
- 修复一个关键 bug
- 确认一个重要设计决策
- 完成一次实验阶段
- 改变项目规则
- 发现旧结论失效

## 必须执行

1. 完整阅读 `ai_rules/events/on_progress.md`。
2. 逐条执行其中列出的步骤，不得跳过。
3. 若 event 文件要求更新进度文件或评估 memory，必须在**当前对话**中完成，不得推迟。
