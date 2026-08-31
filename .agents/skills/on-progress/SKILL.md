---
name: on-progress
description: >-
  在Trellis中记录Nimloth实质进展并评估curated memory。在完成可验证子任务、关键修复、重要设计决策、实验阶段、项目规则变更或推翻既有结论后使用。
---

# 进度记录

## 触发条件

出现上述任一实质里程碑后，必须暂停其他工作并立即执行本skill。禁止推迟到后续对话。

## 必须执行

1. 阅读[任务、进度与memory](../../../.trellis/spec/governance/tasks-progress-and-memory.md)以及当前任务计划。
2. Pi的`trellis_work_item`可用且当前里程碑属于计划项时，先把assignment更新为准确状态并只添加typed ref/短summary证据；完成项必须先更新`implement.md`checkbox，再release。禁止用runtime直接声明done或猜测未选择的item。
3. 在当前Trellis任务中更新当前细节、证据、未决问题和执行检查清单。
4. 只有branch级状态发生变化时，才添加一条简短`AI_branch_progress.md`里程碑。禁止新建`ai_tasks/ai_progress/`记录，也禁止把进行中的工作加入`AI_issues.md`。
5. 评估该里程碑是否产生了spec/文档尚未清晰记载的、紧凑且可复用的经验。只能通过`memory` skill创建memory；禁止直接编辑JSONL。
6. 对每条实际使用的memory，再次执行`get`、重新阅读证据；只有其仍然正确且确实有帮助时才upvote。必须通过skill纠正错误；冲突无法解决时，必须询问人类。
7. 如果新增或修订了待核验memory，提醒人类审批。禁止运行`./skill human ...`。
