# AI_issues.md — 需要人类确认的问题

## 2026-06-10 memory skill/CLI 待确认

1. **审批 pending memory**：当前 `.memory/memories.jsonl` 中有 `M0001`，记录 memory skill/CLI 的存在。请在需要时运行 `./skill human memory-approve` 审批或驳回。
2. **Task skill/CLI**：是否继续按类似方式实现 `task` skill/CLI？
3. **旧记忆系统迁移**：是否逐步废弃 `AI_branch_progress.md`、`AI_issues.md`、`ai_tasks/`，转为由 skill/CLI 生成或替代？

## 人类回答
1. 不需要进入记忆，已经在skill里了。
2. 是，也需要针对issue写类似的CLI
3. 是。