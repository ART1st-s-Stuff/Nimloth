# Event: On Progress

当任务取得阶段性进展时触发。阶段性进展包括：完成一个可验证功能、修复关键问题、确认重要设计、完成一次实验阶段、改变项目规则或发现旧结论失效。

必须执行：

1. 根据 `ai_rules/02_memory_and_progress.md` 更新必要的进度文件。
2. 判断本次进展是否值得形成 durable memory：
   - 只有当本次进展产出了可复用、短小、非重复的项目经验时，才使用 memory SKILL 添加新的 memory。
   - Memory 必须短小、可搜索，并以文件段 evidence 为依据。
   - 不把临时 TODO、流水账、聊天细节、规则原文、实验说明或进度摘要写入 memory。
3. 评估本任务过程中曾经使用过的 memory：
   - 对每条被使用的 memory，重新 `get` 并检查 evidence 文件段是否仍然支持该 memory。
   - 若 memory 仍正确且确实帮助了当前任务，执行 `./skill memory upvote <id>`。
   - 若 memory 错误，使用 `./skill memory set ...` 修正；若无法判断，记录待确认问题并询问人类。
4. 若添加或修改了 pending memory，需要提醒人类通过 `./skill human memory-approve` 审批；AI 不得运行 human-only 命令。
