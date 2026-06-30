# 02 Memory and Progress

## 进度文件

- `AI_branch_progress.md`：当前阶段/分支的权威进展、计划、失效记忆和待确认事项。
- `ai_tasks/ai_progress/`：长任务或系统性任务的实时进度记录。
- `AI_issues.md`：需要人类确认、批准或决策的问题。
- memory SKILL：长期记忆入口。使用 `.agents/skills/memory/SKILL.md` 中定义的协议，通过 `./skill memory ...` 创建、搜索、检查、修正、upvote 记忆；repo 记忆写入 `.memory/memories.jsonl`，本地/环境相关记忆写入 `.local/memory/memories.jsonl`；都禁止手动编辑。Memory 只保存从项目工作中提取的短小有效经验，不复制规则文档、进度文件、实验说明或源码文档。

## 何时更新

必须更新进度/记忆的情况：

- 完成一个阶段性任务；
- 改变项目规则、计划或重要约定；
- 发现旧记忆/旧设计失效；
- 提出需要人类决策的问题；
- 人类明确要求“更新进度/记忆”。

任务过程中，AI 可以随时通过 memory SKILL 查询或更新记忆。创建或依赖记忆时必须遵守 `.agents/skills/memory/SKILL.md`：AI 创建的 memory 默认为 `pending-human-verification`；依赖既有 memory 前必须 `get` 并检查 evidence 指向的文件段；只有确认该 memory 对当前任务有帮助后才可以 upvote。repo 记忆默认面向环境无关经验；本地记忆默认面向当前服务器/工作区相关经验。

Memory 的使用规范：

- Memory 是为 Nimloth 项目提取的短的有效经验，用来帮助未来 Agent 少走弯路。
- Memory 应记录稳定的经验、约束、决策或查找提示，而不是任务过程、TODO、实验流水账或长总结。
- 如果信息已经清楚存在于 `AGENTS.md`、`ai_rules/`、实验 README、代码注释或进度文件中，不要创建只是重复原文的 memory。
- 进度文件记录“做了什么/当前状态/待确认事项”；memory 记录“未来复用时真正有价值的经验”。
- 创建 memory 前先问：这条是否能用一句话节省未来 Agent 的探索成本？如果答案是否定的，不要创建。

## 记录原则

- 简洁、事实性、可检索。
- 区分事实、假设、决策、待确认问题。
- 不记录用户个人隐私。
- 不记录无意义聊天细节。
- 不把过时结论保留为当前事实；如果过时，应标记失效或修正。

## 长任务规范

长任务开始时，在 `ai_tasks/ai_progress/` 建立进度文件，记录：

- 任务目标；
- 当前计划；
- 已完成步骤；
- 文件修改；
- 验证命令和结果；
- 待确认问题。

任务阶段结束时，同步更新 `AI_branch_progress.md`；并把进度文件移动到`ai_tasks/ai_progress/archives/<date>`

## 事件规则

以下事件有额外规则，触发时必须阅读并执行：

- 取得阶段性进展：`ai_rules/events/on_progress.md`
- 实验开始前：`ai_rules/events/on_experiment_start.md`
- 实验结束后：`ai_rules/events/on_experiment_end.md`
