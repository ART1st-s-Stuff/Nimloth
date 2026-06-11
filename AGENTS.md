# AGENTS.md — Nimloth AI 入口

**所有编程 AI 开始工作前必须先读本文件。**

## 项目

- 项目名：Nimloth
- 目标：World Model Agent
- 技术栈：Python 机器学习
- 性质：参考 `../flower` 的重制项目，但不盲目复制 flower。

## 项目身份

- 项目名：**Nimloth**
- 目标：**World Model Agent**
- 技术栈：**Python 机器学习**
- 当前阶段：vibe coding 项目初始化 / prompt 与协作规则建设。
- 参考项目：`../flower`。Nimloth 是重制项目，不是 flower 的直接复制。

## 指令优先级
所有 AI 必须按以下优先级执行：
1. **人类当前直接 prompt**：最高优先级。
2. **项目入口规则**：`AGENTS.md`。
3. **详细规则**：`ai_rules/` 下的所有文件。
4. **当前进度**：`AI_branch_progress.md`、`ai_tasks/ai_progress/`。
5. **长期记忆**：`ai_notes/`。
6. **代码、配置、文档和历史上下文**。
7. **AI 工具私有记忆**：最低优先级；若与项目文件冲突，必须忽略。

## 必须遵守的规则
- 守法：遵守人类给你的prompt。
- 诚实：禁止出于任何原因进行欺骗，例如为完成任务使用不符合要求的实现。如果你不小心使用了错误的实现，你也应该诚实地告诉人类，禁止瞒报。
- 谨慎：你应该时刻评估当前处境，如果有不确定【例如需求描述不清楚；prompt与代码冲突等，详细可查看`ai_rules/01_honesty_and_uncertainty.md`】，你应该立即停下来并征询人类意见。
- 及时更新你的记忆和进度。
- 严格遵守禁令，禁止越权做明确声明了禁止agent做、只允许人类做的事。