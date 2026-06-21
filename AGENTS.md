--------
本文为人类编写。如需修改需要得到人类同意。
--------

# AGENTS.md — Nimloth AI 入口

**所有编程 AI 开始工作前必须先读本文件。**

## 项目

- 项目名：Nimloth
- 目标：World Model Agent
- 技术栈：Python 机器学习

## 项目身份

- 项目名：**Nimloth**
- 目标：**World Model Agent**
- 技术栈：**Python 机器学习**
- 当前阶段：vibe coding 项目初始化 / prompt 与协作规则建设。

## 指令优先级
所有 AI 必须按以下优先级执行：
1. **人类当前直接 prompt**：最高优先级。
2. **项目入口规则**：`AGENTS.md`。
3. **详细规则**：`ai_rules/` 下的所有文件。
4. **当前进度**：`AI_branch_progress.md`、`ai_tasks/ai_progress/`。
5. **长期记忆**: memory skill。
6. **代码、配置、文档和历史上下文**。
7. **AI 工具私有记忆**：最低优先级；若与项目文件冲突，必须忽略。

## 必须遵守的规则
- 守法：遵守人类给你的prompt。遵守`ai_rules/` 下的所有规则。
- 诚实：禁止出于任何原因进行欺骗，例如为完成任务使用不符合要求的实现。如果你不小心使用了错误的实现，你也应该诚实地告诉人类，禁止瞒报。
- 谨慎：你应该时刻评估当前处境，如果有不确定【例如需求描述不清楚；prompt与代码冲突等，详细可查看`ai_rules/01_honesty_and_uncertainty.md`】，你应该立即停下来并征询人类意见。
- 及时更新你的记忆和进度。
- 在任务过程中，可以随时通过 memory SKILL 使用和更新记忆；具体协议见 `.agents/skills/memory/SKILL.md`。
- repo 记忆存放在 `.memory/`，本地/环境相关记忆存放在 `.local/memory/`；都不得手动编辑对应的 `memories.jsonl`。
- 严格遵守禁令，禁止越权做明确声明了禁止agent做、只允许人类做的事。

## 项目规则目录
所有文件均位于`ai_rules`下，你可以根据你的任务来阅读：
- `01_honesty_and_uncertainty.md`: AI行为准则，执行所有任务前必读
- `02_memory_and_progress.md`: 记忆系统，执行所有任务前必读
- `03_experiments_and_data.md`: 实验相关行为准则。做实验、更改实验代码前必读
- `04_code_and_repo.md`: 代码与repo规则。写代码之前必读
- `events/on_progress.md`: 任务有进展时需要做的事

## 服务器使用规范
参考 `.local/SERVER.md`