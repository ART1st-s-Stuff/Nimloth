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

## 语言规范
- 语言清晰：你的所有解释/概念必须清晰明确，禁止充斥看不明白的专业术语，绝对严禁自己发明新词
- 一致性：在整个项目里相同的概念必须具有相同的名字
- 说人话：禁止充斥"不是……而是……"这种AI slop，除非真的有一个错误的东西需要你澄清

## 项目规则目录
所有文件均位于`ai_rules`下，你可以根据你的任务来阅读：
- `01_honesty_and_uncertainty.md`: AI行为准则，执行所有任务前必读
- `02_memory_and_progress.md`: 记忆系统，执行所有任务前必读
- `03_experiments_and_data.md`: 实验相关行为准则。做实验、更改实验代码前必读
- `04_code_and_repo.md`: 代码与repo规则。动代码之前必读
- `events/on_progress.md`: 完成子任务/修复关键问题/确认重要设计后必读
- `events/on_experiment_start.md`: 任何实验性任务开始前必读
- `events/on_experiment_end.md`: 任何实验性任务结束后必读

## 事件触发器（Event Hooks）

AI 在以下时刻必须立即暂停当前工作，去读取并执行对应事件文件。这不是"选读"，而是硬性触发条件。

| 触发条件 | 必须读取 |
|---------|---------|
| 完成一个子任务 / 修复一个关键 bug / 确认一个重要设计决策 / 改变项目规则 / 发现旧结论失效 | `ai_rules/events/on_progress.md` |
| 任何训练、评估、采集、校准、rollout-train、远程长任务、Slurm 任务、或其他需要 GPU/长时间运行的计算任务开始前 | `ai_rules/events/on_experiment_start.md` |
| 任何上述实验性任务结束、失败、取消或暂停后（无论任务是否由当前会话启动） | `ai_rules/events/on_experiment_end.md` |

### 触发机制说明

- 上述触发条件绑定到 AI 的**具体动作**上，不是抽象概念。AI 不需要自己判断"我是不是取得了阶段性进展"——当你完成了上面描述的具体动作时，就触发。
- 事件文件读完后，必须**逐条执行其中列出的步骤**，不得跳过。
- 如果事件文件中要求你"评估 memory"或"更新进度文件"，你必须在当前对话中完成，不能推迟。

## Git worktree
在本地修改代码时，你应该使用../nimloth-<branch-name> （分支名的`/`在文件夹名称内使用`-`），不要直接在main分支修改，除非Prompt里有显式说明。

## 服务器使用规范
在连接服务器以前，你必须阅读并参考 `.local/SERVER.md`

任何服务器上的代码都是不可靠的，禁止在服务器上修改代码，所有代码都必须先在本地更改后使用git同步
