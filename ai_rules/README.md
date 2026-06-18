# ai_rules — Nimloth AI 协作规则

本目录保存 Nimloth 项目的详细 AI 协作 prompt / rules。所有编程 AI 开始工作前必须阅读：

1. `../AGENTS.md`：入口与读取顺序；
2. 本目录下所有规则文件，至少包括：
   - `00_priority_and_identity.md`
   - `01_honesty_and_uncertainty.md`
   - `02_memory_and_progress.md`
   - `03_experiments_and_data.md`
   - `04_code_and_repo.md`
3. 当任务触发特定事件时，还必须阅读并执行 `events/` 下对应事件规则。

## 规则优先级

若规则冲突，优先级如下：

1. 人类当前直接 prompt；
2. `AGENTS.md`；
3. `ai_rules/`；
4. `AI_branch_progress.md`、memory skill、任务文件；
5. 代码和历史上下文；
6. AI 工具私有记忆。

若无法判断冲突如何解决，停止并询问人类。
