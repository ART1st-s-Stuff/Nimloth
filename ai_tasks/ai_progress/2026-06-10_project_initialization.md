# 2026-06-10 Project Initialization

## 任务

交互式初始化 Nimloth，一个参考 `../flower` 的 World Model Agent 重制项目，供多个编程 AI 协作。

## 当前人类指令

- 无需创建代码结构。
- refine prompts。
- 把详细规则都写入 `ai_rules/` 下面。

## 已完成

- 阅读参考项目 `../flower/AGENTS.md`、`../flower/README.md`、`../flower/AI_README.md` 的相关部分。
- 创建/更新 prompt 与规则入口：
  - `AGENTS.md`
  - `AI_README.md`
  - `README.md`
- 创建 `ai_rules/` 详细规则：
  - `ai_rules/README.md`
  - `ai_rules/00_priority_and_identity.md`
  - `ai_rules/01_honesty_and_uncertainty.md`
  - `ai_rules/02_memory_and_progress.md`
  - `ai_rules/03_experiments_and_data.md`
  - `ai_rules/04_code_and_repo.md`
  - `ai_rules/05_flower_reference.md`
- 创建/更新进度和 issue 文件：
  - `AI_branch_progress.md`
  - `AI_issues.md`
- 创建长期记忆入口：
  - `ai_notes/main.md`
  - `ai_notes/today.txt`
- 已移除先前误建的代码/实验空目录与 `src/nimloth/__init__.py`。

## 待确认

- `ai_rules/` 是否需要进一步细分或合并。
- 是否需要为不同 AI 工具创建专门入口文件。
- 是否初始化 git。
