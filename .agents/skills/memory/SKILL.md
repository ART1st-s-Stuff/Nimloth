---
name: memory
description: 轻量、需人类批准的项目memory管理。创建、搜索、检查、纠正或upvote持久项目memory时使用。
---

# memory skill

需要创建、搜索、检查、纠正或upvote持久项目memory时，必须使用本skill。

## 用途

Memory系统是一个轻量、需人类批准且可搜索的存储，用于保存从真实工作中提炼出的简短项目经验。

它包含两个存储区：

- 仓库memory：`.memory/memories.jsonl`，存放与环境无关、应随仓库提交的经验；
- 本地memory：`.local/memory/memories.jsonl`，存放特定于环境/服务器/工作空间的经验。

Memory必须满足：

- 简短；
- 对未来AI agent有用；
- 是从实际工作中提炼的有效项目经验、约束、决策或查找提示；
- 有文件片段证据支持；
- 不重复规则、进度文件、实验文档或源码文档；
- 不是任务日志；
- 不是长篇说明。

Memory通常应回答：“哪条紧凑经验能避免未来agent重复这次发现或错误？”如果信息已清楚存在于`AGENTS.md`、`.trellis/spec/`、实验/模块README或代码注释中，应优先链接并阅读原始来源，而不是创建重复memory。

## 命令

使用仓库skill封装命令：

```bash
./skill memory add <title> <content>
./skill memory add --store local <title> <content>
./skill memory set <id> <field=value> [field=value ...]
./skill memory set --store local <id> <field=value> [field=value ...]
./skill memory search <keyword-regex> [--store all|repo|local] [--field all|title|content|evidence.filename|tags] [--tag TAG] [--level LEVEL] [--include-archived]
./skill memory get --store repo|local <id>
./skill memory upvote --store repo|local <id>
./skill memory human-verify --store repo|local <id>
```

仅限人类的审批命令：

```bash
./skill human memory-approve
./skill human memory-approve --store local
```

AI agent绝对禁止运行任何`./skill human ...`命令。

## 数据模型

每条memory包含`id`、`title`、`content`、`evidence`、`tags`、`level`、时间戳和可选`human_suggestions`。

- `evidence`：文件片段引用的JSON列表：`[{"filename":"...","line_start":1,"total_lines":10}]`
- `tags`：字符串JSON列表
- `level`：`pending-human-verification`、`verified`或`archived`

## AI agent 规则

1. 禁止手工编辑`.memory/memories.jsonl`或`.local/memory/memories.jsonl`。
2. 禁止创建过长memory。每条memory应只保存一条紧凑、可搜索的经验。
3. 禁止把临时进度、TODO、任务日志或实验摘要存入memory。
4. 如果内容只是重复规则、文件清单、命令帮助或容易找到的现有文档，禁止创建memory。
5. 证据必须是文件片段引用，禁止使用自由文本。
6. AI创建的memory初始level必须为`pending-human-verification`。
7. 除非memory的level为`verified`，否则AI禁止声称它已经人类批准。
8. 如果待核验memory包含`human_suggestions`，AI必须先按建议使用`./skill memory set ...`修订memory，才能再次请求审批。
9. 依赖memory之前，必须运行`./skill memory get <id>`、检查证据文件片段，并核验memory仍与引用文件一致。
10. 只有完成上述核验并确认该memory对当前任务确有帮助后，才能运行`./skill memory upvote <id>`。
11. Memory错误时，必须使用`./skill memory set ...`纠正；若已过时，则交由过期归档规则处理，或询问人类。
12. 人类审批必须通过`./skill human memory-approve`完成；本地memory使用带`--store local`的命令。AI禁止代为执行。

## 人类审批流程

如果一条待核验memory保存了紧凑项目经验，且不重复现有文档，AI可以提交它：

```bash
./skill memory add "Dataset split must be verified from loader metadata" "For Nimloth experiments, split names alone are not evidence; verify split semantics from the actual dataset/config/code path before launch."
./skill memory set M0001 'evidence=[{"filename":".trellis/spec/experiments/data-and-splits.md","line_start":1,"total_lines":18}]' 'tags=["experiments","data","split"]'
./skill memory human-verify M0001
```

人类通过以下命令审查待核验memory：

```bash
./skill human memory-approve
```

批准后memory变为`verified`；拒绝后memory会被删除。如果人类输入的内容不是`a/r/s/q`之一，该文本会存入`human_suggestions`，memory保持待核验状态，AI必须按建议修订。Memory获批时，建议会自动移除。

## 过期/归档策略

CLI采用延迟的过期清理。已核验memory满足以下任一条件时会归档：

- 连续7天未通过触发核验；
- 连续14天未被upvote/使用。

`upvote`表示：agent先核验了证据，随后确认该memory对当前任务有用。
