# Machine token 与英文残留 allowlist

## 必须逐字保留

- workflow tags：`[workflow-state:STATUS]`与对应闭合标签；实际status为`no_task`、`planning`、`planning-inline`、`in_progress`、`in_progress-inline`、`completed`。
- task status：`planning`、`in_progress`、`completed`。
- workflow parser headings：`## Phase Index`、`### Phase 1: Plan`、`### Phase 2: Execute`、`### Phase 3: Finish`、`## Phase 1: Plan`、`## Phase 2: Execute`、`## Phase 3: Finish`；为降低anchor/parser风险，全部编号phase/step headings、`### Task threshold`、`### Phase rules`、`## Platform consistency and upgrade boundary`也保持原样。
- frontmatter keys：`name`、`description`；skill ids：`git-worktree`、`memory`、`on-experiment-start`、`on-experiment-end`、`on-progress`、`slurm`及模板占位`your-skill-name`。
- fenced code blocks、CLI命令、flags、shell变量、JSON/YAML/TOML/schema field、路径、文件名、URL/Markdown link destination、hash/runtime token。
- agent/tool/skill identifiers：例如`trellis-implement`、`trellis-check`、`on-progress`、`task.py`、`get_context.py`。

## 允许保留的技术词与专有名词

- 产品/平台：Nimloth、Trellis、Pi、Claude Code、Codex、Slurm、W&B、AI、GPU、QoS、CLI、JSONL、JSON、YAML、TOML、Markdown、README、TypeScript、Git、TaskTree、CoT、SSH、VPN；
- 工作流/平台合同：agent、adapter、extension、hook、probe、reload、callback、session、context、cache、skill、task、spec、prompt、frontmatter、hash、status、phase、progress、finish-work、curated memory、known error；
- Git/结构术语：worktree、branch、commit、remote、main、cwd、dirty、diff、push、amend、merge；
- 实验/运行术语：checkpoint、rollout-train、train/freeze、job、config、split、metadata、objective、output、partition、hold allocation、bash、srun、requeue、runtime、upvote、human-only；
- 模板或机器字段：TODO、name、description、level、seed、focus、backlog、planning、in_progress、completed，以及上述词的标识符复合形式或语法要求的复数形式。

这些词只允许作为机器合同、路径/标识符、专有名词，或嵌在中文句子内且不会造成误解的常用技术词；禁止保留完整英文自然语言指令。链接标签与普通标题必须中文化，唯独上述parser/anchor headings除外。
