# PRD：用中文重写项目自维护的 Trellis prompts

## Goal

把本项目自己维护的 Trellis workflow 与 Nimloth operational skill prompts 用清晰、一致的中文重写，同时保持现有生命周期、授权门禁、角色边界和机器解析合同完全不变；Trellis 上游模板内容不纳入本任务。

## Scope decision

2026-08-28 人类明确选择：

> 中文化所有由项目自己维护的 Trellis 要求文件，不包括 Trellis 上游自己的部分。

因此按文件ownership而不是“本地是否存在”划分范围。Trellis CLI生成并持续更新的platform prompts、agents、commands、workflow skills与bundled manuals均视为上游内容，不翻译。

## In scope

### 需要中文重写的9个项目维护文件

1. `.trellis/workflow.md`；
2. `.agents/skills/README.md`；
3. `.agents/skills/_template/SKILL.md`；
4. `.agents/skills/git-worktree/SKILL.md`；
5. `.agents/skills/memory/SKILL.md`；
6. `.agents/skills/on-experiment-start/SKILL.md`；
7. `.agents/skills/on-experiment-end/SKILL.md`；
8. `.agents/skills/on-progress/SKILL.md`；
9. `.agents/skills/slurm/SKILL.md`。

共约635行。`AGENTS.md`是人类编写且已使用中文，只做覆盖/术语审查；除非发现真实合同缺口，不修改其措辞。

### Audit only

- `AGENTS.md`；
- `.trellis/spec/`：项目合同已主要使用中文，不属于prompt重写目标；
- `.trellis/config.yaml`：配置与注释，不是AI prompt；
- `.pi/extensions/trellis/index.ts`：本地有adapter customization，但其prompt主体来自上游，本任务不拆分翻译。

## Out of scope

- Trellis上游single-file workflow skills：`trellis-start`、`trellis-continue`、`trellis-finish-work`、`trellis-brainstorm`、`trellis-before-dev`、`trellis-check`、`trellis-break-loop`、`trellis-update-spec`；
- bundled upstream skills及references：`trellis-meta`、`trellis-channel`、`trellis-session-insight`、`trellis-spec-bootstrap`；
- `.pi/prompts/`、`.pi/agents/`、`.claude/commands/`、`.claude/agents/`、`.codex/agents/`、`.trellis/agents/`；
- hooks、`.trellis/scripts/`、Pi extension、CLI/session headings与runtime error messages；
- `.trellis/.template-hashes.json`与`.trellis/.runtime/`；
- Trellis upstream npm package或全局安装目录；
- 执行真实`trellis update`；
- 改变workflow semantics、task status、phase顺序或审批门禁；
- worktree迁移或其他active task内容。

## Requirements

### R1 — 中文语义重写

- 对scope内自然语言指令做中文语义重写，不做机械逐词翻译。
- 保持每条must/forbidden/approval/stop条件的强度和适用对象。
- 不新增、删除或重新解释Trellis phase、task status、agent responsibility或experiment gate。

### R2 — 术语一致

统一使用：任务、规划阶段、实施审批、启动审批、验收标准、工作提交、记账提交、worktree、子代理、上下文清单、停止条件、回滚。

命令、status值、文件名、skill id、schema/field名保持英文。

### R3 — Machine contracts

以下内容必须逐字保留：

- `[workflow-state:STATUS]` tags；
- task status：`planning`、`in_progress`、`completed`；
- CLI commands/flags、paths、JSON/YAML/frontmatter keys；
- parser精确匹配的workflow headings；
- tool/agent/skill identifiers。

`.trellis/workflow.md` 默认保留`## Phase Index`、`## Phase 1: Plan`、`## Phase 2: Execute`、`## Phase 3: Finish`等parser headings，只中文化正文。

### R4 — Project-contract preservation

- `AGENTS.md`、governance/experiment specs与中文prompt逐项一致。
- `git-worktree`中文化以当前task已确认的新目标为准：canonical root最终为`/workspace/remote2/nimloth`，必要child worktree位于`nimloth/.worktree/`；但实际路径规则改动仍归worktree任务实施，本任务不能抢先改变该合同。
- experiment/progress/memory/slurm skills不得弱化单独审批、protected memory、remote exact source或human-only边界。

### R5 — Template ownership

- 不修改上游prompt以追求“全中文”。
- 不手工编辑template hashes。
- 中文化后运行`trellis update --dry-run`，确认项目维护文件与上游文件边界清晰；不执行真实update。

### R6 — Validation

- Markdown links、code fences、frontmatter结构有效。
- workflow-state blocks成对且phase extraction仍可用。
- 英文残留扫描仅允许machine token、命令、路径、标识符、专有名词或必要代码示例。
- `task.py validate`与`git diff --check`通过。
- 独立review确认没有语义弱化、越界翻译或遗漏项目维护prompt。

## Acceptance Criteria

- [x] 9个scope文件的自然语言指令已使用一致中文；英文残留均有allowlist理由。
- [x] `AGENTS.md`与相关spec审查完成；没有为风格改写人类文件。
- [x] workflow phases、审批门禁、experiment launch boundary、memory/progress/worktree/slurm safety无弱化或扩张。
- [x] parser-coupled tags/status/headings逐字保留并通过自动检查。
- [x] Markdown/frontmatter/workflow extraction验证通过。
- [x] changed-file清单不包含任何上游prompt/agent/command/bundled skill、hook、script、extension或其他active task文件。
- [x] `trellis update --dry-run`仅作为证据运行，没有实际覆盖。
- [x] 完整diff、英文allowlist、验证结果和残余维护风险已展示。
- [x] 本任务完成后返回`08-28-refactor-local-worktree-layout`规划上下文，不自动开始删除worktree。

## Open questions

无。
