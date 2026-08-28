# Trellis 中文化 prompt surface 审查（2026-08-28）

## 审查边界

本轮仅盘点本项目实际配置的 Trellis prompt/agent/skill/workflow 入口及其模板所有权；未改写任何现有 prompt，未运行 `trellis update`。

当前配置平台：

- Claude Code：`.claude/`
- Codex：`.codex/` + shared `.agents/skills/`
- Pi Agent：`.pi/` + shared `.agents/skills/`
- Trellis channel runtime：`.trellis/agents/`

当前工具报告：

```text
⚠️  Trellis update available: 0.6.15 → 0.6.16
   Run: trellis update
```

`trellis update --dry-run` 已执行且无文件修改。它显示部分 prompt 将在 0.6.16 auto-update，部分项目自定义文件会进入“Modified by you”冲突；因此中文化会扩大本地模板 divergence，必须显式记录维护边界，禁止手工修改 `.trellis/.template-hashes.json`。

## 1. 建议的核心运行时范围

### Workflow source

| 类别 | 文件数 | 行数 |
|---|---:|---:|
| `.trellis/workflow.md` | 1 | 267 |

### Shared core workflow skills

8 个文件，约 1,102 行：

- `.agents/skills/trellis-start/SKILL.md`
- `.agents/skills/trellis-continue/SKILL.md`
- `.agents/skills/trellis-finish-work/SKILL.md`
- `.agents/skills/trellis-brainstorm/SKILL.md`
- `.agents/skills/trellis-before-dev/SKILL.md`
- `.agents/skills/trellis-check/SKILL.md`
- `.agents/skills/trellis-break-loop/SKILL.md`
- `.agents/skills/trellis-update-spec/SKILL.md`

### Platform prompts/commands/agents

16 个文件，约 1,133 行：

- Pi prompts 3 个、Pi agents 3 个；
- Claude commands 2 个、Claude agents 3 个；
- Codex agents 3 个；
- channel runtime agents 2 个。

### Project operational gate skills

6 个文件，约 315 行：

- `git-worktree`
- `memory`
- `on-experiment-start`
- `on-experiment-end`
- `on-progress`
- `slurm`

核心运行时文本共 **31 个 Markdown/TOML 文件、约 2,817 行**。此外 Pi extension 中存在少量直接注入给 AI 的英文 developer/tool instructions；若选择核心运行时中文化，应只翻译这些 AI-facing string literals，不翻译 TypeScript 代码、注释或协议 token。

## 2. Bundled upstream skills

以下四个 skill 是 Trellis CLI 随版本发布并由 `trellis update` 管理的 bundled upstream copies：

- `trellis-meta`
- `trellis-channel`
- `trellis-session-insight`
- `trellis-spec-bootstrap`

它们在 shared `.agents/skills/` 下共有 **38 个 Markdown 文件、约 3,968 行**。全部本地翻译会使这些文件长期与上游模板分叉，每次 Trellis 更新都需要人工保留/重放翻译；这不适合作为默认 MVP，除非人类明确要求“连 Trellis 自带手册也全部中文化”。

`.claude/skills` 指向 shared skill layer，不能把 symlink alias 当成另一套独立翻译目标。

## 3. Hook/CLI-generated context

`.trellis/scripts/common/session_context.py`、Claude/Codex hooks 与 Pi extension还会生成英文标题或提示，例如：

- `SESSION CONTEXT`
- `GIT STATUS`
- `CURRENT TASK`
- `Task context changed on disk...`
- `No active Trellis task found...`

这些属于 runtime-generated context/UI strings，不是纯 Markdown prompt。翻译它们会扩大到 Python/TypeScript runtime代码和平台同步测试。建议核心任务只处理真正传给 AI 的 directive strings；纯标题、CLI诊断和代码注释后置，除非人类选择完整本地中文化。

## 4. 必须原样保留的 machine contracts

中文重写不能改变以下 token/结构：

- dispatch 首行：`Active task: <path>`；
- hook marker：`<!-- trellis-hook-injected -->`；
- truncation marker：`Full hook output saved to: <path>`；
- workflow tags：`[workflow-state:STATUS]` / `[/workflow-state:STATUS]`；
- XML-like tags：`<workflow-state>`、`<trellis-task-context-update>` 等；
- task status：`planning`、`in_progress`、`completed`；
- filenames、paths、CLI commands、flags、JSON/YAML/TOML/frontmatter keys；
- skill/agent identifiers：`trellis-implement`、`trellis-check`、`trellis-research` 等；
- `.trellis/workflow.md` 中被 parser按英文精确匹配的结构标题，例如 `## Phase Index` 与 phase range headings，除非同步修改并验证所有 parser；默认保留这些结构标题。

## 5. 翻译质量要求

- 语义重写，不做逐词直译；中文必须直接、明确、保持现有授权/停止/审批强度。
- 统一术语：任务、规划、实施审批、启动审批、工作提交、归档、工作区、子代理、上下文清单、验收标准。
- 禁止把 `must`、`do not`、`only after approval` 弱化成建议性表达。
- 跨 Pi/Claude/Codex/channel 的同一角色保持责任、读序、写边界、递归保护和禁止命令一致。
- code fence、示例命令、路径和 schema保持可执行。

## 6. 人类最终范围决定

人类明确要求：只中文化项目自己维护的Trellis要求文件，不包括Trellis上游部分。

据Git历史、project migration commit、template ownership与文件职责，最终目标收敛为9个prompt/instruction文件：`.trellis/workflow.md`，以及`.agents/skills/{README.md,_template/SKILL.md,git-worktree/SKILL.md,memory/SKILL.md,on-experiment-start/SKILL.md,on-experiment-end/SKILL.md,on-progress/SKILL.md,slurm/SKILL.md}`，共约635行。

`AGENTS.md`为人类编写且已是中文，只审查不为风格改写；`.trellis/spec/`已经是项目中文合同但不是prompt重写目标。所有upstream workflow skills、bundled manuals、platform prompts/agents/commands、hooks/scripts/extensions/config与template state均排除。
