# Nimloth 项目 Skills

这是仓库自有的skill层，由Pi、Claude Code与Codex共享。`.claude/skills`指向这里；其他平台直接发现`.agents/skills/`。

## 权限边界

- Nimloth规则位于[`AGENTS.md`](../../AGENTS.md)、[`.trellis/workflow.md`](../../.trellis/workflow.md)和[`.trellis/spec/`](../../.trellis/spec/)。
- 下列非`trellis-*`目录是项目自有的操作能力。
- `trellis-*` skills是由上游管理的Trellis资产。禁止向其中加入Nimloth私有规则；`trellis update`可能替换或标记这些内容。
- 机器/服务器专用事实必须留在被忽略的`.local/`下，尤其是`.local/SERVER.md`和本地memory。可移植skill必须是已提交的实体目录，禁止使用指向其他worktree的绝对符号链接。

## 项目自有 skills

| Skill | 用途 | 当前权威合同 |
|---|---|---|
| `memory/` | 有证据且经人类审查的持久经验 | [`governance/tasks-progress-and-memory.md`](../../.trellis/spec/governance/tasks-progress-and-memory.md) |
| `on-progress/` | 里程碑、任务、进度与memory路由 | 同一governance spec与当前任务 |
| `on-experiment-start/` | 单独的实验启动门禁 | [`experiments/`](../../.trellis/spec/experiments/index.md) |
| `on-experiment-end/` | 强制记录实验终止状态 | [`experiments/launch-and-lifecycle.md`](../../.trellis/spec/experiments/launch-and-lifecycle.md) |
| `git-worktree/` | 创建并验证branch worktree | [`governance/git-worktrees-and-protected-files.md`](../../.trellis/spec/governance/git-worktrees-and-protected-files.md) |
| `slurm/` | 可移植的Slurm生命周期；机器细节来自`.local/` | [`experiments/launch-and-lifecycle.md`](../../.trellis/spec/experiments/launch-and-lifecycle.md) |
| `_template/` | 项目skill模板 | 本README |

## 新建项目能力

新建仓库自有的`.agents/skills/<name>/SKILL.md`，提供有效frontmatter，并链接到当前Trellis权威合同。除非整个能力确定只适用于单台机器，否则禁止把它加入`.gitignore`；通常应把机器专用配置放在`.local/`下，同时将可移植工作流纳入版本控制。

Skill名称只能使用小写字母、数字和连字符，最长64个字符。
