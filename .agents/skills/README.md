# Nimloth Project Skills

This is the repository-owned skill layer shared by Pi, Claude Code, and Codex. `.claude/skills` points here; other platforms discover `.agents/skills/` directly.

## Authority boundary

- Nimloth rules live in [`AGENTS.md`](../../AGENTS.md), [`.trellis/workflow.md`](../../.trellis/workflow.md), and [`.trellis/spec/`](../../.trellis/spec/).
- Non-`trellis-*` directories below are project-owned operational capabilities.
- `trellis-*` skills are upstream-managed Trellis assets. Do not add Nimloth-private rules to them; `trellis update` may replace or flag them.
- Machine/server-specific facts remain under ignored `.local/`, especially `.local/SERVER.md` and local memory. Portable skills are committed entity directories, not absolute symlinks to another worktree.

## Project-owned skills

| Skill | Purpose | Live authority |
|---|---|---|
| `memory/` | evidence-backed, human-reviewed durable lessons | [`governance/tasks-progress-and-memory.md`](../../.trellis/spec/governance/tasks-progress-and-memory.md) |
| `on-progress/` | milestone/task/progress/memory routing | same governance spec + active task |
| `on-experiment-start/` | separate experiment launch gate | [`experiments/`](../../.trellis/spec/experiments/index.md) |
| `on-experiment-end/` | mandatory terminal run recording | [`experiments/launch-and-lifecycle.md`](../../.trellis/spec/experiments/launch-and-lifecycle.md) |
| `git-worktree/` | branch worktree creation/validation | [`governance/git-worktrees-and-protected-files.md`](../../.trellis/spec/governance/git-worktrees-and-protected-files.md) |
| `slurm/` | portable Slurm lifecycle; machine details from `.local/` | [`experiments/launch-and-lifecycle.md`](../../.trellis/spec/experiments/launch-and-lifecycle.md) |
| `_template/` | project skill template | this README |

## New project capability

Create a repository-owned `.agents/skills/<name>/SKILL.md` with valid frontmatter and link it to live Trellis authority. Do not add it to `.gitignore` unless the entire capability is intentionally machine-specific; machine-specific configuration should normally be placed under `.local/` while the portable workflow remains versioned.

Skill names use lowercase letters, digits, and hyphens, up to 64 characters.
