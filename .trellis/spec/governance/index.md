# Governance

## Applicability and authority

This layer applies to every Nimloth task and every supported AI platform. It owns cross-repository safety and collaboration rules. The current human prompt and [`AGENTS.md`](../../../AGENTS.md) safety kernel outrank this layer; task artifacts cannot relax a hard rule unless the human explicitly approves that rule change.

## Pre-Development Checklist

- Read `AGENTS.md`, [authority and safety](authority-and-safety.md), and the active Trellis task artifacts.
- Classify uncertainty, authorization, protected-file, main-branch, experiment, and CoT/state risk before editing.
- Confirm the current worktree and branch with `git status --short --branch`.
- Load only the topic specs, source documents, and [relevant known errors](../guides/known-error-routing.md) needed by the task.
- Use Trellis as the only live development task system; do not mirror work into Pi TaskTree.

## Quality Check

- The implementation remains inside the approved scope and does not hide temporary stand-ins or uncertainty.
- Protected files, memory JSONL, data, checkpoints, outputs, and unrelated dirty files are unchanged unless explicitly authorized.
- Task state, validation evidence, unresolved decisions, and milestone progress are routed to the correct stores.
- Git/worktree and human-review gates were followed.
- Any CoT-conditioned state uses the observation's real recorded/generated CoT.

## Topic specs

- [Authority, honesty, authorization, and platform entry](authority-and-safety.md)
- [CoT and state semantics](cot-and-state.md)
- [Tasks, progress, TaskTree, and curated memory](tasks-progress-and-memory.md)
- [Git, worktrees, protected files, and review](git-worktrees-and-protected-files.md)
- [Pi/Claude/Codex platform integration contract](platform-integration.md)

## Live operational sources

- [Trellis workflow](../../workflow.md)
- [Project-local skills](../../../.agents/skills/README.md)
- [Known-errors index](../../../ai_rules/known_errors/README.md)
