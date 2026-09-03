# Trellis重建执行基线（2026-09-03）

## 固定来源

- Nimloth integration base: `cbd05e5d091c9919dff42def51566e149ebfc44d` (`dev`)。
- pi-app integration base: `5dd9adc188569b4b1b4b0b6a0049bd1fb6bd39a7` (`feature/trellis-work-item-visibility`)。
- Trellis package: `/home/user/.local/share/npm/lib/node_modules/@mindfoldhq/trellis`。
- CLI输出：`0.6.16`；project metadata仍为`0.6.15`，目标发布版为`0.6.16`。

## Selected file hashes（修改前）

- `bf548ba0c713749d241f976f5641a936e891d87c76f70171a60ccef461e03fec`  `.trellis/workflow.md`
- `3a6ee34e1c30075aaf051146ac47e9f2c16673bac829f6e4dfe21a1cc51dcee5`  `.trellis/scripts/task.py`
- `727c924fb1a531b4b328fbc6cbbf85d635ba1bf767f365bec5d8b1ca2eb93b0a`  `.pi/extensions/trellis/index.ts`
- `9bb1f70d09b7104a671ef9a0ba072b4500d45b8556a253126a003e2b1e7281a2`  `.pi/agents/trellis-implement.md`
- `1dbfedd3403f201fbfdbae8d810afba0a1f812b97f0f8e308908db7eaceea496`  `.pi/agents/trellis-check.md`
- `28af1eb6645d8b517cf705277d8405370b712926e6b01667d6698564002c6a9d`  `.pi/prompts/trellis-start.md`
- `12c2f0288ff67af3368c0b577a50027a11a25fec1d32ed34282a4dc84be08f1c`  `.pi/prompts/trellis-continue.md`
- `5a8a72fd87d009c15068bae75165af97f1b03e975b24f5dc3b901b01c6914160`  `.pi/prompts/trellis-finish-work.md`

完整旧template registry比较：`template-hash-baseline-2026-09-03.json`。
统计：`{'unchanged': 99, 'modified': 4, 'missing': 0}`。

## 旧live runtime只读inventory

- 文件数：8。
- Aggregate：`{"approval_receipts": 142, "approval_requests": 151, "assignments": 312, "request_status": {"approve": 141, "comment": 1, "pending": 1, "system_cancelled": 8}}`。
- 仅记录path、bytes、hash、schema和数量；没有复制request/receipt正文。
- 完整manifest：`legacy-runtime-inventory-2026-09-03.json`。

## `trellis update --dry-run`原始输出

```text
⚠️  Trellis update available: 0.6.15 → 0.6.16
   Run: trellis update


Trellis Update
══════════════

Project version: 0.6.15
CLI version:     0.6.16
Latest on npm:   0.6.16


Scanning for changes...

  Template updated (will auto-update):
    ↑ .trellis/scripts/common/__init__.py
    ↑ .trellis/scripts/common/paths.py
    ↑ .trellis/scripts/common/developer.py
    ↑ .trellis/scripts/common/task_utils.py
    ↑ .trellis/scripts/common/active_task.py
    ↑ .trellis/scripts/common/config.py
    ↑ .trellis/scripts/common/io.py
    ↑ .trellis/scripts/common/git.py
    ↑ .trellis/scripts/common/tasks.py
    ↑ .trellis/scripts/common/task_context.py
    ↑ .trellis/scripts/common/task_store.py
    ↑ .trellis/scripts/common/workflow_phase.py
    ↑ .trellis/scripts/common/trellis_config.py
    ↑ .trellis/scripts/common/safe_commit.py
    ↑ .trellis/scripts/add_session.py
    ↑ .claude/commands/trellis/continue.md
    ↑ .claude/skills/trellis-brainstorm/SKILL.md
    ↑ .claude/skills/trellis-meta/references/local-architecture/context-injection.md
    ↑ .claude/skills/trellis-meta/references/local-architecture/task-system.md
    ↑ .claude/hooks/inject-subagent-context.py
    ↑ .claude/hooks/inject-workflow-state.py
    ↑ .claude/hooks/session-start.py
    ↑ .agents/skills/trellis-continue/SKILL.md
    ↑ .codex/hooks/session-start.py
    ↑ .codex/hooks/inject-subagent-context.py
    ↑ .codex/hooks/inject-workflow-state.py
    ↑ .pi/prompts/trellis-continue.md
    ↑ .pi/agents/trellis-check.md
    ↑ .pi/agents/trellis-implement.md

  Unchanged files (will skip):
    ○ .trellis/scripts/__init__.py
    ○ .trellis/scripts/common/git_context.py
    ○ .trellis/scripts/common/task_queue.py
    ○ .trellis/scripts/common/cli_adapter.py
    ○ .trellis/scripts/common/log.py
    ... and 104 more

  Modified by you (need your decision):
    ? .trellis/scripts/task.py
    ? .trellis/config.yaml
    ? .trellis/workflow.md
    ? AGENTS.md
    ? .agents/skills/trellis-brainstorm/SKILL.md
    ? .agents/skills/trellis-update-spec/SKILL.md
    ? .agents/skills/trellis-meta/references/local-architecture/context-injection.md
    ? .agents/skills/trellis-meta/references/local-architecture/task-system.md
    ? .pi/extensions/trellis/index.ts

  User data (preserved):
    ○ .trellis/workspace/
    ○ .trellis/tasks/
    ○ .trellis/spec/

This will UPGRADE: 0.6.15 → 0.6.16

[Dry run] No changes made.
```
