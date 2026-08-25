---
name: on-progress
description: >-
  Records substantive Nimloth progress in Trellis and evaluates curated memory.
  Use after a verifiable subtask, critical fix, important design decision,
  experiment stage, project-rule change, or invalidated conclusion.
---

# On Progress

## Trigger

Pause and apply this skill immediately after any listed substantive milestone. Do not defer it to a later conversation.

## Required actions

1. Read [tasks, progress, and memory](../../../.trellis/spec/governance/tasks-progress-and-memory.md) and the active task plan.
2. Update current detail, evidence, unresolved decisions, and the execution checklist in the active Trellis task.
3. Add a concise `AI_branch_progress.md` milestone only when branch-level state changed. Do not create new `ai_tasks/ai_progress/` records or add live work to `AI_issues.md`.
4. Evaluate whether the milestone produced a compact reusable lesson not already clear in specs/docs. Create memory only through the `memory` skill; never edit JSONL.
5. For every memory used, `get` it again, re-read evidence, and upvote only if still correct and genuinely helpful. Correct errors through the skill; unresolved conflicts require a human question.
6. If pending memory was added/revised, remind the human about approval. Never run `./skill human ...`.
