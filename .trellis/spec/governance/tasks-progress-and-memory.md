# Tasks, Progress, and Curated Memory

## One live task authority

Trellis is Nimloth's only development task system. Keep Pi TaskTree empty: do not copy Trellis status, priority, hierarchy, focus, acceptance criteria, or backlog entries into it.

A Trellis task is mandatory for:

- multi-file or ambiguous implementation;
- project-rule or workflow changes;
- experiments, GPU work, Slurm/remote long jobs, collection, evaluation, or rollout-train;
- work needing durable design, handoff, or multi-session progress.

When there is no active task, follow upstream task-creation consent even for simple or trivial code work: ask whether to create the task before creating it. An implementation request does not silently supply task-creation consent. If the human declines, only a one-reply explanation, read-only lookup with no durable decision, or clearly bounded low-risk inline edit may proceed. Experiment launch follows the scoped short/long rules in the experiment contract.

A tightly related low-risk correction inside an existing reviewed scope uses the Fast path and does not create another task. Multi-file/public-contract work uses the Standard path; experiments, remote/destructive operations, protected data and push/merge use the High-risk path.

## Plan authority and optional visibility

- `task.json.parent/children/status` owns task-tree identity and lifecycle; `implement.md` headings, item order/text and checkboxes own the execution plan and done state.
- Human-readable plan text does not require embedded machine IDs. An optional project extension may derive internal fingerprints for live visibility, but it must not rewrite `implement.md`, copy the plan into another writable store, or make its runtime a prerequisite for ordinary Trellis lifecycle commands.
- Optional live assignment/progress data is non-authoritative and gitignored. Missing, stale, invalid, or unavailable visibility data degrades only the display; task artifacts and upstream Trellis remain usable.
- Pi-app and other consumers are read-only projections. They must not mutate task status, synthesize completion, maintain approval receipts, or read/write Pi TaskTree.
- Task, implementation, commit, and integration decisions remain explicit human conversation at the applicable workflow boundary. Comments, silence, task creation, or lifecycle status do not imply approval.

## Persistence routing

- `.trellis/tasks/`: current requirements, design, plan, research, checks, unresolved decisions, and execution state.
- `.trellis/workspace/`: per-session journal written during wrap-up.
- `AI_branch_progress.md`: concise migration-period branch milestones, not detailed task state.
- `ai_tasks/` and `AI_issues.md`: historical evidence only; do not create new `ai_tasks/ai_progress/` records or add new live issues there.
- `trellis mem`: read-only raw dialogue recall; never verified truth.
- `.memory/` and `.local/memory/`: compact, evidence-backed, human-reviewed reusable lessons.

Apply `on-progress` when a work-item completes, risk/design changes, experiment state changes, or a cross-session handoff is needed. Consecutive fixes within one work-item are consolidated; they do not each create progress prose, memory review, or bookkeeping commits. Update a branch milestone only when branch-level state actually changes.

## Curated memory contract

Use [`.agents/skills/memory/SKILL.md`](../../../.agents/skills/memory/SKILL.md) and the `./skill memory ...` wrapper.

- Repo memory (`.memory/memories.jsonl`) is environment-independent; local memory (`.local/memory/memories.jsonl`) is machine/server/workspace-specific.
- Never edit either JSONL manually.
- Memory stores short reusable lessons, constraints, decisions, or lookup hints. It does not store task logs, TODOs, experiment summaries, rules already clear in specs, or source documentation.
- AI-created entries remain `pending-human-verification`; only a human may run `./skill human memory-approve`.
- Do not claim an entry is human-approved unless its level is `verified`.
- Before relying on an entry, run `get`, re-read its evidence segment, and confirm the evidence still supports it.
- Upvote only after that verification and only if the memory genuinely helped this task.
- Correct wrong memory through the skill; do not conceal a stale or conflicting entry. Follow `human_suggestions` before requesting approval again.
- Stable mandatory rules belong in spec. Do not keep duplicated prose in both spec and memory.

When pending memory was added or revised, remind the human that approval is available; the AI must not run the human-only command.

## Historical evidence

Do not rewrite existing `ai_tasks/`, `AI_issues.md`, or old `AI_branch_progress.md` entries for style. Historical references to the pre-Trellis paths remain evidence of what happened at that time and do not restore their authority for new work.
