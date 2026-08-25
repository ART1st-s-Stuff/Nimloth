# Nimloth Development Workflow

Trellis is Nimloth's primary and only live development task system. This workflow keeps Trellis's native `planning` → `in_progress` → `completed/archive` status contract while applying the project safety, experiment, progress, and review gates in [`AGENTS.md`](../AGENTS.md) and [`.trellis/spec/`](spec/).

## Core Principles

1. **Stop on uncertainty or missing authorization.** Research first when read-only evidence can resolve it; otherwise ask the human.
2. **Plan before implementation.** Task creation approves planning only. Reviewed task activation approves implementation. Experiment launch has a separate approval.
3. **Inject selected evidence.** Curate relevant specs, research, and individual known errors; never inject the full known-error library.
4. **Persist by ownership.** Tasks hold current work, workspace journals hold sessions, curated memory holds reviewed reusable lessons, and legacy task/issue files remain history.
5. **Review before commits.** Present the complete scope and validation before work commits; archive/journal bookkeeping comes only after finish review.

## Trellis System

- Tasks: `.trellis/tasks/<task>/` with `task.json`, `prd.md`, optional `design.md`/`implement.md`, research, and JSONL context manifests.
- Specs: `.trellis/spec/{governance,experiments,python,domains,guides}/`.
- Workspace: `.trellis/workspace/` session journals.
- Curated memory: `.memory/`, `.local/memory/`, and `.agents/skills/memory/`; never edit memory JSONL directly.
- Raw recall: `trellis mem` is unverified dialogue search.

Useful commands:

```bash
python3 ./.trellis/scripts/task.py current --source
python3 ./.trellis/scripts/task.py create "<title>" --slug <name>
python3 ./.trellis/scripts/task.py start <task>
python3 ./.trellis/scripts/task.py validate <task>
python3 ./.trellis/scripts/get_context.py --mode packages
python3 ./.trellis/scripts/get_context.py --mode phase --step <X.Y>
```

## Phase Index

```text
Phase 1: Plan    → classify risk, obtain task consent, research, persist and review artifacts
Phase 2: Execute → implement approved scope, apply progress/experiment gates, verify repeatedly
Phase 3: Finish  → full-scope check, memory/spec review, complete-diff review, work commits, wrap-up
```

### Task threshold

A Trellis task is mandatory for multi-file or ambiguous implementation; project-rule/workflow changes; any experiment, GPU, Slurm, remote long job, collection, evaluation, or rollout-train; and work needing durable design, handoff, or multiple sessions.

Ask for task-creation consent before creating a task. If consent is declined, broad work stops. A one-reply explanation, read-only lookup with no durable decision, or clearly bounded low-risk small edit may remain inline after the user declines a task.

Trellis is the sole task authority. Pi TaskTree stays empty and receives no mirrored status, priority, hierarchy, focus, acceptance criteria, or backlog.

[workflow-state:no_task]
No active task. Classify the request and ask for Trellis task-creation consent before creating one.
A task is mandatory for multi-file/ambiguous work, rule/workflow changes, experiments/remote jobs, and durable or multi-session work. If consent is declined, do not perform broad work; only an explanation, read-only lookup, or clearly bounded low-risk small edit may remain inline.
Keep Pi TaskTree empty; Trellis is the sole development task authority.
[/workflow-state:no_task]

### Phase 1: Plan

- 1.0 Create task `[required · once]`
- 1.1 Requirements and risk exploration `[required · repeatable]`
- 1.2 Evidence research `[optional · repeatable]`
- 1.3 Configure selected context `[required · once for sub-agent platforms]`
- 1.4 Final planning review and activate `[required · once]`
- 1.5 Completion criteria

[workflow-state:planning]
Stay in planning. Task creation is not implementation approval.
For complex work, complete and review `prd.md`, `design.md`, and `implement.md`; select relevant specs, source evidence, individual known errors, and verified memory. Curate both JSONL manifests for sub-agent platforms.
For experiments, set `meta.kind=experiment`, complete the experiment contract, and preserve a separate explicit launch-approval gate.
Ask the human to approve the final artifacts before `task.py start`.
[/workflow-state:planning]

[workflow-state:planning-inline]
Stay in planning. Task creation is not implementation approval.
Complete the required task artifacts, selected specs/source/known-error/memory evidence, and experiment contract when applicable. Inline platforms may skip JSONL curation but must load the same evidence before editing.
Ask the human to approve the final artifacts before `task.py start`.
[/workflow-state:planning-inline]

### Phase 2: Execute

- 2.1 Implement `[required · repeatable]`
- 2.2 Quality check `[required · repeatable]`
- 2.3 Roll back or re-plan `[on demand]`

[workflow-state:in_progress]
Execute only the reviewed task scope. Read curated JSONL entries, then `prd.md`, `design.md`, and `implement.md`; inspect adjacent source/tests before editing.
Main session flow: `trellis-implement` -> `trellis-check` -> memory/spec review -> complete-diff and validation review -> approved work commits -> `/trellis:finish-work`.
Sub-agent recursion guard: an active `trellis-implement` or `trellis-check` agent performs its role directly and must not spawn either role again.
After substantive milestones apply `on-progress`. Experiments require a complete contract and separate explicit launch approval; completion/failure/cancellation/pause triggers `on-experiment-end`.
Final checking is full-scope across every affected spec layer and includes relevant known errors, links/config/hooks, and semantic evidence.
[/workflow-state:in_progress]

[workflow-state:in_progress-inline]
Execute only the reviewed task scope. Load task artifacts and relevant governance/domain/experiment/Python specs, source evidence, individual known errors, and verified memory before editing.
Flow: before-development review -> edit -> full-scope check -> memory/spec review -> complete-diff and validation review -> approved work commits -> `/trellis:finish-work`.
After substantive milestones apply `on-progress`; experiments keep separate launch and mandatory end gates.
[/workflow-state:in_progress-inline]

### Phase 3: Finish

- 3.2 Debug retrospective `[on demand]`
- 3.3 Progress, memory, and spec review `[required · once]`
- 3.4 Complete-diff review and work commits `[required · once]`
- 3.5 Finish-work review and bookkeeping `[required · once]`

[workflow-state:completed]
Work commits are complete. Present finish-work/archive/journal effects and obtain human acceptance before `/trellis:finish-work` performs automatic bookkeeping commits.
[/workflow-state:completed]

### Phase rules

1. Follow steps in order; required gates cannot be skipped.
2. Return to planning when requirements, scope, semantics, or authorization change.
3. Artifact presence skips repeated creation, not review of current content.
4. Task artifacts cannot override the human prompt, `AGENTS.md`, or hard specs without explicit human approval.
5. Do not launch experiments, mutate protected files, edit memory JSONL, or commit outside the matching gate.

## Phase 1: Plan

Goal: establish authorized, source-backed, verifiable work before implementation.

#### 1.0 Create task `[required · once]`

After task-creation consent:

```bash
python3 ./.trellis/scripts/task.py create "<title>" --slug <name>
```

Do not run `start` yet. Use parent/child tasks only for independently verifiable deliverables; write dependency order in child artifacts. Skip creation when `task.py current --source` already points to the approved task.

#### 1.1 Requirements and risk exploration `[required · repeatable]`

Write requirements, exclusions, authorization, acceptance criteria, and unresolved decisions to `prd.md`. Complex or risky work also requires:

- `design.md`: ownership, contracts, alternatives, compatibility, rollback;
- `implement.md`: ordered edits, verification commands, review/approval gates.

During planning:

- apply [governance](spec/governance/index.md) and [investigation/uncertainty](spec/guides/investigation-and-uncertainty.md);
- identify protected files, main/worktree risk, CoT/state semantics, experiment/remote operations, and unrecognized dirty files;
- ask one clear question when evidence cannot resolve a required decision;
- never use a temporary stand-in or approximate mechanism without explicit approval.

For an experiment, set `task.json.meta.kind = "experiment"` and add every field in [the experiment task contract](spec/experiments/task-contract.md). Implementation approval still does not approve launch.

#### 1.2 Evidence research `[optional · repeatable]`

Research current source, tests, configs, data/metadata, module READMEs, external references, or runtime behavior. Persist durable findings under the task's `research/` directory. Distinguish verified facts, assumptions, and open decisions.

Use the [known-error index](../ai_rules/known_errors/README.md) to select individual relevant entries by touched path/concept. Search curated memory only when it may save investigation; before reliance, `get` the record and re-read its evidence. Raw `trellis mem` output is only a lead.

#### 1.3 Configure selected context `[required · once for sub-agent platforms]`

Curate `implement.jsonl` and `check.jsonl` with repo-relative `{"file":"...","reason":"..."}` rows for relevant specs, task research, and selected known errors. Skip seed rows without `file`. Do not list source files merely because they will be edited, and do not inject all known errors.

- implement context: contracts/evidence needed to make the change;
- check context: quality/semantic contracts and evidence needed to review it.

Inline platforms load the same evidence directly before editing and may skip JSONL curation.

#### 1.4 Final planning review and activate `[required · once]`

Present final scope, artifacts, assumptions/open decisions, selected evidence, verification plan, and protected/experiment gates. Ask the human for implementation approval. Only after approval run:

```bash
python3 ./.trellis/scripts/task.py start <task>
```

For experiments, activation approves implementation/preparation only. The exact launch contract receives a later separate approval immediately before execution.

#### 1.5 Completion criteria

- `prd.md` exists and matches current human requirements;
- complex work has reviewed `design.md` and `implement.md`;
- relevant specs/source/known errors and any relied-on memory evidence are selected;
- sub-agent platforms have curated implement/check JSONL rows;
- the human approved implementation and task status is `in_progress`;
- experiment tasks have `meta.kind=experiment`, a complete contract, and no launch yet.

## Phase 2: Execute

Goal: implement only the reviewed scope and build current verification evidence.

#### 2.1 Implement `[required · repeatable]`

The main session normally dispatches `trellis-implement` with a prompt whose first line is `Active task: <path>`. The implement agent loads JSONL entries, then task artifacts, reads adjacent source/tests, edits directly, and runs focused checks. It does not spawn another implement/check agent.

Before each edit:

- verify branch/worktree and complete dirty state;
- read every affected layer's index Pre-Development Checklist and owning module docs;
- preserve unrelated changes and protected content;
- if source contradicts planning or scope expands, stop and return to Phase 1.

After a verifiable subtask, critical fix, important design decision, experiment stage, rule change, or invalidated conclusion, apply `.agents/skills/on-progress/` immediately. Current details remain in the Trellis task; add only a concise branch milestone when warranted. Do not create new `ai_tasks/ai_progress/` files.

For experiment launch, apply `on-experiment-start`, present the exact final contract/resources/command, and obtain explicit human approval. Monitor through healthy start. Any end state applies `on-experiment-end` in the current conversation.

#### 2.2 Quality check `[required · repeatable]`

The main session normally dispatches `trellis-check`; an implement sub-agent reports that need instead of spawning it. Review and fix:

- task PRD/design/plan compliance and scope;
- each affected spec index Quality Check;
- selected known errors and final-diff concept search;
- focused tests plus affected cross-module/full-scope checks;
- config/JSON/TOML/YAML parsing, Python/TypeScript/shell syntax as applicable;
- Markdown links, generated-adapter contracts, task/context validation, and `git diff --check` for workflow changes;
- protected files, memory hashes, product/experiment/output boundaries, and unrecognized dirty paths.

Report exact commands and results. Missing dependencies, unavailable hardware, or an unrun platform reload/probe remain explicit unverified items.

#### 2.3 Roll back or re-plan `[on demand]`

- requirement/design defect or new authorization need → update artifacts, ask for review, then reactivate execution;
- implementation defect → revert only this task's changes, preserving unrelated work, then reimplement;
- missing evidence → research read-only and persist findings;
- unsafe/ambiguous condition → stop and ask.

## Phase 3: Finish

Goal: make the full result reviewable, preserve useful knowledge without duplication, and separate work commits from bookkeeping.

#### 3.2 Debug retrospective `[on demand]`

If the same issue required repeated fixes, classify the root cause and why earlier attempts failed. Add a known error only for a confirmed occurred failure. Promote a stable mandatory prevention rule into the owning spec; do not use memory or known errors as task logs.

#### 3.3 Progress, memory, and spec review `[required · once]`

- Complete `implement.md`/task checklist and record verification evidence/open risks.
- Add or update one concise `AI_branch_progress.md` milestone when this task changes branch-level state.
- Evaluate every memory used: `get`, re-read evidence, then upvote only if correct and genuinely helpful. Correct wrong entries via the skill. Never run human-only approval.
- Add new memory only for a compact reusable lesson not already clear in specs/docs.
- Update specs for stable cross-task rules or newly established contracts. Module-local behavior stays in module README.

#### 3.4 Complete-diff review and work commits `[required · once]`

Run the final full-scope checks and inspect:

```bash
git status --porcelain
git diff --stat
git diff --check
git log --oneline -5
```

Present once:

- every changed file grouped by purpose;
- semantic mapping to task acceptance criteria;
- exact validation evidence and unverified items;
- intentional generated/Trellis update conflicts;
- unrecognized dirty files excluded from work commits;
- proposed logical work commit groups/messages.

Ask for one-shot human commit approval. If approved, commit only the listed work groups; do not amend, push, merge, or mix archive/journal bookkeeping into them. If rejected or the human chooses manual commits, stop committing and follow that decision. Platform/task role restrictions may forbid commit entirely; the higher-priority restriction wins.

#### 3.5 Finish-work review and bookkeeping `[required · once]`

After work commits are complete and the worktree is otherwise in the reviewed state, present what `/trellis:finish-work` will archive and journal, including its automatic bookkeeping commits (`session_auto_commit: true`). Obtain human acceptance before invoking finish-work. Archive and journal commits occur after work commits, never before the complete-diff review.

## Platform consistency and upgrade boundary

- Pi: `.pi/extensions/trellis`, prompts, and agents. The extension uses callback/session `ctx.cwd` as active root and includes root in context caches; `process.cwd()` is bootstrap fallback only. Run `/reload` after adapter changes before a live sub-agent probe.
- Claude Code: `.claude/hooks`, commands, agents, and `.claude/skills -> ../.agents/skills`.
- Codex: `.codex/hooks`, agents, config, and shared `.agents/skills`; global native-hook enablement/approval is a user machine action.
- Repository-owned Nimloth rules live in `.trellis/spec/` and non-`trellis-*` project skills. Do not put private rules into bundled upstream `trellis-*` skills.
- `.trellis/workflow.md` and the Pi root adapter intentionally diverge from generated defaults. Review every `trellis update --dry-run` conflict; never hand-edit `.trellis/.template-hashes.json` or runtime session state.
