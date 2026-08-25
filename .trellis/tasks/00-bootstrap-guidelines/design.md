# Design: Trellis-Primary Nimloth AI Workflow

## 1. Architecture

The migrated system has five cooperating layers:

1. **Safety bootstrap — `AGENTS.md`**
   - Loaded by platform conventions even when Trellis hooks fail.
   - Retains project identity, instruction priority, stop-on-uncertainty/authorization boundaries, CoT/state red lines, protected-file and main-branch rules.
   - Routes detailed behavior to `.trellis/workflow.md` and `.trellis/spec/`.

2. **Workflow — `.trellis/workflow.md`**
   - Primary lifecycle for request triage, planning, execution, checking, experiment gates, progress/memory review, commit review, and finish-work.
   - Keeps native Trellis status tags and parser contracts intact.
   - Adds Nimloth-specific required actions to both formal phase text and matching workflow-state breadcrumbs.

3. **Project contracts — `.trellis/spec/`**
   - Owns mandatory cross-module rules.
   - Uses indexes as routing surfaces for task JSONL manifests.
   - Links module-local architecture to existing `src/nimloth/**/README.md` files.

4. **Capabilities — `.agents/skills/`**
   - Existing project-local skills remain the operational layer for worktrees, Slurm, experiment start/end, progress, and curated memory.
   - `git-worktree` and `slurm` are converted from main-worktree symlinks into repository-owned entity directories; portable behavior is versioned, while machine/server details remain under the shared ignored `.local/` directory.
   - Skills reference live Trellis specs instead of archived legacy rules.
   - `.claude/skills -> ../.agents/skills` keeps one shared project-local source for Claude Code, Codex, and Pi.
   - Bundled `trellis-*` skills are not customized with private rules.

5. **Persistence**
   - `.trellis/tasks/`: requirements, designs, plans, research, and checks for new work.
   - `.trellis/workspace/`: per-session activity summaries.
   - `trellis mem`: read-only raw dialogue recall.
   - `.memory/` and `.local/memory/`: evidence-backed, human-reviewed durable lessons.
   - `AI_branch_progress.md`: concise migration-period milestones.
   - existing `ai_tasks/` and `AI_issues.md`: retained historical evidence, no new detailed work.
   - Pi TaskTree: deliberately unused; it must remain empty because its status, priority, hierarchy, focus, and acceptance fields overlap Trellis task ownership.

## 2. Spec Tree

```text
.trellis/spec/
├── governance/
│   ├── index.md
│   ├── authority-and-safety.md
│   ├── cot-and-state.md
│   ├── tasks-progress-and-memory.md
│   ├── git-worktrees-and-protected-files.md
│   └── platform-integration.md
├── experiments/
│   ├── index.md
│   ├── task-contract.md
│   ├── data-and-splits.md
│   ├── launch-and-lifecycle.md
│   └── outputs-checkpoints-and-evidence.md
├── python/
│   ├── index.md
│   ├── structure-and-module-indexes.md
│   ├── configuration-and-interfaces.md
│   └── quality-and-testing.md
├── domains/
│   ├── index.md
│   ├── terminology-and-ownership.md
│   ├── agent-rollout-and-state.md
│   ├── world-model-and-training.md
│   └── reconstruction-and-evaluation.md
└── guides/
    ├── index.md
    ├── investigation-and-uncertainty.md
    └── known-error-routing.md
```

Every layer index contains:

- applicability and authoritative boundaries;
- pre-development checklist;
- quality-check checklist;
- links to topic specs and source-backed module docs.

Generated `frontend/`, `backend/`, and generic Trellis-maintainer guide content is deleted.

## 3. Authority and Legacy Migration

### Live authority after migration

1. current human prompt;
2. `AGENTS.md` safety bootstrap;
3. `.trellis/workflow.md` for lifecycle;
4. task artifacts for approved task scope;
5. `.trellis/spec/` for detailed project contracts;
6. current source/config/docs and relevant known errors;
7. verified curated memory after evidence recheck;
8. raw conversation/tool-private memory last.

A task artifact cannot override the human prompt or safety/spec hard rules unless the human explicitly approves a rule change.

### Archive layout

```text
ai_rules/
├── archive/
│   └── pre-trellis/
│       ├── README.md
│       ├── 01_honesty_and_uncertainty.md
│       ├── 02_memory_and_progress.md
│       ├── 03_experiments_and_data.md
│       ├── 04_code_and_repo.md
│       └── events/
└── known_errors/
```

The archive README states that files are historical snapshots and links to their live Trellis destinations. Archive occurs only after active links and project-local skills have been updated.

`ai_rules/known_errors/` remains live. Its index classifies entries by agent/governance, data/rollout, model/state, training/checkpoint, distributed/runtime, Slurm/experiment, and evaluation/reporting themes. Task planning/checking selects relevant individual entries; it never injects the whole library.

## 4. Task and Progress Contract

### Task threshold

Mandatory Trellis task:

- multi-file or ambiguous implementation;
- project-rule/workflow change;
- experiment, GPU, Slurm, remote long job, collection, evaluation, or rollout-train;
- work requiring durable design, handoff, or multi-session progress.

Optional after explicit consent is declined:

- one-reply explanation;
- read-only lookup with no durable decision;
- clearly bounded small edit with obvious acceptance and low risk.

### New progress routing

- current details and unresolved decisions: current task PRD/design/notes;
- task execution checklist: `implement.md` and `task.json`;
- session summary: Trellis workspace journal;
- branch-level milestone: concise `AI_branch_progress.md` entry;
- no new `ai_tasks/ai_progress/` files;
- old `ai_tasks/` and `AI_issues.md` remain historical, with a migration notice rather than rewritten content.

### Task-system exclusivity

Trellis is the only live development task authority. Pi TaskTree currently has revision 0 with no tasks and remains empty. Agents must not mirror Trellis tasks into TaskTree or create TaskTree backlog entries unless a future human-approved migration changes this contract.

## 5. Curated Memory Contract

- Specs store mandatory future behavior.
- Tasks store current facts and decisions.
- Workspace stores what happened in a session.
- `trellis mem` searches raw dialogue and is never treated as verified truth.
- Existing memory stores compact reusable lessons with evidence and human approval.
- AI-created memory remains pending; AI never runs human approval commands.
- Before relying on memory, retrieve it, re-read evidence, and upvote only if it genuinely helped.
- Stable mandatory lessons may be promoted into spec after review; do not copy the same prose into both stores.

## 6. Experiment Contract

An experiment task uses `task.json.meta.kind = "experiment"` and a required PRD section containing:

- purpose and falsifiable question;
- exact code entry and command/config;
- full parameter names for ambiguous concepts;
- dataset source and split evidence;
- checkpoint initialization and ownership;
- trainable/frozen modules and each objective;
- output directory and uniqueness check;
- checkpoint/resume strategy;
- metrics and validity gates;
- resource/time estimate;
- explicit user launch approval.

Before launch, the on-experiment-start skill verifies the contract and refuses to infer missing values. After completion, failure, cancellation, or pause, on-experiment-end records scheduler/runtime state, outputs, metrics, provenance, validity limits, memory/spec implications, and the task/branch milestone.

Lifecycle behavior remains project-local skill logic because generic Trellis `after_start`/`after_finish` hooks cannot reliably mean “an experiment process actually started/stopped.”

## 7. Workflow Customization

Keep the native Plan → Execute → Check → Finish structure and status names. Add:

- risk-based task triage;
- explicit “task creation is not implementation approval” gate;
- required spec/known-error/memory evidence selection in planning;
- experiment contract and separate launch approval;
- before-development source/spec checklist;
- final full-scope tests and semantic checks;
- on-progress routing after substantive milestones;
- curated-memory evaluation without forced memory creation;
- one-shot commit review;
- automatic journal/archive bookkeeping only after finish-work review.

Do not introduce custom task statuses in this migration. This avoids changing command route tables and platform prompt semantics.

## 8. Platform Consistency

- Pi: `.pi/extensions/trellis`, prompts, and agents.
- Claude Code: `.claude/settings.json`, hooks, commands, agents, shared skill symlink.
- Codex: `.codex/config.toml`, hooks, agents, and shared `.agents/skills/`.

Pi Desktop runs extension NodeServices from the desktop application's OS cwd rather than the active project. The local Pi adapter therefore treats `ctx.cwd` as authoritative on tool/event callbacks, uses `process.cwd()` only as a bootstrap fallback, and includes the resolved root in context-cache keys. This prevents agent discovery and task context from leaking across project sessions.

Project rules live in shared specs and project-local skills. Platform files remain thin adapters. Codex native hooks still require the user's global `features.hooks = true` and one-time `/hooks` approval; this machine-level action is reported but not silently performed by the repository migration.

## 9. Upgrade Boundary

- Do not edit bundled upstream skill directories to store Nimloth rules.
- Local edits to `.trellis/workflow.md` and generated platform adapters may be flagged by `trellis update`; document intentional divergence.
- `.trellis/.template-hashes.json` and runtime session state are not hand-edited.
- Run `trellis update --dry-run` after migration and review every conflict.

## 10. Rollback

Before commit, rollback is `git restore`/removal limited to files changed by this migration. The archived legacy rules remain in the same commit as their live replacements and link repairs, so reverting that commit restores the old authority atomically. No data-store migration or destructive rewrite of memory/history is performed.
