# Pre-Trellis Rule Archive

These files are byte-preserved historical snapshots of Nimloth's rule system before the Trellis-primary migration. They are **not live authority**. New work follows `AGENTS.md`, `.trellis/workflow.md`, `.trellis/spec/`, project-owned skills, and task-relevant live `ai_rules/known_errors/`.

Do not update archived wording to match current style. Historical documents may continue to link here as evidence.

## Live mapping

| Archived source/section | Live destination |
|---|---|
| `01` — false implementation, temporary substitute disclosure | [Authority and safety](../../../.trellis/spec/governance/authority-and-safety.md) → Honesty red lines |
| `01` — authorization boundary and mandatory uncertainty stops | [`AGENTS.md`](../../../AGENTS.md); [authority/safety](../../../.trellis/spec/governance/authority-and-safety.md); [investigation/uncertainty](../../../.trellis/spec/guides/investigation-and-uncertainty.md); [workflow](../../../.trellis/workflow.md) planning/rollback gates |
| `01` — reporting categories | [Authority and safety](../../../.trellis/spec/governance/authority-and-safety.md) → Reporting and language |
| `02` — progress stores and long-task records | [Tasks/progress/memory](../../../.trellis/spec/governance/tasks-progress-and-memory.md); [workflow](../../../.trellis/workflow.md) 2.1/3.3; [`on-progress`](../../../.agents/skills/on-progress/SKILL.md) |
| `02` — memory stores, evidence, verification, upvote, human-only approval, stale behavior | [`memory` skill](../../../.agents/skills/memory/SKILL.md); [tasks/progress/memory](../../../.trellis/spec/governance/tasks-progress-and-memory.md) |
| `02` — event triggers | [`on-progress`](../../../.agents/skills/on-progress/SKILL.md), [`on-experiment-start`](../../../.agents/skills/on-experiment-start/SKILL.md), [`on-experiment-end`](../../../.agents/skills/on-experiment-end/SKILL.md); [workflow](../../../.trellis/workflow.md) 2.1 |
| `03` — required pre-launch inputs and stop condition | [Experiment task contract](../../../.trellis/spec/experiments/task-contract.md); [`on-experiment-start`](../../../.agents/skills/on-experiment-start/SKILL.md) |
| `03` — server-only outputs, groups, unique run directories, metadata and logs | [Outputs/checkpoints/evidence](../../../.trellis/spec/experiments/outputs-checkpoints-and-evidence.md); [`experiments/README.md`](../../../experiments/README.md) |
| `03` — dataset split verification and non-overlap | [Data and splits](../../../.trellis/spec/experiments/data-and-splits.md) |
| `03` — checkpoint/resume and committed launch state | [Outputs/checkpoints/evidence](../../../.trellis/spec/experiments/outputs-checkpoints-and-evidence.md); [launch/lifecycle](../../../.trellis/spec/experiments/launch-and-lifecycle.md) |
| `03` — >3 minute/expensive job disclosure and approval | [Launch/lifecycle](../../../.trellis/spec/experiments/launch-and-lifecycle.md); [workflow](../../../.trellis/workflow.md) experiment launch gate |
| `03` — experiment start/end events | [`on-experiment-start`](../../../.agents/skills/on-experiment-start/SKILL.md), [`on-experiment-end`](../../../.agents/skills/on-experiment-end/SKILL.md), and [launch/lifecycle](../../../.trellis/spec/experiments/launch-and-lifecycle.md) |
| `04` — scoped, clear, configurable Python changes and README module indexes | [Python specs](../../../.trellis/spec/python/index.md) |
| `04` — worktree/main safety | [`AGENTS.md`](../../../AGENTS.md); [Git/worktree/protected files](../../../.trellis/spec/governance/git-worktrees-and-protected-files.md); [`git-worktree`](../../../.agents/skills/git-worktree/SKILL.md) |
| `04` — protected files | [`AGENTS.md`](../../../AGENTS.md); [Git/worktree/protected files](../../../.trellis/spec/governance/git-worktrees-and-protected-files.md) |
| `04` — status, dirty changes, review, semi-linear merge | [Git/worktree/protected files](../../../.trellis/spec/governance/git-worktrees-and-protected-files.md); [workflow](../../../.trellis/workflow.md) 3.4 |
| `events/on_progress` — milestone routing and memory evaluation | [`on-progress`](../../../.agents/skills/on-progress/SKILL.md); [tasks/progress/memory](../../../.trellis/spec/governance/tasks-progress-and-memory.md); [workflow](../../../.trellis/workflow.md) 2.1/3.3 |
| `events/on_experiment_start` — memory/source checks, launch contract, W&B naming, Slurm/resource/monitoring | [`on-experiment-start`](../../../.agents/skills/on-experiment-start/SKILL.md); [task contract](../../../.trellis/spec/experiments/task-contract.md); [launch/lifecycle](../../../.trellis/spec/experiments/launch-and-lifecycle.md); [`slurm`](../../../.agents/skills/slurm/SKILL.md) |
| `events/on_experiment_end` — status/provenance/results/resume/progress/memory | [`on-experiment-end`](../../../.agents/skills/on-experiment-end/SKILL.md); [launch/lifecycle](../../../.trellis/spec/experiments/launch-and-lifecycle.md); [outputs/checkpoints/evidence](../../../.trellis/spec/experiments/outputs-checkpoints-and-evidence.md) |

## Migration changes explicitly approved by the human

- Trellis tasks replace new `ai_tasks/ai_progress/` and `AI_issues.md` live writes; historical content remains.
- `AI_branch_progress.md` receives only concise migration-period milestones.
- Trellis is the sole development task authority; Pi TaskTree stays empty.
- Automatic Trellis archive/journal bookkeeping commits remain enabled only after finish-work review.
- Portable `git-worktree` and `slurm` skills are repository-owned directories; machine details remain under `.local/`.
