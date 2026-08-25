# Legacy Workflow Inventory

## Authority and safety

- `AGENTS.md` is the cross-platform entry and declares human prompt → AGENTS → detailed rules → progress → memory priority.
- `AGENTS.md` directly contains the fixed-CoT prohibition, real-CoT state requirement, human-only boundaries, language rules, and worktree rule.
- `ai_rules/01_honesty_and_uncertainty.md` requires stopping on ambiguity, conflicts, broad/destructive changes, or unverifiable semantics.
- `ai_rules/04_code_and_repo.md` defines source layout, protected files, worktree policy, and pre-commit review.

## Experiments and data

- `ai_rules/03_experiments_and_data.md` owns launch inputs, dataset split evidence, unique outputs, resume/checkpoint behavior, monitoring, and expensive-job approval.
- `ai_rules/events/on_experiment_start.md` and `on_experiment_end.md` define mandatory lifecycle actions.
- `.agents/skills/on-experiment-start/`, `on-experiment-end/`, and `slurm/` are the trigger/operation layer.
- `experiments/README.md` and `experiments/training/README.md` route reusable launchers, configs, and output records.

## Progress, tasks, issues, and memory

- `AI_branch_progress.md` is a large branch/stage history and current milestone feed.
- `ai_tasks/ai_progress/` contains long-task live records; top-level `ai_tasks/*.md` contains historical plans and experiment designs.
- `AI_issues.md` stores unresolved human decisions.
- `.memory/memories.jsonl` and `.local/memory/memories.jsonl` are managed only through `./skill memory`; records have evidence, verification level, usage/upvote, and stale behavior.
- `.agents/skills/on-progress/` routes milestone updates and memory review.

## Known errors

- `ai_rules/known_errors/README.md` defines one confirmed failure pattern per file.
- The library contains more than ninety entries spanning experiments, Slurm, distributed training, model/data semantics, and agent behavior.
- Injecting the full directory into every task would consume context and dilute relevant signals; it needs a categorized index and task-specific selection.

## Platform integration

- `.claude/skills` is a symlink to `../.agents/skills`, so project-local shared skills already serve Claude Code, Codex, and Pi from one source.
- Trellis adds platform-specific hooks/agents/prompts under `.claude/`, `.codex/`, and `.pi/`.
- `CLAUDE.md` points to `AGENTS.md` and can remain a compatibility entry.

## Migration implications

1. `AGENTS.md` must remain a safety fallback even after Trellis becomes primary.
2. Detailed rules can move to `.trellis/spec/`, but core/event files must not be archived until every active link and skill is updated.
3. Curated memory is not equivalent to Trellis workspace journals or `trellis mem` raw dialogue search.
4. Existing task/progress history should remain readable; new detailed work can cut over to Trellis tasks.
5. Bundled `trellis-*` skills are upstream-managed; Nimloth rules belong in specs or existing/new project-local skills.
