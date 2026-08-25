# Migrate Nimloth AI Workflow Into Trellis

## Goal

Migrate Nimloth's repository-owned AI workflow into a Trellis-primary workflow for Pi, Claude Code, and Codex without weakening project rules, losing durable knowledge, or creating conflicting authorities.

## User Value

Future AI sessions should enter through one clear workflow, receive the right project rules and task context automatically, preserve cross-session progress, and obey Nimloth's experiment and human-approval boundaries consistently across all three platforms.

## Background

- Trellis 0.6.15 is initialized in single-repository mode for Pi, Claude Code, and Codex.
- The existing workflow is mature:
  - `AGENTS.md` defines instruction priority, hard CoT/state rules, language rules, and worktree requirements.
  - `ai_rules/` defines honesty/uncertainty, memory/progress, experiments/data, code/repository behavior, event hooks, and known errors.
  - `AI_branch_progress.md`, `ai_tasks/ai_progress/`, and `AI_issues.md` hold progress and decisions.
  - `.memory/`, `.local/memory/`, and the `memory` skill implement human-reviewed durable memory.
  - project-local skills implement worktree, Slurm, experiment start/end, and on-progress behavior.
- `AGENTS.md` is human-authored and may only be changed with explicit human consent; that consent has been given for this reviewed migration scope.
- Generated Trellis frontend/backend specs and generic guides do not describe this Python ML repository.
- The generated bootstrap task was reset from `in_progress` to `planning` so broad edits require approval of the final plan below.

## Requirements

1. Preserve every existing hard safety, honesty, human-approval, experiment, CoT/state, data, Git, and file-protection rule unless explicitly changed by the human.
2. Make Trellis the primary entry for new tasks, specs, and session flow through a staged migration.
3. Give Pi, Claude Code, and Codex equivalent access to project workflow, specs, project-local skills, task state, and verification requirements.
4. Replace the fullstack spec templates with a source-backed Nimloth taxonomy: `governance/`, `experiments/`, `python/`, `domains/`, and cross-layer `guides/`.
5. Keep detailed module ownership in existing `src/nimloth/**/README.md` files; Trellis specs should index them and own cross-module contracts rather than duplicate each package.
6. For new work, store requirements, design, execution state, and checks in Trellis tasks. Keep `AI_branch_progress.md` as a concise migration-period milestone summary and stop creating new `ai_tasks/ai_progress/` files.
7. Retain existing `ai_tasks/` and `AI_issues.md` content as historical evidence; do not rewrite history for style.
8. Retain `.memory/`, `.local/memory/`, and the project `memory` skill as Trellis's specialized human-reviewed knowledge subsystem.
9. Keep `ai_rules/known_errors/` live, create a categorized index, and load only task-relevant entries through Trellis context.
10. Give experiments a dedicated Trellis contract covering purpose, entry point, full parameter names, data/split evidence, checkpoint ownership, train/freeze boundaries, output, resume, monitoring, resource estimate, explicit launch approval, and mandatory end recording.
11. Keep Trellis's default automatic bookkeeping commits for workspace journals and task archives, but only after the workflow has presented scope and validation at the finish-work review gate.
12. Use customized native Trellis phases. Require tasks for multi-file/ambiguous work, project-rule changes, experiments/remote jobs, and long tasks; allow one-reply explanations and clearly bounded small edits to remain inline after task consent is declined.
13. Keep upstream-generated Trellis assets separate from project-local specs and skills so `trellis update` cannot silently replace Nimloth conventions. Convert the existing `git-worktree` and `slurm` symlinks into repository-owned entity skills so all clones and all three platforms receive the same workflow; keep machine-specific details under `.local/`.
14. Make the generated Pi integration resolve project files from Pi's session `ctx.cwd`, because Pi Desktop's NodeService `process.cwd()` points at the desktop app rather than the active worktree.
15. Use Trellis as the only development task system. Keep Pi TaskTree empty and do not duplicate Trellis task status, priority, hierarchy, focus, or acceptance criteria there.
16. Do not modify product code, launch experiments, alter datasets, touch checkpoints, or submit remote jobs in this task.
17. Make all changes reviewable, reversible, and validated before commit.

## Key Decisions

- **Authority:** staged Trellis-primary.
- **Progress:** Trellis task detail plus concise `AI_branch_progress.md` milestones; no new legacy progress files.
- **Memory:** preserve the evidence-backed, human-approved memory subsystem; Trellis workspace is a session journal and `trellis mem` is raw dialogue recall.
- **AGENTS:** update now as a concise bootstrap that retains a direct safety kernel.
- **Spec shape:** four project layers plus guides; no module-by-module duplication.
- **Legacy rules:** after lossless migration and link repair, archive core `ai_rules/01-04` and `ai_rules/events/` immediately; keep known errors live.
- **Commits:** retain Trellis automatic journal/archive bookkeeping commits after review.
- **Task threshold:** complexity/risk based.
- **Experiments:** dedicated task and lifecycle contract.
- **Known errors:** categorized index plus selective task loading.
- **Workflow template:** customize native, not global TDD or channel-driven.
- **Task system:** Trellis only; Pi TaskTree remains unused and empty.
- **Shared skills:** `git-worktree` and `slurm` become repository-owned `.agents/skills/` directories; `.local/` remains machine-specific.

## Acceptance Criteria

- [ ] `AGENTS.md` is a concise cross-platform Trellis bootstrap and retains instruction priority, honesty/uncertainty, authorization, CoT/state, protected-file, and main-branch safety rules directly.
- [ ] `.trellis/spec/` contains only Nimloth-relevant, source-backed indexes and guidance with no generated fullstack placeholders.
- [ ] `.trellis/workflow.md` implements the approved task threshold, planning approval, experiment gates, relevant known-error selection, memory routing, progress handling, checking, and finish review.
- [ ] Project-local worktree, Slurm, progress, experiment, and memory skills point to live Trellis authority paths and work from the shared `.agents/skills/` layer.
- [ ] Pi, Claude Code, and Codex context-loading paths are documented and validated; Pi Desktop can resolve and launch project-local Trellis agents after `/reload`.
- [ ] Project guidance names Trellis as the sole development task authority and does not require TaskTree writes.
- [ ] Core legacy rules/events are archived only after every semantic requirement has a mapped live destination and active links are updated.
- [ ] `ai_rules/known_errors/` remains live with a usable categorized index and selective-loading instructions.
- [ ] Legacy task/progress/issue history is preserved and clearly marked as historical where needed.
- [ ] Existing memory records keep their original verification levels and are never rewritten directly.
- [ ] Trellis task/context validation, config parsing, Python hook compilation, link checks, placeholder scans, platform detection, and `git diff --check` pass.
- [ ] No product source, experiment output, dataset, checkpoint, or remote job changes.
- [ ] The complete diff and validation evidence are presented before any work commit.

## Out of Scope

- Refactoring Nimloth model or training code.
- Running training, evaluation, collection, calibration, rollout, or GPU gates.
- Rewriting historical progress, issues, or known-error entries for style.
- Changing established CoT/state semantics or experiment ownership.
- Reimplementing curated memory inside Trellis.
- Contributing to the upstream Trellis npm package.
