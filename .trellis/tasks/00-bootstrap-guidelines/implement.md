# Implementation Plan

## Phase A — Establish source-backed Trellis specs

- [x] Delete generated `frontend/`, `backend/`, and generic Trellis-maintainer guide templates.
- [x] Create the reviewed governance, experiments, python, domains, and guides tree.
- [x] Populate every index with applicability, pre-development checklist, quality check, and live links.
- [x] Migrate legacy rule semantics losslessly, preserving the fixed-CoT/state, uncertainty, authorization, data split, experiment, protected-file, and Git rules.
- [x] Reference real module READMEs, tests, configs, and experiment docs instead of inventing patterns.

**Gate A**

```bash
rg -n "To be filled|TODO: fill|placeholder|your project" .trellis/spec
python3 .trellis/scripts/get_context.py --mode packages
```

Expected: placeholder search has no matches; package context lists only the reviewed Nimloth layers.

## Phase B — Migrate workflow and project-local capabilities

- [x] Customize `.trellis/workflow.md` without changing native status names or breaking tag pairs.
- [x] Add risk-based task triage, final implementation approval, known-error selection, memory routing, experiment launch/end gates, on-progress behavior, full-scope checks, and finish review.
- [x] Keep default workspace/task-archive auto-commit behavior and make its prior review gate explicit.
- [x] Set project configuration explicitly where behavior should not depend on undocumented defaults.
- [x] Convert `git-worktree` and `slurm` from main-worktree symlinks into repository-owned entity skills; update `.gitignore` while preserving `.local/` as machine-specific shared state.
- [x] Update project-local `memory`, `on-progress`, `on-experiment-start`, `on-experiment-end`, `git-worktree`, `slurm`, and related README references to live Trellis specs.
- [x] Do not edit bundled upstream `trellis-*` skills for project-private behavior.

**Gate B**

```bash
python3 .trellis/scripts/get_context.py --mode phase
python3 .trellis/scripts/task.py validate 00-bootstrap-guidelines
python3 -m compileall -q .trellis/scripts .claude/hooks .codex/hooks
```

## Phase C — Change entry and persistence routing

- [x] Rewrite `AGENTS.md` as the approved safety-kernel bootstrap and Trellis router.
- [x] Update root/project workflow documentation and active experiment README links.
- [x] Add migration notices to historical `ai_tasks/` and `AI_issues.md` surfaces without rewriting history.
- [x] Document new Trellis-task detail plus concise branch-summary policy.
- [x] Preserve `.memory/` and `.local/memory/` files byte-for-byte.

**Gate C**

- Confirm `AGENTS.md` directly retains the approved safety kernel.
- Confirm no live document says legacy progress files remain authoritative for new work.
- Confirm memory JSONL files are unchanged.

## Phase D — Archive legacy rule sources and index known errors

- [x] Create `ai_rules/archive/pre-trellis/README.md` with historical status and live mapping.
- [x] Move legacy core rules 01–04 and event docs into the archive only after live replacements exist.
- [x] Keep `ai_rules/known_errors/` live.
- [x] Build a categorized known-error index covering every `E*.md` file exactly once or through an explicit multi-category mapping.
- [x] Update all active links and project-local skills; allow historical changelog text to describe old paths only when it is clearly historical.

**Gate D**

```bash
rg -n "ai_rules/(0[1-4]|events/)" \
  AGENTS.md README.md experiments src configs tests .agents/skills .trellis \
  --glob '!ai_rules/archive/**'
```

Expected: no active dependency on archived authority paths.

## Phase E — Three-platform and full-scope verification

- [x] Parse generated/project JSON and TOML files.
- [x] Validate Python Trellis/platform hooks.
- [x] Validate task JSONL context and current task resolution.
- [x] Confirm TaskTree remains empty and project guidance does not instruct agents to mirror Trellis tasks into it.
- [x] Validate the Pi extension parses and resolves project state from callback `ctx.cwd` rather than Desktop NodeService `process.cwd()`.
- [x] After `/reload`, dispatch a read-only/safe `trellis-implement` probe and confirm it finds `.pi/agents/trellis-implement.md` in this worktree.
- [x] Run `trellis platforms` and confirm Pi, Claude Code, and Codex.
- [x] Run `trellis update --dry-run`; document intentional local modifications/conflicts.
- [x] Check relative Markdown links in changed live docs.
- [x] Verify spec indexes match actual files and no fullstack template content remains.
- [x] Verify no product source, config values, experiment scripts, datasets, checkpoints, outputs, or remote jobs changed.
- [x] Run `git diff --check`.
- [x] Perform a semantic checklist comparison from each archived rule section to its live spec/workflow/skill destination.

Suggested commands:

```bash
python3 .trellis/scripts/task.py validate 00-bootstrap-guidelines
python3 .trellis/scripts/get_context.py --mode packages
python3 .trellis/scripts/get_context.py --mode phase
python3 -m compileall -q .trellis/scripts .claude/hooks .codex/hooks
trellis platforms
trellis update --dry-run
git diff --check
git status --short
```

## Phase F — Review and commit

- [x] Update the bootstrap task checklist and concise `AI_branch_progress.md` milestone.
- [x] Run `trellis-check` full-scope review (main session only; implement sub-agent recursion guard applies).
- [x] Present all changed files, semantic mapping, validation evidence, intentional Trellis update conflicts, and unrecognized dirty files.
- [ ] Obtain one-shot commit approval before creating work commits.
- [ ] Invoke finish-work only after work commits are complete and the user accepts automatic bookkeeping commits.

## Verification Evidence (implement agent)

Completed on 2026-08-25:

- `task.py validate 00-bootstrap-guidelines`: both manifests valid, 10 entries each after the check agent added the directly relevant fixed-CoT and worktree-mutation known errors.
- `get_context.py --mode packages`: single-repo layers are `domains`, `experiments`, `governance`, `python`; cross-layer guides are present under `.trellis/spec/guides/`.
- `get_context.py --mode phase` and steps 1.4/2.2/3.4: custom phase/index parsing passed; all native workflow tag pairs and required step headings were checked exactly once.
- Trellis config consumers parsed `session_auto_commit=true`, Codex `auto`, explicit context limits, and prompt skip keyword. Generated/project JSON/JSONL and Codex TOML parsed with standard parsers.
- `compileall` passed for `.trellis/scripts`, `.claude/hooks`, and `.codex/hooks`.
- Bun built `.pi/extensions/trellis/index.ts`; a runtime harness launched from a foreign OS cwd and confirmed callback `ctx.cwd` selected the temporary project root. The current delegated implement session also resolved `.pi/agents/trellis-implement.md` and this task from the Nimloth worktree.
- `trellis platforms` reported Pi, Claude Code, and Codex.
- `trellis update --dry-run` made no changes. It reports intentional local divergence in `.trellis/config.yaml`, `.trellis/workflow.md`, `AGENTS.md`, and `.pi/extensions/trellis/index.ts`. It also reports `.agents/skills/trellis-update-spec/SKILL.md`; the implement agent did not edit that upstream skill, and its SHA-256 equals the recorded `.trellis/.template-hashes.json` value.
- Relative Markdown links passed across changed live docs/specs/skills/archive mapping; generated-placeholder/fullstack scans passed.
- The known-error index covers all 97 `E*.md` files exactly once.
- Archived rule/event SHA-256 values equal their pre-move values; repo/local memory SHA-256 values remain `15e423...072` and `6c3362...253`.
- No `.pi/task-tree` path exists in this worktree and no TaskTree operation was performed.
- `git status` shows no `src/`, `configs/`, or `tests/` changes; experiment changes are README-only. No experiment, dataset, checkpoint, output, GPU, Slurm, or remote operation ran.
- `git diff --check` passed.

Independent `trellis-check` review on 2026-08-25:

- Fixed the archive mapping so every live destination is an actual validated relative Markdown link.
- Added task-specific `E0045` and `E0094` entries to both context manifests; task validation now reports 10 entries per manifest.
- Rechecked archived snapshots against their original Git blobs, memory hashes, protected/product/data boundaries, TaskTree absence, known-error coverage, spec-tree shape, live/archive Markdown links, config/JSON/JSONL/TOML parsing, Python compilation, Pi TypeScript build, all three platform context paths, and `git diff --check`.
- A foreign-OS-cwd Pi harness with a fake child CLI confirmed that callback `ctx.cwd` supplies this task and that same-session context caches remain isolated by active root.
- No additional semantic, behavioral, or platform blockers remain from the check review.

Pending main-session gates: present/accept the one-shot work commit plan, create approved work commits, then obtain acceptance before finish-work bookkeeping.

## Guardrails

- No product Python implementation changes.
- No experiment/config parameter changes.
- No experiment, remote, GPU, or Slurm execution.
- No manual edits to memory JSONL, Trellis template hashes, or runtime session pointers.
- No archival move before replacement content and link updates exist in the same working tree.
- No claim of semantic migration until every legacy rule section is mapped and reviewed.
- If a protected or previously unrecognized file requires change beyond this plan, stop and ask.
