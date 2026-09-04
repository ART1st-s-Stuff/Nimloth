# Policy overlay validation — 2026-09-04

## Scope

14 files: approved 49-line `AGENTS.md`, seven governance/experiment specs, and five operational skills plus one governance index. No Python source, experiment run, known-error payload, legacy archive, memory JSONL, live runtime, or pi-app file is included.

## Contracts established

- Direct human conversation replaces typed approval receipt governance; task status does not manufacture authorization.
- Trellis remains the only task authority; optional work-item visibility is non-authoritative and cannot block upstream lifecycle.
- Pi-app transports generic questions without approval kinds/receipts/task mutation.
- Every repository-changing task uses a dedicated branch; `dev` is integration/base, canonical has one primary task slot, and parallel tasks use child worktrees.
- All real experiments are remote-only. A scoped reproduction/partial rerun expected within 10 minutes needs no additional launch question and has a 15-minute total deadline from successful submission across pending/running; longer/materially new runs require explicit launch approval, and timeout cancellation routes to defer/blocker.
- Operational details remain in project-local experiment/Slurm/worktree/progress skills.

## Evidence

- Initial `test_nimloth_policy_overlay.py`: 4/4 passed; final-review remediation expands it to 5/5.
- Initial `test_upstream_trellis_baseline.py`: 1/1 passed; final-review remediation expands it to 2/2 and a fixed 144-file manifest.
- Context measurement: main 1750 B, implement 3822 B, check 3766 B; each selected artifact marker appears once.
- `task.py validate`: implement/check manifests valid.
- 54 relative Markdown links across the 14 changed files resolve.
- Stale custom governance search found no validated receipt, typed Trellis approval, dashboard consumer, execution runtime, release-assignment, or heartbeat contract.
- `git diff --check` passed for policy files.
- `trellis update --dry-run` reports `.trellis/config.yaml` and approved project-owned `AGENTS.md`; it also reports regular `.agents/skills/trellis-update-spec/SKILL.md` through the symlinked `.claude/skills` parent alias although its working hash equals HEAD and the recorded template hash (`d975db7a...`). The malformed update output was removed by exact edit; this is an updater false-positive, not retained drift.

The human approved the complete policy diff and local commit `18c3a51a`. A later independent review found four parent-contract mismatches; their follow-up corrections remain uncommitted until repeat validation and review.
