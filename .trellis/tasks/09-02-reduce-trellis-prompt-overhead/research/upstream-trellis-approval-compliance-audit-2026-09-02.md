# Upstream Trellis approval/compliance audit

Date: 2026-09-02
Scope: read-only comparison of the published local Trellis CLI/templates, Nimloth effective prompts, task runtime, and relevant git history. No upstream package or canonical pi-app source was modified during this audit.

## Evidence baseline

- Project `.trellis/.version`: `0.6.15`.
- Installed published CLI: `@mindfoldhq/trellis 0.6.16`.
- `trellis update --dry-run`: project update `0.6.15 -> 0.6.16`; `.pi/agents/trellis-{implement,check}.md` would auto-update; `.pi/extensions/trellis/index.ts`, `.trellis/workflow.md`, `AGENTS.md`, and `.trellis/scripts/task.py` are local conflicts requiring a decision.
- Published workflow: `/home/user/.local/share/npm/lib/node_modules/@mindfoldhq/trellis/dist/templates/trellis/workflow.md`.
- Published planning contract: `/home/user/.local/share/npm/lib/node_modules/@mindfoldhq/trellis/dist/templates/common/skills/brainstorm.md`.
- Published Pi extension: `/home/user/.local/share/npm/lib/node_modules/@mindfoldhq/trellis/dist/templates/pi/extensions/trellis/index.ts.txt`.
- Local workflow and platform integration: `.trellis/workflow.md`, `.pi/extensions/trellis/index.ts`, `.pi/agents/`, `.pi/prompts/`, `AGENTS.md`.

## What upstream Trellis calls approval

Upstream 0.6.16 does not define a generic typed-approval runtime, approval receipt schema, or approval-kind enum. Its published Pi extension registers only `trellis_subagent`; it has no `trellis_approval` or `trellis_work_item` tool. Published `task.py` has no request/record/validate-approval actions.

Upstream approval is a conversational workflow gate at specific phase boundaries:

1. **Task creation consent** before `task.py create`.
2. **Final planning review** after requirement convergence and the final PRD pass. The AI presents a summary, stops, and only a subsequent explicit user message authorizes `task.py start` and implementation.
3. **Repeat final review only after material planning changes**, not after every byte-level artifact change.
4. **One-shot commit-plan confirmation** in Phase 3.4 after showing logical commit groups/messages/file lists and unrecognized dirty files. Upstream explicitly does not push in this step.
5. Optional one-shot confirmation for archiving extra completed tasks during finish-work.

The initial request to “build/fix/go ahead” is explicitly not implementation approval in the published brainstorm contract.

## What the human is expected to review

Before implementation, the final planning summary must include:

- Goal;
- In Scope;
- Out of Scope;
- Acceptance Criteria;
- Key Decisions;
- relevant Risks / Deferred Items;
- artifact status.

The authoritative artifacts are `prd.md`, and for complex tasks also `design.md` and `implement.md`. The user approves the latest final summary after those artifacts converge; Trellis does not require a generic approval popup taxonomy.

Before commit, the human is shown the proposed commit messages/groups and exact file lists, with unrecognized dirty files separated. This is different from approving hashes of `prd.md`/`design.md`/`implement.md`.

## Local mechanisms that are not upstream

Commit `4d1c845c` added the following local system:

- `trellis_work_item` and `.trellis/.runtime/execution/`;
- dashboard-v1, heartbeat and assignment projection;
- `trellis_approval`;
- approval kinds `planning | implementation | experiment_launch | commit | push_merge`;
- hash-bound requests/receipts and pi-app approval transport.

The parent task explicitly requested those five categories, so the implementation followed that reviewed local task. They nevertheless remain a Nimloth/pi-app customization, not upstream Trellis behavior.

Current request count demonstrates that the local typed system became a parallel workflow: 126 requests total — implementation 23, commit 37, push_merge 33, experiment_launch 32, planning 1. The largest task generated 59 requests. This is not the upstream phase-gate shape.

## Compliance findings

### Non-compliant / internally inconsistent

1. **Typed approval is presented as generic Trellis.** Upstream has no such tool or taxonomy.
2. **Any artifact byte change invalidates approval.** Local `validate_approval_receipt()` hashes all of `prd.md`, `design.md`, and `implement.md`; checking a completed plan item changes `implement.md` and invalidates the receipt even when scope did not materially change. This conflicts with upstream’s material-change rule and with Nimloth’s own same-scope authorization-reuse language.
3. **Implementation receipt is not a `task.py start` gate.** Local `cmd_start()` never validates a receipt. The typed runtime is advisory/parallel state, not the lifecycle authority claimed by the UI design.
4. **Commit approval binds the wrong primary evidence.** It binds planning-artifact hashes and free-form strings, not the exact Phase 3.4 diff/commit plan/unrecognized dirty set required upstream.
5. **`push_merge` and `experiment_launch` are project policy, not upstream workflow categories.** They may be valid Nimloth high-risk gates but must not be hard-coded as generic Trellis behavior.
6. **Recent W-014/W-015 execution skipped the required renewed final review.** Both changed public/schema/workflow behavior after task artifacts changed, so they were outside Fast path. Treating the direct “fix this” message as immediate implementation approval did not follow the effective Phase 1.4/material-change contract.
7. **Child context optimization weakens the upstream context contract.** The generated implement/check agents require every JSONL-selected spec/research file to be read. The local child context says to read only entries needed, creating conflicting instructions.
8. **Version skew remains.** The project is on 0.6.15 while the published 0.6.16 Pi implement/check templates remove the old self-loading prelude because the extension already injects context. The local workaround was built around the 0.6.15 duplication instead of first rebasing on 0.6.16.

### Valid local customization, if clearly separated

- Work-item IDs, assignment/heartbeat projection and dashboard read model are useful for the explicitly requested pi-app visibility feature. They are optional observer/runtime extensions and must not redefine Trellis lifecycle or completion authority.
- Compact Main task locator and bounded deltas address a measured Pi provider-context problem. This is generic Pi-integration work that should be rebased onto 0.6.16 and ideally proposed upstream; it should not carry Nimloth approval policy.
- Nimloth experiment launch, protected-data, remote/destructive and push/merge gates belong in `AGENTS.md`, `.trellis/spec/experiments/`, governance specs and project skills.

## Effective prompt assessment

The current project prompt is **not fully upstream-compliant**.

- The generated Pi prompts/agents retained the project’s template versions, but 0.6.16 has newer implement/check templates.
- `.trellis/workflow.md` is intentionally customized, which Trellis permits, but it is a broad replacement rather than a narrow project overlay.
- Higher-priority `AGENTS.md`, the local workflow, the upstream-derived brainstorm skill, and `trellis_approval` tool guidance disagree on whether an initial/direct prompt can satisfy implementation approval and when renewed review is needed.
- The effective prompt still states the correct Phase 1.4 rule in several places, but recent execution demonstrated that the direct-prompt/reuse wording can override it in practice.

## Recommended correction direction (not yet implemented)

1. Treat upstream 0.6.16 workflow/brainstorm/agents as the generic baseline.
2. Preserve project policy only as a narrow Nimloth overlay in `AGENTS.md`, specs and project-specific skills.
3. Remove or redesign generic typed approval so it mirrors upstream phase gates rather than inventing five universal kinds.
4. Restore upstream final-summary review and one-shot commit-plan confirmation semantics.
5. Keep optional work-item/dashboard visibility separate from authorization.
6. Rebase the minimal context-overhead patch onto 0.6.16 and restore the mandatory selected-context read contract.
7. Re-run `trellis update --dry-run` and exact prompt/integration tests before changing active runtime behavior.
