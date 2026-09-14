# Research: weighted convergence resume

- Query: Minimal existing-entry support for epoch12 continuation with a newly authorized weighted convergence monitor.
- Scope: internal, read-only source inspection in `.worktree/fix-sft1-termination-retrain`
- Date: 2026-09-12

## Findings

All source paths below are relative to that worktree.

### Files and current behavior
- `src/nimloth/training/sft/stage1/trainer.py:493`: hardcoded monitor selection, format uses unweighted LM; query uses total loss.
- `trainer.py:717`: monitor and policy are checkpoint identity fields. `checkpoint.py:92` permits only known loss-weight differences, so merely switching the monitor fails identity validation.
- `trainer.py:1175`: restores convergence; `:1184` restores best; `:1224` requires last_epoch=start_epoch-1. `:1226` repopulates infinite best from the legacy CSV (unweighted!), which must not run during monitor migration.
- `trainer.py:1235`: optimizer and scheduler restoration is separate; rank RNG/data cursor restoration must remain untouched.
- `trainer.py:1276`: already-converged checkpoint exits training before an update. Blind --resume cannot continue epoch12.
- `trainer.py:1445,1480,1487,1545`: selected metric must consistently drive observe, best scalar and best checkpoint publication, while both original metrics stay separately logged.
- `convergence.py:30`: state enforces consecutive absolute epochs and requires previous/best for any nonzero last_epoch. An empty reset at epoch12 is currently invalid on later checkpoint restore.
- `workflow.py:124`: strict override identity; authorized monitor transition must be accepted here as well, without relaxing config/data/model identity.
- `stage1/README.md:46-50,90`: explicit current contract: unweighted convergence, weighted comparison only; successful convergence creates CONVERGED.json.

### Minimal coherent design, conditional on human decision
1. Keep existing train.py/workflow --resume; add a narrow monitor option on the existing CLI/config if authorized (an option is not another executable entrypoint). Keep Stage2 behavior unchanged.
2. Implement one narrowly typed transition: legacy/current Stage1 validation_lm_loss -> validation_weighted_lm_loss, warning and recording source epoch/step, old/new policy, and resetting convergence/best selection only. Same-monitor resumes must faithfully restore history, including later restarts of this continuation. Do not add blanket identity-key stripping or a generic guard bypass.
3. Prefer reseeding weighted baseline from epoch12 validation using the resumed model (or verified matching persisted weighted metric 0.45263 with full precision), set last_epoch=12, previous_loss=best_loss=baseline, bad_epochs=0, converged=False, best_val=baseline. This satisfies existing state schema and means epoch13 compares to epoch12. Alternative fresh-baseline epoch13 requires explicit state schema support for nonzero last_epoch with no observation; simply assigning ConvergenceState(last_epoch=12) fails deserialization. Parent must resolve which baseline semantics are intended.
4. Do not read old unweighted CSV into weighted best. Do not overwrite historical checkpoint bytes or manually patch training_state. Ensure stale CONVERGED.json no longer advertises current continuation completion; preserve old terminal evidence with provenance through approved artifact handling.
5. Keep optimizer/scheduler/RNG/step/data identity checks unchanged across both epoch and step checkpoint paths. Include resolved monitor in new checkpoints, validation logs and terminal summary; workflow history must show the authorized transition.
6. If sampling format rate is also a stopping requirement, threshold and patience semantics remain unspecified. Weighted plateau alone cannot establish format quality. Do not silently invent a threshold or redefine best checkpoint selection.

### Tests
Extend `tests/training/sft1/test_convergence.py`, `test_config.py`, `tests/training/sft/test_preempt_resume.py`, `test_protocol_loss.py`; retain `tests/training/sft/stage2/test_convergence_metrics.py` as nonregression. Test old unweighted converged epoch12 -> epoch13 update with preserved optimizer/scheduler/RNG; known transition warning; unknown monitor and unrelated identity mismatch rejection; same-monitor resume does not reset repeatedly; weighted and unweighted loss moving opposite directions selects weighted best; checkpoint reload after reset; no CSV contamination; correctly scoped terminal marker. CPU checks are not real FSDP save/resume evidence.

### Related specs and approval boundaries
Read `.trellis/workflow.md`, `.trellis/spec/governance/authority-and-safety.md`, `.trellis/spec/experiments/task-contract.md`. Governance requires full proposed spec diff shown before submission, and unresolved stopping-policy decisions require the human decision. Existing user approval for weight16 did not approve switching convergence monitor. Preserve pseudocode style if touching `src/nimloth/training/sft/spec.md`; operational convergence is currently documented in stage1 README. No external references needed for this source-local question.

## Caveats / Not Found
No implementation or remote inspection performed. Pending clarification from main session controls stopping rule. Source may change concurrently. A plain resume presently exits because epoch12 is converged; report this concretely rather than claiming training started.
