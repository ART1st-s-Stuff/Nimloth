# Investigation and Uncertainty

## Evidence-first sequence

1. Restate the exact requested outcome, authorization, exclusions, and acceptance criteria in the task.
2. Inspect current branch/worktree and complete dirty state.
3. Read owning specs, module READMEs, adjacent source/tests/config, and task-relevant known errors.
4. Trace the real control/data/state/gradient/checkpoint path. Names, resolved config, docs, and mocks are leads, not proof.
5. For data or experiments, inspect actual asset/config/code/metadata and preserve provenance.
6. Identify facts, assumptions, competing designs, verification gaps, and protected actions.
7. Resolve by local read-only research when possible; otherwise stop and ask one clear human question.

## Mandatory stop conditions

Stop when requirements or semantics are unclear; several materially different designs remain; sources/rules conflict; the needed change is broad, destructive, protected, unexpected, or out of scope; exact implementation cannot be verified; or an experiment field/approval is missing.

Do not keep working by selecting a plausible default, adding a temporary stand-in, weakening validation, or silently broadening scope.

## Verification framing

Before editing, state which evidence would prove the outcome and which checks are available locally. After editing, report:

- completed and verified;
- completed but unverified and why;
- incomplete;
- risks/assumptions;
- human decisions required.

For ML work, distinguish static/schema/interface/CPU checks from real GPU optimizer, rollout, resume, or model-quality evidence. The lower level never implies the higher one.
