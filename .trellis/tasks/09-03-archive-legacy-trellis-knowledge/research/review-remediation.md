# Legacy pre-move review remediation

## Conservative disposition reconciliation

`known-error-audit.jsonl` now requires `contract_mappings` for every `retain-contract` row. Each mapping names one owning spec, one unique section anchor, and one exact excerpt inside that section. The validator checks the source evidence files and the exact spec text. Records with implementation-local, dependency-version, cluster/machine, numeric-timeout, or otherwise non-general behavior were changed to `archive-only`; no new technical rule was added merely to preserve an incident. E0066, E0068, E0072, and E0147 remain `needs-human-decision` with no target spec or mapping.

Current reconciliation: 33 `retain-contract`, 117 `archive-only`, 2 `obsolete`, and 4 `needs-human-decision`.

## Exact destructive operation (not executed)

The reviewed operation is bound by two SHA-256 fingerprints:

- `research/legacy-move-manifest.jsonl`: 254 source-to-destination moves with source SHA-256 and Git blob IDs.
- `research/approved-rewrite-patch.json`: 18 whole-file pre/post hashes and 28 exact replacements: 19 active-task context entries, all four deferred eval-test contract paths, and five SFT1 Slurm purpose lines.

The five Slurm files intentionally remain at their valid pre-move text until execution. Their reviewed post-move lines point to `ai_tasks/archive/pre-trellis/sft1_exp.md` and retain `no actor/critic update`.

## Execute and rollback

`tools/propose-legacy-archive-moves.sh` remains dry-run by default. The exact `--execute-approved-moves` mode validates the pre-state, rejects symlink/non-directory destination ancestors and path escape, requires every destination absent, creates and validates every destination directory before the first move, and rechecks source bytes/blob plus destination ancestry/absence before each move. It then applies the hash-bound rewrite patch and runs post-move source/destination hash/blob, active-reference, active-context, Markdown-link, and focused eval-test checks.

`tools/rollback-legacy-archive.py` is also dry-run by default. It classifies every manifest row as untouched or moved, rejects ambiguous/changed states, reverses only reviewed post-hash rewrites, restores moved rows in reverse order, and verifies the pre-move plan. Its mutation flag is `--execute-approved-rollback`; it has not been run.
