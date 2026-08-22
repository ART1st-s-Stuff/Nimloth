# E0137 — filtered validation manifest counts must change together

## Error

ID189 retry1 Job `527282` filtered validation from five 60-row sources to Base60+Common60 and correctly expected 120 rows, but a copied uniqueness assertion still required 300 rollout IDs. The job failed before model load.

## Cause

Only the total-row and per-source assertions were updated; the independent unique-ID cardinality assertion was missed.

## Correct practice

When filtering evaluation sources, derive or update total rows, per-source counts, seed coverage, and unique semantic/transport identity counts together. Regression tests must reject stale cardinalities from the source experiment.

## Evidence

- `experiments/training/rl/run_vagen_k4_id189_source20_base_common120.sh`
- `tests/training/rl/test_id189_source20_base_common120.py`
- retry1 output `progress.md` for Job `527282`.
