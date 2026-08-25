# E0132: cross-run task selection must not reuse a transport-bound sample ID

## Incident

ID185 visualization Job `524894` attempted to select formal-evaluation Base seed1 by its old `rollout_sample_id`. The new run replaced the validation environment URL with its current dynamically allocated server URL. `AgenticDataset` correctly includes the complete environment config in the canonical sample-identity hash, so no row matched the old ID and validation failed before generation.

## Rule

- `rollout_sample_id` remains the sticky episode identity inside one concrete dataset/runtime contract.
- When intentionally replaying the same semantic navigation task under a new transport URL, select the source row by explicit `data_source + seed` (and verify uniqueness), then persist the newly derived `rollout_sample_id`.
- Do not rewrite dataset config or force the historical identity onto the new row.
- A cross-run visualization is a new rollout, not the original stochastic trajectory, even when checkpoint, task asset, seed, sampling parameters, and policy contract match.
