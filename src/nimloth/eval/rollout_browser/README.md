# Rollout browser

This package converts behavior-time evidence from rollout-based evaluations into immutable offline HTML archives.

- `schema.py` validates the policy-capability-aware audit contract.
- `sft_adapter.py` losslessly adapts existing `RolloutTrajectory` records without model replay.
- `storage.py` copies true PNG observations and atomically publishes a complete evaluation directory.
- `render.py` builds a searchable evaluation index and per-rollout step view without network dependencies.

A missing capability is rendered as unavailable; Q, planner, token, or terminal evidence is never reconstructed after the rollout.
