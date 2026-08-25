# Rollout browser

This package converts behavior-time evidence from rollout-based evaluations into immutable offline HTML archives.

- `schema.py` validates the policy-capability-aware audit contract.
- `sft_adapter.py` losslessly adapts existing `RolloutTrajectory` records without model replay.
- `storage.py` copies true PNG observations, verifies full float32 `.npz` model-state archives, and atomically publishes a complete evaluation directory.
- `render.py` builds a searchable evaluation index and per-rollout step view without network dependencies.

A missing capability is rendered as unavailable; Q, planner, token, state, or terminal evidence is never reconstructed after the rollout.

`nimloth_rollout_audit_v3` adds explicit `model_state` and `mcts_process` capabilities. A complete K4 MCTS turn stores the same-generation `16×2048` latent hidden, the behavior-time projector output `16×1024`, every unique predicted MCTS node state, and all 100 chronological UCT simulations (selection/expansion inputs, leaf action values, scalar leaf value, and backup before/after values). State tensors are stored once per unique node in a hash-bound `.npz`; JSON process records reference those node indices. Legacy v2 audits remain readable and declare both new capabilities unavailable.
