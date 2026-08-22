# E0138 — benchmark and instrument full audit batches before blaming compression

## Error

ID189 retry2 was cancelled after the first 40 rollouts completed but the single TaskRunner spent more than 2h21m CPU, reached about 111 GB RSS, and produced no atomically committed batch. The delay was attributed to serial `np.savez_compressed` without phase-level timing evidence, then retry3 was launched with ZIP-stored `np.savez`.

Retry3 disproved that root-cause claim: after its first 40 environments completed, it again spent over two hours in in-memory artifact construction with zero NPZ writes. `/proc/<TaskRunner>/io` showed only the roughly 400 MB planner snapshot had been written. The dominant delay therefore occurs before archive serialization, likely in audit materialization/copying/conversion; the exact function is still unmeasured.

## Correct practice

Before scaling a full-state audit:

1. benchmark a production-sized atomic batch, not only one rollout;
2. separately time rollout collection, artifact materialization, JSON conversion, NPZ serialization, validation, and atomic commit;
3. record archive compression ratio and compare uncompressed, serial compressed, and parallel compressed implementations;
4. do not cancel, change storage format, or launch a replacement without human approval when the evidence does not isolate the bottleneck.

Float32 keys, dtypes, shapes, finite checks, and SHA256 gates must remain unchanged regardless of the chosen storage implementation.

## Evidence

- VAGEN `vagen/ray_trainer.py::_pack_k4_state_trace` and `_build_validation_rollout_browser_artifacts`.
- ID189 retry2 Job `527287` and retry3 Job `527471` outputs/progress.
- `tests/test_evaluation_rollout_browser.py`.
