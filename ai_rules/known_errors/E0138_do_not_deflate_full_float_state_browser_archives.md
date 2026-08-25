# E0138 — benchmark and instrument full audit batches before blaming compression

## Error

ID189 retry2 was cancelled after the first 40 rollouts completed but the single TaskRunner spent more than 2h21m CPU, reached about 111 GB RSS, and produced no atomically committed batch. The delay was attributed to serial `np.savez_compressed` without phase-level timing evidence, then retry3 was launched with ZIP-stored `np.savez`.

Retry3 disproved that root-cause claim: after its first 40 environments completed, it again spent over two hours in in-memory artifact construction with zero NPZ writes. `/proc/<TaskRunner>/io` showed only the roughly 400 MB planner snapshot had been written.

Read-only runtime and code inspection then isolated the mechanism. `_jsonable_planning_evidence` converts every predicted `[16,1024]` tensor to nested Python lists before transport. A 40-rollout × 20-turn × about271-node batch therefore carries about 3.55 billion Python float entries. `_build_validation_rollout_browser_artifacts` runs on one TaskRunner thread; `_pack_k4_state_trace` first `deepcopy`s each complete MCTS trace, then converts every node list back through `np.asarray`, stacks it, and retains the in-memory NPZ bytes until the whole atomic batch is built. At elapsed 3h16, one thread was continuously consuming one CPU, the process had about123 GB RSS/about99 GB private anonymous memory, output I/O remained about400 MB, and no NPZ/output file descriptor existed. Ptrace was disabled, so an exact live Python line could not be sampled, but the active control-flow region and pathological list representation are established. Compression is downstream and was not the dominant delay.

## Correct practice

Before scaling a full-state audit:

1. benchmark a production-sized atomic batch, not only one rollout;
2. separately time rollout collection, artifact materialization, JSON conversion, NPZ serialization, validation, and atomic commit;
3. keep predicted states as tensor/NumPy buffers across the trusted transport rather than billions of Python floats; avoid deep-copying state-bearing traces;
4. stream or atomically commit bounded rollout-sized units instead of retaining a 40-rollout batch of archives in memory;
5. record archive compression ratio and compare uncompressed, serial compressed, and parallel compressed implementations;
6. do not cancel, change storage format, or launch a replacement without human approval when the evidence does not isolate the bottleneck.

Float32 keys, dtypes, shapes, finite checks, and SHA256 gates must remain unchanged regardless of the chosen storage implementation.

## Evidence

- VAGEN `vagen/ray_trainer.py::_pack_k4_state_trace` and `_build_validation_rollout_browser_artifacts`.
- ID189 retry2 Job `527287` and retry3 Job `527471` outputs/progress.
- `tests/test_evaluation_rollout_browser.py`.
