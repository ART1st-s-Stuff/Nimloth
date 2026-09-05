# E0131: checkpoint transport-path migration does not copy the active snapshot

## Incident

ID185 retry4 Job `524485` completed all 300 validation episodes, wrote eight atomic journal batch markers, published the 300-row journal complete marker and final validation dump, and finished the only W&B history step20. The post-run finalizer nevertheless failed because it required:

`$RUN_OUT/planning_snapshots/source_step_796/frozen_k4_planner.pt`

The runtime output contains the bootstrap `source_step_776` artifact. Exact checkpoint restore installs the immutable ID184 `source_step_796` transport directly from the source checkpoint path; training-contract transport-root migration permits this path change but does not copy the source796 file into the new output.

## Confirmed evidence

- Source checkpoint owner state identifies source796 and snapshot `sha256:6648780b...`.
- `restore_frozen_q_checkpoint_state()` constructs and installs that transport, then returns active source/snapshot identity.
- `_load_checkpoint()` compares the returned owner identity with every restored actor rank before printing `ID185_K4_FULL_EVAL_RESTORE_OK global_step=20`.
- A read-only audit loaded the immutable source796 transport and verified all 300 journal/final rows and hashes.

## Rule

- Finalizers must validate the active transport path recorded in the immutable source checkpoint, its source step, snapshot ID, contract ID, and runtime restore marker.
- A transport-root contract migration must not be interpreted as evidence that the active snapshot file was copied into the destination root.
- Do not require a destination `source_step_*` file unless the runtime contract explicitly promises and verifies such a copy.
- A complete evaluation whose packaging finalizer fails remains non-official until an approved strict read-only finalization or a fresh full retry succeeds; do not silently rewrite the failed output.
