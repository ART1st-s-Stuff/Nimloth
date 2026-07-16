# E0028: Zero-update RL runs must not write a final checkpoint

## Error

Dynamic RL smoke job `477219` collected zero valid trajectories after environment startup timed out. The trainer skipped the iteration and then entered unconditional final-checkpoint saving with `global_step=0`. It wrote optimizer files and began a multi-GB model shard before the job was cancelled.

Such a directory can look like a trained final checkpoint even though no optimizer update occurred.

## Correct practice

Before final checkpoint materialization, require `global_step > 0`. If no optimizer step completed:

1. finish logging with a failed status;
2. clean up the distributed process group collectively;
3. raise an explicit error;
4. do not create `final/`.

Preserve any already-created partial output for diagnosis; do not resume or reuse it as a successful training output.
