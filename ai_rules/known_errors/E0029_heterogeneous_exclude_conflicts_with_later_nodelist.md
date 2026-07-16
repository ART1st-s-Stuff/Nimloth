# E0029: heterogeneous `--exclude` can conflict with a later component `--nodelist`

## Error

A heterogeneous Slurm script put `--exclude=dgx-37` in trainer group0 and
`--nodelist=dgx-37` in env group1. `sbatch` rejected the entire atomic submission
with `Invalid node name specified` before creating a job.

## Rule

Do not assume `--exclude` is scoped only to the component where it appears. When
a later heterogeneous component explicitly needs a node, constrain the earlier
component with a positive `--nodelist` allowlist that omits that node instead of
excluding it.

## Verification

A failed `sbatch` command creates no job/allocation; verify this before reusing
the same experiment identity and output for the corrected submission.
