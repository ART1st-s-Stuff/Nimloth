# E0077 — Full preflight must not depend on an SSH session

## Symptom

A full SFT2 cache-reader preflight continued for several minutes after its SSH
connection closed, then disappeared without writing its atomic `preflight.json`.
The lost stdout/stderr made it impossible to distinguish a later assertion from
session cleanup.

## Cause

The production-scale preflight was owned by a login SSH command even though it
can take longer than the connection remains stable.

## Prevention

- Run the full preflight under a CPU-only Slurm batch job.
- Write its log beside `RUN_OUTPUT` and require an atomic result JSON before
  training submission.
- Keep external monitoring read-only and never start a second preflight while
  the first process is alive.
