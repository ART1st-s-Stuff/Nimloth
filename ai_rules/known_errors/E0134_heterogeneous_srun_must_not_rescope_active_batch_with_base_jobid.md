# E0134 — Heterogeneous srun must not rescope active batch with base job ID

## Error

ID187 retry8 Job `525644` obtained the intended preempt allocation (`dgx-52` with 6 H800 and `dgx-20` with 2 H800), but the first cross-component step used:

```bash
srun --jobid=<base-job-id> --het-group=0,1 --nodes=2 ...
```

Inside the already-running heterogeneous batch, Slurm scoped this nested step to component0 and rejected the two-node request with `Only allocated 1 nodes asked for 2`. The job failed in one second before output, W&B, Ray, environment, or model work.

## Rule

Inside an active heterogeneous batch, use the inherited heterogeneous allocation and explicit `--het-group` without adding the base `--jobid`. Use `--jobid` only when attaching from an external process where no active batch allocation is inherited.

Every cross-component preflight must run before the guarded experiment output is created. A failed attempt receives a fresh output/W&B identity.
