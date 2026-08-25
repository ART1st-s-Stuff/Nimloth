# Launch and Lifecycle

## Before launch

Apply the repository `on-experiment-start` skill in the current conversation:

1. Search curated memory for relevant experiment/data/split/checkpoint/resume/output/module constraints. `get` each entry you plan to use and re-read its evidence.
2. Recheck every field in the [experiment task contract](task-contract.md) against current source/config/data and relevant known errors.
3. Confirm local changes are committed and record the exact Git commit in the experiment description. The server worktree must use that commit; do not edit code on the server.
4. For work estimated over three minutes or otherwise expensive, present train/freeze modules, each objective, checkpoint initialization, output, resume, resource/time estimate, and final command to the human. Launch only after explicit approval of that exact contract.
5. Verify output/run identity is unused and checkpoint/data paths exist with compatible metadata.

## W&B naming

When W&B is used, record project, full run name, numeric ID, optional comment, and parameter-token meanings before launch.

- VAGEN retraining uses project `vagen`.
- Other experiments use `nimloth-<stage>`; reuse an existing stable stage for the same experiment family.
- Run name: `<id>[_<comment>]_<params>`.
- Query the target project immediately before launch and choose the next unused numeric ID; a pending Slurm job does not reserve it.
- Smoke runs include comment `smoke`.
- `<params>` distinguishes important hyperparameters; do not use only `default`, a date, or a machine name.

## Slurm and remote jobs

Read `.local/SERVER.md` and the repository-owned `slurm` skill. Before submission:

- ask the human to confirm partition and total GPU allocation;
- query current cluster availability and recheck immediately before submission/replacement;
- do not lock to a node or fixed topology unless the task and available hardware require it; when the implementation supports it, use currently available GPUs across nodes to reach the largest human-approved parallelism rather than waiting for an assumed whole eight-GPU node;
- generally acquire one bash/hold allocation first, then use `srun` inside it; do not create several hold allocations that can trigger QoS contention;
- bind the remote worktree to the approved commit and keep machine-specific paths/credentials under `.local/`.

## Monitoring

Starting a job is not completion. Monitor scheduler state, process logs, resource usage, step-level metrics, NaN/OOM/errors, output creation, and W&B identity until the run is confirmed healthy or a terminal/failure state is established. Do not abandon a launched experiment. If all nodes appear occupied, verify cluster state before waiting.

## Mandatory end event

Completion, failure, cancellation, pause, or discovery that a prior-session run ended triggers `on-experiment-end`. In the current conversation record:

- scheduler/runtime status and actual command/config/commit;
- data/split/checkpoint/output provenance;
- key metrics/anomalies, purpose outcome, and validity limits;
- failure cause or cancellation reason;
- latest checkpoint and exact resume procedure, or why resume is impossible;
- experiment group `progress.md`, task evidence/checklist, concise branch milestone when warranted, and memory/spec implications.

A failed or partial run remains evidence; never rewrite it as success or silently replace it with a smoke run.
