# Launch and Lifecycle

## Before launch

Apply the repository `on-experiment-start` skill in the current conversation:

1. Search curated memory for relevant experiment/data/split/checkpoint/resume/output/module constraints. `get` each entry you plan to use and re-read its evidence.
2. Recheck every field in the [experiment task contract](task-contract.md) against current source/config/data and relevant known errors.
3. Confirm local changes are committed and record the exact Git commit in the experiment description. The server worktree must use that commit; do not edit code on the server.
4. Present train/freeze modules, each objective, checkpoint initialization, output, resume, resource/time estimate, and final command in the task. A qualifying remote reproduction/partial rerun expected within 10 minutes may launch without an additional approval question; a materially new run or one expected over 10 minutes launches only after explicit approval of that exact contract.
5. Verify output/run identity is unused and checkpoint/data paths exist with compatible metadata.
6. Confirm the command will execute remotely from the declared committed source. Do not launch real experiment work on the local development machine.

## Short experiment deadline

A short experiment is a scoped remote reproduction or partial-parameter rerun expected to complete within 10 minutes. With a complete lightweight experiment task and unchanged reviewed scope, it needs no additional launch question. From successful scheduler submission it has a 15-minute total deadline covering both pending and running states; queue time is not free. Monitor elapsed wall time from the recorded submission event.

At the deadline, cancel the job and verify its terminal scheduler/process state. Record partial outputs as partial evidence only, then either defer the experiment or mark the task blocked with the concrete resource/runtime blocker. Cancellation, a smoke substitute, or a later unrelated run must not be reported as successful validation of the expired experiment.

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

- use the partition and total GPU allocation declared in the reviewed task; ask again only when a longer/materially new run or resource-scope change requires launch approval;
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
