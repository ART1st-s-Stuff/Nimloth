---
name: slurm
description: >-
  Guides Nimloth Slurm and remote GPU operations using the machine-specific
  .local server contract. Use for resource queries, hold allocations, srun,
  remote worktrees, or any Slurm experiment.
---

# Slurm

## Trigger

Use for SSH/server access, Slurm resource queries/submission, hold allocations, `srun`, remote GPU/long jobs, or remote worktree synchronization.

## Authority and machine boundary

Read before any remote action:

- [experiment index](../../../.trellis/spec/experiments/index.md);
- [launch/lifecycle contract](../../../.trellis/spec/experiments/launch-and-lifecycle.md);
- [experiment task contract](../../../.trellis/spec/experiments/task-contract.md);
- `.local/SERVER.md` for current host aliases, remote paths, credentials, partitions, and machine-specific commands.

This repository skill contains only portable behavior. Do not copy host names, absolute server paths, credentials, current node inventory, or transient cluster facts into it; keep them under `.local/`.

## Refusal gates

- Remote/GPU/Slurm work requires a Trellis experiment task with `task.json.meta.kind = "experiment"`.
- Apply `on-experiment-start`; missing parameters or data/checkpoint/output semantics stop the operation.
- Ask the human to confirm partition and total GPU resources.
- Present the exact command, train/freeze/objectives, checkpoint, output, resume, monitoring, and resource/time estimate; obtain separate launch approval.
- Local changes must be committed and the remote worktree must point to that exact commit. Never edit code directly on the server.

## Connection and resources

Use the current commands/aliases documented in `.local/SERVER.md`. If connection times out and local documentation identifies VPN as the likely cause, stop and ask the human to restore connectivity; do not loop retries.

Query cluster state before submission and again immediately before replacing or launching a pending job. Prefer the repository's local wrapper when available:

```bash
.local/scripts/query-resources.sh
.local/scripts/query-resources.sh --only-free-gpu
```

Do not infer availability from stale notes or a previous command.

## Hold allocation and execution

Unless the approved task requires another topology, prefer one bash/hold allocation and launch work inside it with `srun`. One hold reduces requeue waste after script failures; several simultaneous holds can trigger QoS contention.

```bash
srun --jobid <approved-job-id> --pty <command>
srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w <allocated-node> bash -lc '<approved-command>'
```

Do not hard-code a node or fixed topology merely for convenience. Use the human-approved total resources and current availability while preserving the training/runtime topology contract.

## Monitor and close

Monitor scheduler state, logs, resources, metrics, output creation, and experiment identity until the job is healthy or terminal. Completion, failure, cancellation, or pause immediately triggers `on-experiment-end`; record scheduler/runtime evidence, outputs, metrics/limits, checkpoint/resume, task progress, and experiment-group progress in the current conversation.
