---
name: on-experiment-start
description: >-
  Enforces Nimloth's experiment launch contract. Use immediately before any
  training, evaluation, collection, calibration, rollout-train, remote long
  job, Slurm task, or GPU/expensive computation.
---

# On Experiment Start

## Trigger

Pause before the command that starts any experiment or expensive/remote job, even when implementation preparation was already approved.

## Required actions

1. Read the [experiment index](../../../.trellis/spec/experiments/index.md), [task contract](../../../.trellis/spec/experiments/task-contract.md), [data/split rules](../../../.trellis/spec/experiments/data-and-splits.md), [launch/lifecycle](../../../.trellis/spec/experiments/launch-and-lifecycle.md), and [output/checkpoint evidence](../../../.trellis/spec/experiments/outputs-checkpoints-and-evidence.md).
2. Confirm the active task has `task.json.meta.kind = "experiment"` and every required field is explicit and source-verified.
3. Search relevant curated memory. `get` and re-read evidence for each memory that will influence the launch.
4. Read task-relevant known errors and, for remote work, `.local/SERVER.md` plus the `slurm` skill.
5. Confirm the local work is committed, the exact commit is recorded, and the remote worktree uses it.
6. Verify final command/config, full parameter names, data/split, checkpoint ownership, train/freeze/objectives, unique output, resume, metrics/validity gates, W&B identity, and resource/time estimate.
7. Present that exact contract and obtain separate explicit human launch approval. Implementation/task-start approval is insufficient.
8. If any item is missing or changed after approval, stop and ask again. Do not launch an approximate replacement.
9. After launch, monitor scheduler/process/log/resource/metric/output state until healthy or terminal; do not abandon the job.
