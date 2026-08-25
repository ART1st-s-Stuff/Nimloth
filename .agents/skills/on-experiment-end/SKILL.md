---
name: on-experiment-end
description: >-
  Records a Nimloth experiment after completion, failure, cancellation, or
  pause. Use whenever such an end state is observed, even for a run started by
  another session.
---

# On Experiment End

## Trigger

Apply immediately after any training, evaluation, collection, calibration, rollout-train, remote long job, Slurm task, or other expensive computation completes, fails, is cancelled, or is paused.

## Required actions

1. Read [launch/lifecycle](../../../.trellis/spec/experiments/launch-and-lifecycle.md), [output/checkpoint evidence](../../../.trellis/spec/experiments/outputs-checkpoints-and-evidence.md), and [tasks/progress/memory](../../../.trellis/spec/governance/tasks-progress-and-memory.md).
2. Update the run README/metadata with status, scheduler/runtime evidence, actual command/config/commit, data/split/checkpoint/output provenance, W&B identity, and train/freeze/objective boundary.
3. Record key metrics/anomalies, failure/cancellation cause, whether the purpose was met, validity limits, and next recommendation.
4. Record the latest checkpoint and exact resume method, or explain why faithful resume is impossible.
5. Update `outputs/experiments/<group>/progress.md` with the latest **valid** result for the parameter setting; do not promote an invalid retry.
6. Update active Trellis task evidence/checklist and a concise `AI_branch_progress.md` milestone when branch-level state changed. Do not create new legacy progress files.
7. Apply the `on-progress` memory evaluation: recheck/upvote used memory only when helpful, add only non-duplicative reusable lessons, and never run human-only approval commands.
