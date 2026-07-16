# E0026: Recheck Slurm state immediately before replacing a pending job

## Error

A job was observed as pending, then obtained its requested node while the agent was preparing a replacement submission. The agent cancelled it without one final state check, interrupting a healthy startup after the environment server had passed health.

This occurred for RL smoke job `477075`: it transitioned from pending to running on dgx-13 and was cancelled 19 seconds later. Replacement `477078` correctly refused the now-nonempty output.

## Cause

Slurm state can change between separate query and cancellation commands. A prior `PENDING` observation is not authorization to assume elapsed0/no output later.

## Correct practice

Immediately before cancelling or replacing a supposedly pending job:

1. Query `squeue` and `sacct` again in the same remote critical section.
2. Cancel only if the job is still pending and has no allocation, elapsed runtime, batch step, or output.
3. If it has started, leave it running unless the human explicitly requests cancellation or a correctness problem requires stopping it.
4. Never reuse a nonempty output after such a race; preserve it and use a new W&B numeric ID/output for retry.
