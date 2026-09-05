# E0025: Pending job name does not reserve a W&B numeric ID

## Error

A W&B numeric run ID was selected and written into a Slurm job name/config while the job was expected to wait in queue. No actual W&B run existed yet, so concurrent experiments consumed that number and advanced the project sequence before the queued job started.

This occurred for the k=1 SFT2 control: documentation-only IDs 3 and 4 became stale while its GPU job remained pending.

## Cause

Slurm metadata and output-directory names are invisible to W&B and other sessions. Treating them as an ID reservation is unsafe in an active shared project.

## Correct practice

Immediately before submission, re-query the target W&B project. If a long queue is expected, create the actual W&B run reservation and persist its internal run ID in the training output before waiting. The reservation may log only explicit queued metadata at step 0; training must reopen the same internal run ID and begin transport steps at 1 or later.

If no actual reservation exists, re-check and rename to the next numeric ID immediately before W&B initialization. Never launch using a numeric ID that has since been consumed or overtaken by later project runs.
