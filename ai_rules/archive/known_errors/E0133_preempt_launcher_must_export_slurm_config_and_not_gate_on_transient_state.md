# E0133 — Preempt launcher must export Slurm config and not gate on transient job state

## Error

A Slurm batch launcher treated an immediate in-job `scontrol JobState=RUNNING` result as a hard gate and assigned the fixed `SLURM_CONF` without explicitly exporting it after `module load slurm`.

On preempt Jobs `525570` and `525572`, external `squeue` showed the batch running while the in-job state assertion failed. After removing that assertion, Job `525575` showed the in-job client lookup still lacked the expected `JobId`. All jobs exited before output, W&B, Ray, environment, or model work.

## Rule

For Slurm-owned batch controllers:

1. Explicitly export the fixed `SLURM_CONF` and prepend the verified Slurm binary directory before the first Slurm client command.
2. Do not require an immediate `JobState=RUNNING` string from inside the already-executing batch script.
3. Prove allocation ownership with exact `SLURM_JOB_ID`, nonempty allocated `NodeList`, requested/allocated TRES, expanded node/GPU mapping, and actual CUDA visibility.
4. Preserve an ERR diagnostic that records the failed line and command before retrying with a fresh output/W&B identity.

## Regression gate

`tests/training/rl/test_vagen_k4_id185_visualization_launcher.py` checks the exported Slurm configuration, fixed binary path, allocation identity checks, and absence of the transient state assertion.
