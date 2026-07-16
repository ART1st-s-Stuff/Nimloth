# E0036: Do not estimate compute from checkpoint-contaminated step timestamps

## Error

A Full8192 SFT2 runtime estimate treated adjacent CSV step timestamps as pure
training time. The smoke command saved a multi-gigabyte checkpoint every step
with `--checkpoint-interval-steps 1`; most of the observed 60–80 seconds was
NFS checkpoint I/O. The factorized smoke disabled step checkpoints, so the two
timings were not comparable.

## Required practice

- Compare throughput only when checkpoint cadence and logging boundaries match.
- For compute estimates, disable step checkpoints or use section timing around
  forward/backward/optimizer with explicit synchronization.
- Report checkpoint write time separately from training step time.
- Prefer the first production steps for the final ETA and replace smoke-based
  estimates once they are available.

## Evidence

The Full8192 smoke script under server output
`sft2/3_smoke_state8192full_k8inject_n2_ep1_b2/retry.slurm` used checkpoint
interval one and keep-last two. The factorized smoke
`sft2/6_smoke_dtypefix_state8192_factorwm2048_n2_ep1/run.slurm` set both step
and minute checkpoint intervals to zero.
