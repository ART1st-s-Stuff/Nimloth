# E0038: A successful launcher return is not proof training completed

## Error

An SFT2 control shell invoked `srun` without the Slurm module loaded. The local
site wrapper displayed help and returned success without launching training.
The outer script then wrote `sft2_done.flag` even though no process, W&B run, or
checkpoint existed.

## Required practice

- Before writing a done flag, load and validate the terminal training
  checkpoint.
- Require the requested terminal epoch and `epoch_complete=true`.
- Treat a done flag without that checkpoint as invalid; preserve it as failed
  launch evidence instead of accepting or silently deleting it.
- Load the Slurm module before invoking an allocation-attached launcher from a
  login shell.

## Evidence

The first control attempt under Full8192 pair2 output ID18 produced only the
site's "Did you forget to load the slurm module?" message and a false done
flag. The launcher was hardened before the real retry.
