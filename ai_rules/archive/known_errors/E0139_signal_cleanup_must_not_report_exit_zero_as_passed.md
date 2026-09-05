# E0139 — signal cleanup must not report exit zero as passed

## Error

When ID189 retry2 Job `527287` was cancelled, the runner's EXIT trap observed status zero and wrote `phase_status.json` as passed even though Slurm reported cancellation and no validation/browser finalization existed.

## Cause

The shell had no TERM/INT handler preserving a nonzero termination status before the EXIT cleanup trap ran.

## Correct practice

Long runners must trap TERM/INT, set an explicit termination status, and make cleanup use that status. A phase may report passed only after its final validator; Slurm terminal state and final artifacts remain authoritative. Correct misleading metadata during the experiment-end hook.

## Evidence

- `experiments/training/rl/run_vagen_k4_id189_source20_base_common120*.sh`.
- Corrected retry2 `base_common120/phase_status.json` and `progress.md`.
