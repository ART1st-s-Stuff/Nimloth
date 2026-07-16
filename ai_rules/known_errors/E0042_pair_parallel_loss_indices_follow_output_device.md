# E0042: Pair-parallel loss indices must follow the output device

## Error

With Qwen and auxiliary modules on different GPUs, value-head outputs were on
the auxiliary GPU while `action_indices` remained on the Qwen hidden-state
GPU. `values.gather(...)` failed before step1 on ranks whose devices differed.

## Required practice

- Align gather indices and masks to the tensor they index, not to an upstream
  hidden state's device.
- Align scalar/regression targets to the final output tensor as well.
- Exercise an actual mixed-device forward; communicator-only smoke tests cannot
  detect loss-local device mismatches.

## Evidence

Full8192 SFT2 ID24 passed split process-group initialization and reached the
value loss, then failed with `cuda:2` versus `cuda:3` and `cuda:4` versus
`cuda:5`. W&B was `z55zmh4r`; no optimizer step or checkpoint was produced.
