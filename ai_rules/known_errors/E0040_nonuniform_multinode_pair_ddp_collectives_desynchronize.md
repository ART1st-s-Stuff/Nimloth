# E0040: Non-uniform multi-node pair-parallel DDP collectives desynchronize

## Error

Full8192 pair-parallel SFT2 used three two-GPU ranks on one node and one
two-GPU rank on another. Even after correct primary-device bootstrap, DDP's
multi-device Qwen synchronization and subsequent single-device auxiliary DDP
constructors entered inconsistent collective sequences. StateProjector shape
verification saw six parameters on some ranks and zero on another; the NCCL
watchdog timed out.

## Required practice

- Do not use the 3+1 non-uniform multi-node pair topology for this trainer.
- Pair-parallel success on one homogeneous node does not prove this topology.
- Prefer stride-one DDP with a smaller trajectory image budget, or redesign
  process groups so multi-device Qwen and auxiliary modules have explicit,
  independently ordered collectives.
- Preserve each pre-step failure in a fresh output; never resume it.

## Evidence

Full8192 pair attempt ID20 reached Qwen's primary and secondary GPU
communicators, then failed in `DDP(state_proj)` with inconsistent parameter
counts and NCCL sequence37 timeout. W&B was `zh8endhv`; no step or checkpoint
was produced.
