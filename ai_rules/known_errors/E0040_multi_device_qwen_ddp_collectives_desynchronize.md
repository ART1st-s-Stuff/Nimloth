# E0040: Multi-device Qwen must not use one DDP reducer

## Error

Pair-parallel SFT2 wrapped a Qwen model spanning two GPUs in ordinary DDP.
Ranks used different CUDA ordinal pairs. Qwen's multi-device DDP collectives
and the following single-device auxiliary DDP constructors entered inconsistent
collective sequences. StateProjector shape verification eventually saw six
parameters on some ranks and zero on another. This occurred in the 3+1
multi-node topology; an all-local three-pair retry showed the same prolonged
collective stall pattern.

## Required practice

- Do not wrap a pair-sharded Qwen module in one ordinary DDP reducer.
- Keep single-device auxiliary modules in DDP.
- Accumulate pair-sharded Qwen gradients locally, then explicitly all-reduce
  and average every trainable gradient in deterministic parameter order at the
  optimizer boundary.
- Validate every trainable Qwen parameter has a gradient before beginning the
  ordered collectives.
- Preserve each pre-step failure in a fresh output; never resume it.

## Evidence

Full8192 pair attempt ID20 timed out at step0 with inconsistent DDP parameter
counts. ID22 repeated the post-secondary-communicator stall with all three
pairs on one node, ruling out the node boundary as the sufficient cause.
