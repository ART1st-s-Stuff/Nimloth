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
- Keep single-device auxiliary modules in DDP, but give them a fresh NCCL
  process group whose first collective uses each rank's actual auxiliary GPU.
- Accumulate pair-sharded Qwen gradients locally, then explicitly all-reduce
  and average every trainable gradient in deterministic parameter order through
  a CPU Gloo group at the optimizer boundary.
- Validate every trainable Qwen parameter has a gradient before beginning the
  ordered collectives.
- Preserve each pre-step failure in a fresh output; never resume it.

## Evidence

Full8192 pair attempt ID20 timed out at step0 with inconsistent DDP parameter
counts. ID22 repeated the post-secondary-communicator stall with all three
pairs on one node. ID23 removed Qwen DDP, then exposed that auto placement put
rank0's auxiliary modules on its secondary GPU but ranks1/2's on their primary
GPUs; reusing the primary NCCL group stalled again.
