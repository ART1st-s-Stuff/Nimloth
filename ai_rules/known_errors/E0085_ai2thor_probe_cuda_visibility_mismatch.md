# E0085: AI2-THOR probe CUDA visibility differed from the formal env service

## Symptom

A node-level direct render probe can pass a physical GPU while the formal
rollout shard using that same physical GPU still hangs before its first
navigation observation.

ID135 hold Job `508268` exposed the mismatch on `dgx-32`.  With all eight GPUs
visible, direct probes using `gpu_device=0` and `gpu_device=4` returned valid
frames.  The formal 1x8 runner instead made only the shard's renderer GPU
visible and used relative `navigation.devices=[0]`: shard 0 passed prewarm in
4.924 seconds, while shard 1 on physical GPU 4 exceeded the 300-second gate and
hung during cleanup.

## Cause

AI2-THOR's CUDA/Vulkan mapping depends on `CUDA_VISIBLE_DEVICES` as well as the
requested `gpu_device`.  These contracts are not equivalent:

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, `gpu_device=4`;
- `CUDA_VISIBLE_DEVICES=4`, `gpu_device=0`.

A probe from the first contract cannot qualify the second contract.

## Prevention

- Reproduce each formal environment service's exact `CUDA_VISIBLE_DEVICES`,
  relative `gpu_device`, AI2-THOR home and concurrency before releasing a node.
- Keep probe homes isolated so mapping files do not race, but do not change the
  CUDA visibility being qualified.
- Require every formal rollout shard's real navigation prewarm to pass.  A
  healthy shard's local trajectories cannot compensate for another shard that
  fails before global strict merge.
- If timeout cleanup leaves the server process stuck, cancel the affected Slurm
  steps, verify no Unity/Ray/vLLM/GPU processes remain, and preserve the output
  as a failed non-resumable identity.

## ID135 boundary

The failed 1x8 attempt wrote seven local shard-0 trajectories and no shard-1
trajectory.  It created no global fresh manifest, consumption record, optimizer
update, checkpoint or W&B run.  The local trajectories are not consumable and
the identity must not be reused.
