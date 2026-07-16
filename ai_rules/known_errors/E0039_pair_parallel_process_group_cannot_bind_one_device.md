# E0039: Pair-parallel DDP cannot bind its process group to one GPU

## Error

A pair-parallel Qwen rank was initialized with
`init_process_group(device_id=primary_gpu)`. DDP then rejected parameters on
the rank's secondary GPU during module-state broadcast.

## Required practice

- Keep explicit `device_id` for stride-one ranks; it is required by non-uniform
  node launchers to avoid ambiguous NCCL rank-to-device mapping.
- When `NIMLOTH_DDP_GPU_STRIDE>1`, select the primary CUDA device locally but
  initialize NCCL without a single `device_id`, because the module spans the
  whole GPU pair.
- Cover both paths with unit tests.

## Evidence

Full8192 pair2 output ID18 failed before step1 with
`Tensor found on device cuda:1 but backend constrained to cuda:0` (and the
corresponding device pairs for every rank). No checkpoint was produced.
