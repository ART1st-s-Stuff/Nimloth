# E0036: AI2-THOR must not share an FSDP trainer GPU

## Error

ID14 job `478603` used eight trainer ranks but placed AI2-THOR on rank7's H100 to fit an eight-GPU normal allocation. The environment health check passed and all ranks initialized, but the first policy forward made no progress.

`nvidia-smi pmon` on dgx-51 showed:

```text
thor-CloudRendering  C+G  99% SM
python3              C     trainer rank7
```

Rank7 remained blocked in the GPU runtime while the other seven FSDP ranks waited at collectives with 0% GPU utilization. The job was cancelled after 11:18 with no trajectory or optimizer step.

## Correct practice

Dynamic FSDP smoke/training requires eight dedicated trainer GPUs **plus** a separate AI2-THOR GPU. Never use `srun --overlap` to place the environment and a trainer rank on the same GPU. Treat the env GPU as part of the minimum allocation and fail submission rather than silently sharing it. Image-only or env-health checks do not detect this failure; verify live per-process GPU utilization when a first collective policy turn stalls.
