# E0041: Aggregate image budget cannot split one prefix

## Error

A Full8192 LoRA/LoRA world8 retry lowered `max_images_per_batch` from12 to8,
but a late per-prefix row can itself contain eight history images. The sampler
must admit that row as one micro-batch. Its first backward still used76.92GiB
and OOMed while requesting another2.46GiB.

## Required practice

- Treat `max_images_per_batch` as a batch aggregation limit, not a hard bound
  below the image count of one sample.
- Never split one prefix across forwards merely to satisfy the aggregate
  budget; that changes the required per-prefix semantics.
- If one row cannot fit with LoRA, use valid model/pair parallelism or an
  explicitly approved representation-resolution change.
- Preserve the failed output and do not resume when no checkpoint exists.

## Evidence

Full8192 SFT2 attempt ID21, W&B `yjt7ade3`, failed on rank7 before optimizer
step1 with image budget8. No checkpoint was produced.
