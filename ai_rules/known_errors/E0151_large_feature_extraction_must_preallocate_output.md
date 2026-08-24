# E0151 — Large feature extraction must preallocate output

## Error

ID192 retry1 accumulated thousands of per-batch NumPy arrays backed by temporary
Torch tensors for two multi-GiB feature products. All 2,229 forwards completed,
but post-loop chunk assembly did not reach atomic cache creation before the
45-minute limit; Job `529788` was cancelled without usable output.

## Rule

For multi-GiB feature extraction, allocate each final float32 array once before
the forward loop. Build an explicit source-index to output-row mapping and copy
each selected batch directly into its final slice. Do not retain per-batch
Torch-backed NumPy chunks for end-of-run concatenation. Log forward-complete,
model-release, cache-write and probe stages so a slow stage is observable.
