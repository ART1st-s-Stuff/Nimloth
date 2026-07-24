# SFT2 diagnosis scripts

One-off debugging, probing, validation, cache estimation, and performance smoke scripts. These are not production training entrypoints.

- `debug_*` / `diagnose_*`: inspect encodings and trajectory equivalence failures.
- `probe_*`: isolate Qwen2.5-VL prefix, KV-cache, attention, and vision behavior.
- `validate_*`: GPU validation gates for experimental forward paths.
- `smoke_speedup.*`: performance/equivalence smoke only.
- `estimate_preprocess_cache.py`: cache-size estimator.
