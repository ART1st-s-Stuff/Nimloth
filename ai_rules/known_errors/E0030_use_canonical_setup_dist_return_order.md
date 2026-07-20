# E0030 — Use canonical `setup_dist()` return order

## Error

A new training entry point unpacked `setup_dist()` as three values. The canonical helper returns four values in this order:

```python
rank, world_size, local_rank, device = setup_dist()
```

The SFT1 DINO-grid smoke job `481441` therefore failed before model initialization with `ValueError: too many values to unpack (expected 3)`.

## Prevention

- Read `src/nimloth/training/common/dist.py` before writing a new distributed entry point.
- Reuse the returned `device`; do not reconstruct it from `local_rank`, because the helper also implements `NIMLOTH_DDP_GPU_STRIDE`.
- Exercise a new entry point through a real one-GPU smoke before formal training.
