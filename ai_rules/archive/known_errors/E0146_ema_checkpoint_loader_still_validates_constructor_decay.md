# E0146: EMA checkpoint loaders may validate constructor arguments first

## Error

ID58 retry1 passed `decay=0.0` to `build_vision_ema` under the assumption that
the archived checkpoint would immediately overwrite that placeholder. The
factory constructs `VisionEncoderEMA(decay=...)` before loading the checkpoint,
and its constructor requires `0 < decay < 1`.

## Impact

Job `528804` failed after frozen SFT1 and ID74-online hidden extraction but
before vision-EMA extraction or any matrix metric. No parameter update or model
checkpoint occurred; the output and W&B identity cannot be reused.

## Required prevention

- Pass an architecture-valid EMA decay even when a checkpoint will overwrite
  the value; use the checkpoint's configured decay when known.
- Read constructor validation order before treating arguments as ignored
  placeholders.
- Add a preflight regression that rejects invalid placeholder values before GPU
  launch.
