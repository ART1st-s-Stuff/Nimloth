# E0078 — Zero-byte completion flags require existence checks

## Symptom

Formal SFT2 job `500936` obtained its full normal `4+4+2+2` H800 allocation,
then both heterogeneous components failed after one second before controller
logging, model loading, W&B creation, or training output.

## Cause

The preprocessing pipeline creates `cache_done.flag` with `touch`, so the valid
sentinel is intentionally zero bytes. Formal batch launchers included the flag
in a `test -s` loop, which rejects empty files, while their node launchers and
the full preflight correctly used `test -f`.

## Prevention

- Validate completion sentinels created by `touch` with `test -f`, not
  `test -s`.
- Continue to require non-empty model configs, datasets, manifests, and atomic
  preflight result files with `test -s`.
- Keep a static launcher regression that rejects putting `cache_done.flag` back
  into the non-empty-file loop.
