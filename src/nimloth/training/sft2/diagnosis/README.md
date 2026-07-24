# SFT2 diagnosis

Diagnostic implementations for packed/KV trajectory equivalence investigations. Production SFT2 training remains in the parent package.

| File | Purpose |
|------|---------|
| `trajectory_forward.py` | Prefix-vs-full forward comparison |
| `trajectory_equiv.py` | Legacy-vs-trajectory loss comparison |
| `packed_trajectory.py` | KV incremental-forward prototypes |
| `trajectory_once.py` | Non-equivalent full-trajectory Qwen forward prototype |
| `trajectory_batching.py` | Record grouping used by packed diagnostics |
| `trajectory_cache.py` | Research-only packed-trajectory cache |

These modules are not imported by the production trainer or CLI.
