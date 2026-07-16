# E0043: Pair-sharded Qwen gradients need relative GPU groups

## Error

Pair-parallel LoRA synchronized164,496,896 FP32 trainable parameters through
CPU Gloo. Every optimizer step copied about658MB/rank from GPU to CPU,
all-reduced it across nodes, then copied it back. Main rank processes consumed
about10 CPU cores while20-second GPU utilization averaged only12–26%.

A separate placement bug used `device_map.get("lm_head") or ...`; CUDA device0
is falsey, so ranks whose pair starts at0 selected the final norm device while
ranks starting at2/4 selected the lm-head device.

## Required practice

- Never select integer CUDA device IDs with boolean `or`; test `is None`.
- Place auxiliary modules with the final language-model norm that produces the
  latent hidden state.
- Before training, gather every trainable Qwen parameter's relative pair slot
  through Gloo and fail if any rank differs.
- Synchronize primary-slot and secondary-slot gradients through two independent
  NCCL process groups in deterministic parameter/bucket order.
- Keep CPU Gloo only as a diagnostic fallback, not the production data path.
