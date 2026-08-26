# VERL training infrastructure

This package owns reusable execution mechanics only: exact parent VAGEN/nested
VERL source verification, complete-root device/FSDP assembly, optimizer-after-
wrap construction, and framework-owned gradient clipping. Stage data schemas,
objectives, masks, normalization, checkpoint meaning, and metrics remain in the
owning training stage.

It must not import or reproduce VAGEN PPO/GAE/reward semantics. Structural local
tests prove assembly order only; real CUDA and multi-rank behavior require the
separate project launch gate.
