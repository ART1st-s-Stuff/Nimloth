# E0031: Shared env service requires unique manager IDs or serial policy managers

## What happened

Formal SFT1 rollout array `480518` allowed two independent policy managers to use one navigation service concurrently. Both managers generated the same environment IDs (`val1`, `val2`, ...). One manager overwrote or closed environments owned by the other, producing server `NoneType` step errors and policy `KeyError: metrics`. No complete JSONL was produced.

## Required prevention

Do not run multiple rollout-manager processes concurrently against one stateful VAGEN env service unless environment IDs are guaranteed globally unique per manager.

For the current source-compatible SFT1 path, use one policy array task at a time (`array concurrency %1`). It may use multiple policy GPUs internally, but there must be only one manager issuing create/reset/step/close calls to the service.
