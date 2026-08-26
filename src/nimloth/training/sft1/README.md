# SFT1 training

`objective.py` owns the state-interface-v2 canary's complete differentiable
objective. It preserves one deployed row-major `[B,16,1024]` state and adds only
training-time linear visual, fixed-mean instruction, observed-movement, and
state-policy readouts. DINO supervision is cosine plus slot relations through a
readout; there is no direct state-to-DINO MSE. The worker supplies the global
executed-movement valid count and framework gradient-average world size; the
objective exposes its local feasibility numerator/count and scales the local
sum so official FSDP averaging yields the global valid-row mean. An empty local
rank keeps a zero autograd graph and never invents a label.

`config.py` keeps the historical CLI defaults adapter separate from the strict
`nimloth_sft1_state_v2_config_v1` schema. The v2 parser rejects unknown, missing,
legacy-version, freeze-contract-changing, non-ID176 dimension, and noncanonical
movement-action mapping values.

`manifest.py` binds the actor/processor/prompt/token table, K16/eight-action
contract, DINO identity, trajectory/teacher hashes, and exact parent
VAGEN/nested VERL commits. `data.py` accepts the current trajectory schema only,
requires original-image digest parity and real archived response/CoT, and masks
nonmovement or unavailable outcomes without inventing labels. `verl_adapter.py`
transports detached teacher targets plus unpadded encoded prompt rows in pinned
VERL `DataProto`; deterministic first-fit packing budgets exact padded-token cost
and never truncates a row. It also rejects precomputed student hidden/state.
`verl_worker.py` computes update-global valid counts before microbatching, pads
only collective order with `row_valid=false`, accumulates through the complete
wrapped root, and delegates synchronization/clipping to official FSDP. Detached
count/metric reductions never synchronize gradients manually.

`checkpoint.py` writes immutable rank-owned model/optimizer/scheduler/RNG shards,
then publishes control/data-cursor identity with an atomic completion marker.
Resume fails on world-size, manifest, config, objective, partial-shard, or marker
mismatch. Deployable export accepts only actor/query and processor exporters,
the fresh projector state, and strict K16 metadata; training heads, optimizer,
WM, and ValueHead are excluded.

Qwen rendering and the one-forward K16/action-boundary capability remain in
`nimloth.backbone.qwen25vl.state_training`. State rows require persisted real
assistant response/CoT provenance and exact contextual instruction BPE spans.

This package is code/interface canary infrastructure only. It does not establish
state quality and does not authorize teacher-cache generation, model training,
or an accelerator/remote experiment. A separately approved real-data canary
with pre-registered quality gates is required before any SFT2 claim.
