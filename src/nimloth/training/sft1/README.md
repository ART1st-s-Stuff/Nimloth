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

`config.py` retains the code-canary objective schema and historical CLI defaults
without widening that public interface. `experiment_config.py` separately owns
the strict `nimloth_sft1_state_v2_experiment_v1` early-4 launch schema: source,
data, teacher/cache/output identities, approved 12,841/1,420/1,413 row counts,
all objective constants, dual AdamW rates, three epochs, fixed contrastive-B2
budgets, FSDP/checkpoint/report fields, and unknown-field rejection.
`launch_config.py` can publish one immutable resolved launch config only when
every commit/path/processor/W&B/topology value is explicit.

`real_rows.py` reopens immutable current-format trajectories, selects executed
steps 0–3, hashes each original image, derives exact image/instruction groups,
and applies the existing Qwen inject renderer. Its K8→K16 check permits only the
structural query replacement and locks the archived CoT and action boundary.
`teacher_cache.py` owns deterministic modulo shard ownership, chunked exact-
prefix resume, detached `[16,1024]`/`[2048]`/`[8]` rows, one-time indexed reads,
hash sidecars, and atomic root publication. `teachers.py` batches the real frozen
ID176/original-image DINO forward. `cache_runtime.py` wires fresh generation and
ID60/ID192 parity evidence without importing those old arrays as targets.
`identity.py` hashes the complete processor/tokenizer/template/K16-action bundle.

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

`verl_worker.py` captures original trainable parameter identities before wrapping
and reconstructs exactly two groups after `use_orig_params=True` FSDP: query
adapter at `1e-4`, and fresh projector/readouts at `1e-3`. `driver.py` owns whole-
row deterministic global-update schedules (including at least one real movement
label), raw-row `DataProto` rebuilding, the real Qwen production constructor,
three-epoch loop, complete data cursor, and shared smoke/formal checkpoint path.
`train_runtime.py` wires smoke, exact resume, epoch validation, durable step logs,
and formal result metadata. Resume-smoke additionally gathers one real FSDP full
state and exercises the restricted actor/query+processor+projector export; that
artifact is explicitly smoke evidence, never a selected model.

`checkpoint.py` writes immutable rank-owned model/optimizer/scheduler/RNG shards,
hashes every shard, then publishes run/commit/control/data-cursor identity with
an atomic completion marker. Resume fails on world-size, run, commit, manifest,
config, objective, shard hash, or marker mismatch. Deployable export accepts only actor/query and processor exporters,
the fresh projector state, and strict K16 metadata; training heads, optimizer,
WM, and ValueHead are excluded.

Qwen rendering and the one-forward K16/action-boundary capability remain in
`nimloth.backbone.qwen25vl.state_training`. State rows require persisted real
assistant response/CoT provenance and exact contextual instruction BPE spans.

`validation.py` emits report-first epoch 0/1/2/3 component metrics with fixed-
seed intervals and natural archived groups as the bootstrap unit.
`validation_runtime.py` performs equal-order FSDP validation and low-capacity
teacher-centroid instruction/target probes. Actor KL/agreement are continuation
safety stops, never an automatic quality/SFT2 pass. `controller.py` keeps
cache→smoke→resume-smoke→formal→report sequential, records failed phases, and
performs clean source/interpreter/ID176/DINO/hash/non-overwrite checks before
executing the separately approved exact command.

This package is implementation-ready interface infrastructure only. The checked
YAML remains `launch_locked=false` with unresolved processor/output identities.
It does not authorize cache creation, model training, GPU/Slurm/W&B activity, or
an SFT2/WM/ValueHead change. CPU tests prove transaction/structure only—not real
Qwen, imported VERL transport, CUDA, FSDP numerics, or model quality.
