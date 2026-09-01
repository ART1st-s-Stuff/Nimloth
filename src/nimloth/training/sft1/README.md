# SFT1 training

`query_state.py` owns the new `nimloth_sft1_query_state_v1` objective/root. It
consumes one same-forward raw K16 hidden/action/LM-CE result, maps hidden through
the unique no-bias `DirectSlotProjector`, and activates exactly two globally
normalized terms: `2 * direct original-observation DINO MSE + 1 * real final-assistant CE`.
It returns raw Query hidden and canonical state separately. Its inventory helper
requires full language parameters including the top-level LM head, frozen visual
parameters, an absent query additive adapter, and one disjoint direct-state group.

The Query-State production-preparation path is separate from legacy v2:

- `query_state_data.py` renders the complete current archived response (real CoT,
  K8→K16 structural replacement, actual action and action-end), labels only the
  final current assistant span, masks every validated structural Query position,
  re-hashes the original image, and accepts only detached `[16,1024]` DINO targets
  produced from that image. It never accepts fixed/synthetic CoT or student state.
- `query_state_adapter.py` owns `nimloth_sft1_query_state_dataproto_v1`; it transports
  raw encoded rows including labels plus detached DINO targets and rejects hidden,
  Query-hidden, projected-state, or state cache keys before supervised collation.
- `query_state_runtime.py` constructs only full-language/frozen-vision K16 Qwen plus
  `DirectSlotProjector`, rejects LoRA/query adapters, captures exhaustive/disjoint
  language and direct-state parameter identities before complete-root FSDP, and
  reconstructs exactly those two AdamW groups after `use_orig_params=True` wrapping.
- `query_state_distributed.py` computes update-global valid state-element and
  final-assistant LM-token denominators before microbatching. It equalizes
  complete-root forward/backward order with explicit invalid padding, performs
  exactly one optimizer step, and delegates synchronization/global clipping to
  official FSDP rather than reducing parameter gradients manually.
- `query_state_config.py` is an explicit-field, non-launching code-canary schema.
  It has no command/output/resource/update-budget defaults and rejects any
  `launch_authorized=true` value. `query_state_driver.py` owns deterministic raw-
  row partition/padding/resume, fresh render+DINO→DataProto assembly, and the
  barriered immutable rank-checkpoint transaction; it is not an experiment loop.
- `query_state_validation.py` accumulates detached raw Query hidden, unique
  canonical state, LM CE, and same-forward action-logit diagnostics only. Its
  report has no model-quality pass or checkpoint-selection bit.
- `query_state_checkpoint.py` keeps the direct-state/local-resume identities and
  adds immutable rank-owned model/optimizer/scheduler/RNG shards. Rank zero
  publishes data+metric cursors and exact source/config/run/world identity only
  after hashing every shard, then writes an atomic completion marker. A separate
  deployable bundle has distinct full-Qwen, processor, and no-bias direct-state
  owners and rejects legacy/`SharedSlotProjector` or training-only payloads.

The formal Query-State owner is also schema-distinct from both paths above:

- `query_state_training_config.py` owns the explicit template/preflight/launch
  lifecycle, pilot/formal/visual-forensic-fork incompatibility, complete optimizer/runtime/FSDP/
  environment/command fields, and formal W&B fresh (`resume=never`) versus exact
  restart (`resume=must`) state machine. Shared environment values are reapplied
  from the locked run identity after credential sourcing; init/query disagreement
  is an all-rank hard failure.
- `query_state_training_manifest.py` validates the 12,836/5/1,420/1,413/42/101
  audit, resolves coverage-first strata from the canonical navigation action
  table without inventing movement feedback, and splits external rows by whole
  exact-image/normalized-instruction connected components. Pilot access resolves
  only calibration rows; formal access resolves only untouched holdout rows. A
  separate immutable generation-format manifest registers exact rows, production
  prompt/spec/parser identities, and reasoning/output budgets.
- `query_state_training_runtime.py` makes checkpoint cadence a resumable segment
  boundary. Formal `epoch_updates` is separate from checkpoint cadence: sub-epoch
  commits save the full exact-resume state without running calibration or advancing
  early-stop patience, while real epoch boundaries retain calibration/holdout and
  terminal semantics. Update records remain pending until due validation and safety,
  exact checkpoint control/cursor hashes, and a same-run immutable mirror batch
  exist. Preflight binds the measured complete-checkpoint byte estimate to every
  commit in the currently approved process window plus the minimum-free reserve;
  it does not assume retention or delete authoritative checkpoints. A formal
  `approved_pause_update` may stop only at a real epoch boundary after its safe
  authoritative commit and W&B mirror. The field limits process authorization,
  remains outside resume-critical identity, and cannot mark a terminal-primary
  checkpoint; continuation requires a newly approved higher boundary. The atomic
  authoritative index moves before
  idempotent W&B replay. A due safety failure preserves the complete rank transaction
  under `forensics/unsafe_update_*` and binds it in immutable failure evidence for
  explicit read-only debugging, but marks it forensic-only/non-resumable and never
  advances the authoritative index or safe W&B mirror.
  Pre-index crashes quarantine the segment and replay from the prior commit. Pilot
  restart receipts require a different process plus exact model/optimizer/scheduler/
  RNG fingerprints and data/validation/log/W&B cursors.
- `query_state_training_backend.py` runs the shared production
  `TurnGenerationSpec`/parser through current complete-root FSDP logits on the
  registered real unacted response-policy prompts. Update 0 and terminal are
  mandatory; additional format checks require explicit cadence. Parse failure is
  non-resumable and never executes actions, persists rollout, or exports.
- `query_state_visual_forensic_fork.py` is consumed only by the production
  training config/preflight/backend owner; it has no alternate launch schema or
  entry point. It binds the current runtime checkout commit/source-manifest separately from the
  immutable Formal38 ancestor source commit/source-manifest, authenticates that ancestor's
  update-1605 forensic control and eight rank shards, loads model/direct-head tensors only, and proves fresh
  optimizer/scheduler/RNG/data/W&B ownership remains untouched on the fresh fork;
  later exact resume accepts only fork-owned payload-present checkpoints. Its fixed event
  plan covers equivalent epochs 2–5 from schedule offset 1605; durable log/W&B
  cursors begin at that offset, and the segment index reopens against the stable
  semantic run identity rather than process-specific config/approval identity.
  Actor/generation remain report-only, calibration runs at fork step zero and every epoch end, and
  holdout runs only at fixed epoch 5. The segment store permits successor-first
  compaction only for this mode: after successor index and W&B mirror publication,
  it inventories and hashes exactly eight rank payloads, writes a tombstone, removes
  only superseded non-epoch-final payload files, and marks the historical index
  entry non-resumable only after all inventoried rank payloads are absent. An
  interrupted deletion records remaining payloads and recovery deterministically
  authenticates the successor before removing every survivor. Before deletion it validates canonical control/manifests
  and shard hashes against the store's complete trusted fork resume identity;
  no source or manifest identity is learned from the candidate checkpoint.
  Ancestor forensic paths are outside the fork checkpoint root.
- `query_state_training_controller.py` owns non-overwrite run claims, immutable
  nonterminal pause receipts, and distinct completed/failed/preempted/
  validator-failed terminals. It never submits Slurm,
  extends pilot into formal, starts SFT2, or exports automatically.
- `query_state_training_validation.py` owns detached same-forward metadata and
  feature joins, globally attributable raw/direct/DINO/actor/upstream/natural-pair
  metrics, the unique float64 centered entropy-effective-rank formula,
  validation-mode restoration, and local-shard per-layer norm reductions. No
  diagnostic readout enters the objective.
- `query_state_export.py` is a separate human-gated exporter. It revalidates the
  terminal-primary checkpoint/control and pass receipt before all ranks enter the
  official FSDP full-state context, rejects local shards/training payloads and
  existing outputs, and emits no optimizer/scheduler/RNG or SFT2 authorization.

`FreshQueryStateDINOTeacher` now fronts online DINO with a strict process-local
memo keyed by `original_image_sha256 + exact DINO identity`. Cached targets are
immutable detached CPU clones, memory-accounted, and intentionally have no
checkpoint serialization path; a fresh process starts empty.

The production-path smoke preparation remains schema-distinct from all paths above:

- `query_state_smoke_config.py` owns `nimloth_sft1_query_state_smoke_v1`. The
  committed template is unresolved/unlocked; an external `preflight_locked=true`
  artifact resolves every operational field and exact row while remaining
  `launch_locked=false`; only a subsequent approval-bound launch artifact may
  enter CUDA. Resolved artifacts are hash-recorded and immutable outside the
  clean exact-commit source worktree, avoiding an impossible self-referential
  Git commit field.
- `query_state_smoke_runtime.py` binds the complete ordered source manifest,
  exact processor-rendered row descriptors, pre-wrap parameter inventory,
  detached per-optimizer-group gradient norms, model/optimizer/scheduler/RNG
  fingerprints, and immutable fresh/resume output ownership. It never selects
  rows dynamically or synchronizes parameter gradients manually.
- `query_state_smoke_train.py` is the real constructor owner: Qwen is loaded as
  full-language/frozen-vision K16, the complete root is moved and FULL_SHARD
  wrapped before frozen online DINO loads, then exactly one real row per rank is
  rebuilt for one update. Fresh and resume each run in a distinct process;
  resume must restore exact checkpoint fingerprints before its single update.

`configs/training/sft1/query_state_smoke_prep.yaml` remains explicitly unresolved
and cannot pass the CUDA gate. The thin `experiments/training/sft1/query_state_smoke.py`
entry point runs a read-only CPU `preflight` without CUDA, or one already
approved `torchrun` phase. Its two-line command manifest must match both canonical
fresh/resume child argv identities. It never submits Slurm, enables W&B, infers
resources, or mutates a config. There is still no
approved resolved config/command/output/resource identity, real Qwen/DINO/CUDA/
FSDP evidence, distributed model-quality validation, or SFT2 authorization.

`objective.py` owns the legacy state-interface-v2 canary's complete differentiable
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
data, teacher/cache/output identities, approved 12,836 valid train + 5 excluded
empty-CoT / 1,420 raw validation + 0 excluded / 1,413 external row counts.
External eligibility reproduces the pre-registered ID60 boundary: validation
record-initial, current, and next original-image SHA256 values must all be
absent from the corresponding train state lineage; this is stricter than
current-row-only overlap. The same schema locks all objective constants, dual
AdamW rates, three epochs, fixed contrastive-B2 budgets,
FSDP/checkpoint/report fields, and unknown-field rejection.
`launch_config.py` can publish one immutable resolved launch config only when
every commit/path/processor/W&B/topology value is explicit; it reloads the
persisted YAML and requires the same config identity before atomic publication.

`real_rows.py` reopens the hash-pinned exact 28-field pre-RL archive schema,
selects executed steps 0–3, hashes each original image, and derives exact
image/instruction groups. RL replay fields such as archived `action_log_probs`
are neither required nor invented. The authoritative instruction is the single
bounded `Human Instruction: ...\nDecide your next action(s).` span in the actual
first observation; its character offsets feed complete-context BPE selection.
`actions` and `think_texts` are cross-checks only, never aliases or fallbacks.
Exactly five train and zero validation rows with empty migrated CoT are counted
and excluded before selection; they are never accepted, repaired, or replaced.
The K8→K16 renderer permits only structural query replacement and locks every
archived CoT/action boundary. The approved context owns two system-prompt format
examples plus one initial-observation example, so a selected row requires all
verified non-assistant context examples plus `step_index + 1` trajectory pairs. Only the final pair is the current state; the
differentiable Qwen forward revalidates every pair and selects that final pair
without treating system examples as trajectory state turns.
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

Qwen rendering and the one-forward K16/action-boundary/optional selected-CE capability remain in
`nimloth.backbone.qwen25vl.state_training`. State rows require persisted real
assistant response/CoT provenance and exact contextual instruction BPE spans. Query-State roots require a complete archived action and final-assistant labels; the new Query-State adapter now carries those labels, while legacy v2 transport intentionally continues omitting them and retains its old assertions.

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
