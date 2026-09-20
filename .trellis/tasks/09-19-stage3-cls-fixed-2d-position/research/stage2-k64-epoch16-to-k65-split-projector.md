# Research: Stage2 K64 epoch16 to K65 split-projector continuation

- Query: Inspect how to initialize a K65 Stage2 continuation directly from the retained K64 Stage2 epoch16 while replacing the shared projector with independent spatial and CLS projectors; do not replay or consume Stage3.
- Scope: internal
- Date: 2026-09-20

## Findings

### Correct source and experiment boundary

- The requested source is the committed Stage2 checkpoint
  `/mnt/nimloth/outputs/experiments/sft2-deepsight-full/20260914_epoch15_projector_lr8e5_continue/train/epoch_016`, not any K64 Stage3 checkpoint. The retained handoff identifies it as epoch 16 / global step 432 with K64, row-major 8x8 DINOv2-large state, state dimension 1024, Qwen/projector hidden dimension 2048 (`.local/handoff-20260914-stage3-epoch16-preflight.md:3-6`).
- Its last recorded validation total/LM/DINO losses are `1.6330998838 / 0.4440948665 / 0.5945025086`; it was produced by the eight-rank FSDP Stage2 chain (`.local/handoff-20260914-1010-stage2-projector4x.md:72-80`). The later `resume_step_00000446` belongs to an incomplete epoch 17 with no epoch validation (`.local/handoff-20260914-1010-stage2-projector4x.md:83-89`), so epoch16 is the clearer fixed initialization boundary for the new architecture.
- This is a Stage2 weight migration followed by a new evaluation-only Stage2 run. Because both vocabulary size and projector parameter topology change, it cannot faithfully resume epoch16's optimizer, scheduler, cursor, RNG, or convergence state. The new run should record `parent_checkpoint=.../epoch_016`, `parent_epoch=16`, and start its own optimizer step/epoch counters. Calling it “Stage2 continuation from epoch16” is accurate for model lineage; calling it an exact resume is not.
- The new run should keep the previously used answer-view train/eval files, K65 DINO cache, DINO weight 2, effective batch 64, seed, Query LR `1e-4`, projector LR `8e-5`, and DINO convergence rule unless the human changes them. It remains evaluation-only because the old split has known upstream exposure and the architecture did not exist from Stage2 epoch1.

### Existing Stage2 path is close but currently uses one shared projector

- `QueryAlignmentModel.build()` always creates one `SharedSlotProjector` with `grid_tokens=objective.state_tokens`; therefore `include_global_token=True` currently sends all K65 slots through the same MLP (`src/nimloth/training/sft/stage2/model.py:51-76`).
- The forward already validates ordered contiguous K65 Query positions and exact DINO target shape (`src/nimloth/training/sft/stage2/model.py:95-155`). It already separates spatial and CLS MSE with `GridStateLayout`, normalizes each independently, and sums them (`src/nimloth/training/sft/stage2/model.py:185-201`). No new loss implementation or cache tensor order is needed.
- `query_projector_only` already freezes Qwen/vision/LM head/protocol rows, installs input-only FP32 masters for all Query rows, trains the projector in FP32, and uses a fresh optimizer (`src/nimloth/training/sft/stage1/trainer.py:1122-1144`; `src/nimloth/training/sft/stage2/README.md:31-38`). This is the right training scope to preserve for the split-projector variant.
- Current optimizer construction collapses every parameter whose path contains `projector` into one group (`src/nimloth/training/sft/stage1/trainer.py:357-419`). The two branches would train, but separate named spatial/global groups are preferable for identity, auditing, per-branch update checks, and future independent LRs. Both groups should initially use the same approved projector LR `8e-5`; introducing a different CLS LR would add an unapproved experimental variable.
- Existing convergence behavior is already correct: `query_projector_only` monitors `validation_dino_loss`, which is spatial MSE plus CLS MSE (`src/nimloth/training/sft/stage1/trainer.py:347-354`). Spatial and CLS validation values are separately reduced and logged (`src/nimloth/training/sft/stage1/trainer.py:305-345`).

### Reusable split projector and the Stage2-specific gap

- `SplitSpatialGlobalProjector` already implements the desired `(B,65,H) -> (B,65,1024)` interface. It routes the first 64 slots through `spatial`, the last slot through `global_projector`, and concatenates in row-major-spatial-then-global order (`src/nimloth/wm/grid.py:59-109`).
- `SplitSpatialGlobalProjector.from_k64_shared()` loads the same K64 shared state strictly into both branches (`src/nimloth/wm/grid.py:130-151`). This satisfies the desired initialization: spatial exactly preserves the epoch16 mapping, while the new CLS projector begins as an exact copy and can then adapt independently.
- The existing loader `load_k64_projector_for_k65_migration()` is Stage3-specific: it requires `state_proj.pt` plus a Stage3 `training_state.pt` and reads `latent_token_count` (`src/nimloth/wm/grid.py:154-199`). Stage2 epoch16 instead owns `slot_projector.pt` and `grid_state_config.json`. Stage2 needs a separate strict loader, or a generalized loader with explicit artifact-kind input; it must not pretend a Stage2 checkpoint is Stage3.
- Stage2 metadata currently hard-codes `shared_slot_projector=True` and saves a bare `slot_projector.pt` (`src/nimloth/training/sft/stage2/model.py:229-283`). The new path needs `projector_layout=split_spatial_global_v1`, `shared_slot_projector=false`, explicit spatial/global dimensions/order, source checkpoint, and `global_initialization=copy_of_spatial_v1`. Normal resume must construct the split module before strict state loading. A shared K64 checkpoint must only enter through the explicit one-time migration path.
- `load_sft1_slot_projector()` also requires `shared_slot_projector=True` and returns `SharedSlotProjector` (`src/nimloth/wm/grid.py:202-253`). When this Stage2 output later feeds Stage3, that loader or the Stage3 constructor must explicitly recognize the split Stage2 schema. This is needed for downstream compatibility, but Stage3 training itself is not part of the present run.

### Token and selected-row migration semantics

- K64 to K65 must append exactly one Query token after the 64 spatial Query tokens. Current `global_query_only` already rejects anything except exactly one newly registered token and initializes it from the spatial Query mean (`src/nimloth/training/sft/stage1/trainer.py:1017-1035`). That initialization should be reused.
- The epoch16 run used the full-language selected-row schema, whose authoritative FP32 sidecar contains both input and output Query/protocol tables. The direct K64→K65 `query_projector_only` path currently installs one input-only K65 table and then calls `restore_selected_rows_subset()` (`src/nimloth/training/sft/stage1/trainer.py:1122-1142`). That helper only accepts a one-table input-only source, so it cannot correctly consume epoch16's two-table sidecar (`src/nimloth/backbone/selected_token_rows.py:343-365`).
- A dedicated migration helper is required. It should:
  1. require epoch16 `selected_token_rows.pt` and identify the input and output source tables;
  2. verify 64 source Query IDs are the exact prefix of the 65 target IDs and all source protocol IDs are unchanged;
  3. copy the 64 input Query FP32 masters exactly into the K65 input-only master;
  4. initialize the new CLS input master as the FP32 mean of those 64 rows;
  5. verify the frozen dense output/protocol rows loaded from the checkpoint match the authoritative sidecar, then initialize only the new dense CLS output row by the existing mean rule;
  6. save the new run's exact 65-row input-only sidecar.
- `restore_selected_rows_with_appended_query()` already expresses the strict prefix, protocol-identity, shape/dtype, finiteness, and FP32-mean rules for a two-table target (`src/nimloth/backbone/selected_token_rows.py:260-340`). Its validation pattern is reusable, but its output schema is wrong for input-only `query_projector_only`; do not install a trainable output table merely to reuse it.
- All existing 64 Query rows must be shown equal to the source FP32 sidecar before the first update. Frozen protocol rows, Qwen backbone, vision stack, and LM head must remain outside the optimizer. The newly added output row is necessary for vocabulary shape consistency but remains frozen in this diagnostic.

### Checkpoint and optimizer semantics

- Checkpoint saving already atomically stores dense model weights, `selected_token_rows.pt`, `slot_projector.pt`, `grid_state_config.json`, optimizer/scheduler, per-rank RNG, cursor, and identity (`src/nimloth/training/sft/stage1/checkpoint.py:116-216`; `src/nimloth/training/sft/stage2/model.py:265-283`). It can support the new module after metadata and strict restore logic are generalized.
- Use a fresh AdamW with three disjoint named groups: `state_proj_spatial` at `8e-5`, `state_proj_global` at `8e-5`, and `query_rows` at `1e-4`, with zero weight decay for Query rows. The optimizer parameter union must equal all and only trainable parameters. Do not load epoch16 optimizer moments; there is no parameter correspondence for the new CLS row and global projector branch.
- Save migration provenance in both `grid_state_config.json` and training identity so it survives normal resume. Required fields should include source path and hashes, source K64 schema, target K65 layout, projector-copy rule, selected-row rule, DINO cache fingerprint, `optimizer_initialization=fresh_adamw_v1`, and `evaluation_only=true/formal_stage2=false`.
- A normal resume of the new K65 split run should restore exact split projector weights, 65-row sidecar, optimizer groups/state, scheduler, RNG, data cursor, and convergence state. It must reject a shared K64 `slot_projector.pt`, a shared K65 checkpoint, missing migration provenance, reordered Query IDs, or a different cache fingerprint.

### Cache and data implications

- No new DINO computation is implied by changing the projector. Stage2 already consumes a K65 v2 cache containing true frozen DINO CLS plus K64 spatial grid. The v2 loader verifies 64 spatial tokens, one global token, ordering, teacher/processor identity, source JSONL hashes, image byte hashes, shard hashes, and build commit (`src/nimloth/backbone/dino_grid.py:576-650`).
- The previously used cache path was `/mnt/nimloth/outputs/experiments/stage3-cls-fixed2d/20260919_canary/dino_stage2_state_v2` (`.local/stage2_epoch8_query_projector_only_dino2_r1.contract.json:12`). Task progress says this cache was retained when obsolete K65 checkpoints were deleted. It still requires a fresh live manifest/hash audit before launch because server state is mutable.
- Stage2 tokenization/collation remains online and one full trajectory is processed in one teacher-forced forward; the projector split does not require a new Qwen preprocess cache. Data must remain the same 1709-train/193-val answer-view split for comparability. Its known exposure means the result is mechanism evidence, not unseen-task generalization.

### Concrete affected files

- `src/nimloth/training/sft/stage1/cli.py`: add an explicit split-projector K64→K65 Stage2 migration mode/flag and source checkpoint argument; require K65, DDP, BF16 frozen dense tables, evaluation-only, convergence policy, and no exact-resume/continuation flags.
- `src/nimloth/training/sft/stage1/trainer.py`: construct the split Stage2 model, perform strict token-row migration, record parent/provenance, build three optimizer groups, and distinguish initialization from resume.
- `src/nimloth/training/sft/stage2/model.py`: allow `SplitSpatialGlobalProjector`, select it only for the explicit mode, emit split metadata, save/load strict split states, and keep existing spatial/CLS loss routing.
- `src/nimloth/backbone/selected_token_rows.py`: add strict two-table K64 source to one-table K65 input-only migration without BF16 recovery.
- `src/nimloth/wm/grid.py`: add a strict Stage2 `slot_projector.pt`/`grid_state_config.json` K64 loader that returns a copied split projector; do not reuse the Stage3 artifact loader implicitly.
- `src/nimloth/training/sft/stage2/README.md`: document that this is a fresh-optimizer evaluation-only Stage2 lineage continuation, including exact train/freeze groups and fail-closed resume rules.
- Downstream follow-up: `src/nimloth/training/sft/stage3/trainer.py` and/or `load_sft1_slot_projector()` must accept the split Stage2 schema before a later Stage3 run can consume it.

### Focused tests required

- Stage2 artifact migration loads a strict K64 `slot_projector.pt`; both target branches initially equal the source bitwise, and spatial output on identical K64 hidden states equals the parent output.
- Spatial-only loss produces gradients only in the spatial projector; CLS-only loss produces gradients only in the global projector; combined loss reaches both and the corresponding Query input rows.
- K64 full selected-row sidecar to K65 input-only conversion preserves all 64 input Query FP32 rows exactly, appends one mean-initialized CLS row, verifies source output/protocol identity, and rejects reordered/missing/extra IDs or BF16-only fallback.
- Optimizer group names and membership are exactly `state_proj_spatial`, `state_proj_global`, and `query_rows`; groups are disjoint and exclude Qwen, vision, LM head, protocol rows, DINO teacher, and cache tensors.
- `grid_state_config.json` and `training_state.pt` round-trip split layout and migration provenance. Normal split resume succeeds; shared K64/K65, missing provenance, tampered source, and layout/cache mismatch fail closed.
- K65 forward uses ordered `[spatial_00..63, cls]`, routes spatial/CLS target slices to the correct branch, logs both components, and preserves `validation_dino_loss=spatial+cls` convergence.
- One-update production canary proves eight-rank DDP, finite losses/gradients, both projector branches and Query rows update, frozen parameters do not, and a complete resume checkpoint reloads in a fresh process.

## Files found

- `.local/handoff-20260914-1010-stage2-projector4x.md` — authoritative retained history for the epoch15→16 continuation and paused epoch17 boundary.
- `.local/handoff-20260914-stage3-epoch16-preflight.md` — retained K64 epoch16 model/state dimensions and exact checkpoint path.
- `src/nimloth/training/sft/stage2/model.py` — current Stage2 shared-projector construction, spatial/CLS loss, metadata, and projector persistence.
- `src/nimloth/training/sft/stage1/cli.py` — shared Stage1/Stage2 CLI and tuning-mode gates.
- `src/nimloth/training/sft/stage1/trainer.py` — Stage2 model/token setup, optimizer grouping, convergence, save, and resume behavior.
- `src/nimloth/backbone/selected_token_rows.py` — authoritative FP32 selected-row schemas and exact restore helpers.
- `src/nimloth/wm/grid.py` — shared and split projector implementations plus Stage3-specific migration loader.
- `src/nimloth/backbone/dino_grid.py` — K65 spatial/CLS cache schema and lineage validation.
- `src/nimloth/training/sft/stage1/checkpoint.py` — atomic full-state early-stage checkpoint and resume contract.
- `tests/training/sft/stage2/` and `tests/training/sft/test_stage3_k64_k65_migration.py` — existing Stage2 query/projector and split-migration test patterns.

## Related specs

- `.trellis/spec/domains/world-model-and-training.md` requires explicit source/target axes, objective recipients, gradient paths, train/freeze groups, checkpoint ownership, and distributed topology.
- `.trellis/spec/experiments/task-contract.md` requires exact remote checkpoint/data/cache identity, a unique output, resource preflight, and a separate launch contract for a run over ten minutes.
- `.trellis/spec/experiments/outputs-checkpoints-and-evidence.md` distinguishes initialization from faithful resume and requires explicit best/last and recovery semantics.

## External references

- None. This decision is governed by repository code, retained experiment evidence, and project specs.

## Caveats / Not Found

- A live read-only SSH audit of epoch16 was attempted twice but the current environment's SSH agent was rejected with `Permission denied (publickey)`. The source facts above are therefore from retained, project-local handoff evidence and may be stale. Before implementation launch, verify live existence, `COMMITTED`, shard/index completeness, `slot_projector.pt`, `grid_state_config.json`, `selected_token_rows.pt`, `training_state.pt`, hashes, and free disk.
- The current task PRD/design still has an authoritative 2026-09-20 override selecting K64 Stage3 replay. The user's newest instruction supersedes that override: task artifacts must be revised to state Stage2 epoch16 → K65 split-projector Stage2, and the stopped Stage3 replay must remain unused.
- The exact Qwen/protocol selected-row contents of epoch16 were not live-inspected in this research pass. Implementation must fail closed if the expected full-language two-table FP32 sidecar is absent or incompatible; it must not fall back to rounded dense rows silently.
