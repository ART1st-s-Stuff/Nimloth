# Reconstruction decoder training

This package trains a **post-hoc** image decoder for WM diagnostics.  It freezes
Qwen, `StateProjector`, and the WM predictor checkpoint, then trains only
`WMImageDecoder` to reconstruct observations from projected WM states.

Reusable CFM and RCDM model code lives under `nimloth.recon`; this package owns
their training orchestration and command-line entrypoints.

## Direct Query-State diagnostics

The new SFT1 direct Query-State path is intentionally separate from the legacy
SFT2 adapters:

- `query_state_cache.py` validates a human-gated Query-State deployable bundle,
  rebuilds rows with the production pre-RL `SFT1V2Early4Row` indexer, loads the
  full actor/processor and direct head only from hash-bound bundle owner paths,
  then replays each original observation with its real archived assistant
  response/CoT. Its public builder accepts explicit device, model dtype,
  attention implementation, and maximum sequence length, but no owner-loader
  callback, in-memory actor/processor/direct head, state tensor, or row
  provenance. The internal production loader reads only validated
  `bundle/actor`, `bundle/processor`, and `direct_state.pt`, checks owner hashes
  before and after loading, enforces the exact vocabulary/token/K16 contract,
  recursively freezes/evaluates Qwen and the direct head, and stores the ordered
  canonical state as `[N,16,1024]`. Readers revalidate the live bundle, both
  source JSONLs, the complete SFT1 resume identity, split identity, owner hashes,
  image hashes, and shard hashes.
- Cache selection is explicit and fingerprint-bound: `all_train` contains every
  audited train row, while `external_validation` contains only rows whose live
  source index marks `external_eligible`. The formal source audit is 1420 raw
  validation rows, 1413 external rows, and 5 cross-split image hashes; these are
  reconstructed from source and persisted as audit evidence, never synthesized.
- `cfm_query_state.py` consumes only those schema-distinct `all_train` and
  `external_validation` caches, enforces row/image non-overlap, and binds both row sets and
  bundle/source/template/checkpoint identities into resume invariants. It flattens
  K16 only at the generic CFM boundary and trains a 16-token-conditioned RGB
  decoder. Every checkpoint comparison reuses the same explicitly preregistered
  validation noise seeds; the final full-split gate may add robustness seeds and
  retains per-seed plus aggregate correct-vs-global-shuffled sensitivity.
  RGB interpretation is published only by an authoritative checkpoint-bound path:
  it accepts no caller-provided reconstructions, originals, rows, config, or
  sensitivity evidence. It loads and validates the decoder plus optimizer, reloads
  the complete live validation split and recomputes all registered multi-noise evidence.
  The experiment must preregister a positive
  `--publication-min-shuffled-minus-correct` threshold (there is no project default),
  and every publication seed must meet it before any artifact directory is created or
  sampling begins. The path then deterministically selects/samples rows using
  checkpoint-bound ODE/noise settings and persists the gate evidence/verdict with
  original/output hashes and preprocessing metadata.

### Formal38 unsafe forensic diagnostics

The deployable path above is unchanged and has no `--allow-unsafe` mode. It
accepts only a human-gated terminal bundle and
`nimloth_query_state_reconstruction_cache_v1`. The separate forensic path is:

```text
Formal38 Job540589 unsafe_update_00001605 failure evidence + exact 8 rank shards
  -> load_query_state_forensic_model_for_debug (model tensors only)
  -> frozen/eval/inference-only final-current K16 + DirectSlotProjector
  -> nimloth_query_state_forensic_reconstruction_cache_v1
  -> forensic_query_state_features / cfm_forensic_query_state
```

`forensic_query_state_cache.py` owns a distinct
`unsafe_forensic_query_state` cache. It binds the actor failure, exact source,
config/run/control/failure identities, WS8 topology and shard hashes, plus each
original image, row, real archived response/CoT, prompt history, renderer,
template, and encoded-input identity. Cache state remains finite
`[N,16,1024]`. Publication atomically claims the destination with a durable
non-overwriting `mkdir`, moves validated shards, and commits `manifest.json`
last; readers reject every pre-manifest state and then revalidate live
source/checkpoint/image/shard provenance. This manifest-gated protocol works on
the production shared NFS without relying on unsupported
`renameat2(RENAME_NOREPLACE)` flags. The deployable and forensic readers
reject each other's schemas. No legacy `StateProjector`, WM, Value, grid encoder,
or synthetic/fixed CoT is accepted.

`forensic_query_state_production.py` is the only production composition for that
cache. Its strict JSON config has no checkpoint/data/output/distributed defaults:
it binds the clean integrated source commit, immutable Formal38 resolved config
SHA and identity, full forensic checkpoint identity, output, selection seed, and
exact WS8 NCCL topology. The module reconstructs ID176 plus the fresh no-bias
direct head from the Formal38 config, wraps the complete root with the same
FULL_SHARD policy **without constructing an optimizer or scheduler**, calls only
`load_query_state_forensic_model_for_debug`, recursively freezes/evaluates the
root, renders real archived-response rows, and feeds the collective-safe cache
builder. It imports or constructs no DINO teacher.

The later approved worker command must be exactly shaped as:

```bash
torchrun --nnodes=2 --nproc-per-node=4 --max-restarts=0 \
  -m nimloth.training.reconstruction.forensic_query_state_production \
  --config /absolute/path/to/locked-forensic-cache-config.json
```

The config still owns the exact rendezvous/launcher command through the separate
launch contract; this example does not choose remote paths, resources, output,
or authorize execution. The entry requires torchrun's
`TORCHELASTIC_MAX_RESTARTS=0` evidence and rejects any other world/rank topology.

`cfm_forensic_query_state.py` has separate typed `mechanics_only` and
`stage_b_diagnostic` owners. Both train only `TokenConditionedFlowUNet` with
16×1024 conditions and matching original observations; the unsafe producer is
absent from decoder optimizer/checkpoints. Stage A retains its exact 48/16,
64px, step10000 mechanics contract. Stage B strictly requires a fresh 128px
decoder with base channels 64, 4,000 random-batch steps (batch32), LR/WD 1e-4, clip1,
eval/save1000, seed 20260921, and full 12,836 `all_train` / 1,413
`external_validation` cache. Its final-only external publication gate uses
seeds 20260931/32/33, per-seed delta >=0.01 and aggregate shuffled/correct ratio
>=1.05 before producing 16 deterministic Euler50 external RGB examples.
Stage/cache/checkpoint invariants reject every cross-stage resume or reuse.

When the Stage B final step4000 publication gate fails, the separate
`cfm_forensic_posthoc_inspection.py` owner may produce only the human-requested
correct-condition RGB inspection under a new launch approval. It is hard-bound
to Job543457's exact final checkpoint, failed-gate metadata and summary hashes,
the exact Stage B cache manifest/fingerprint/selection, and the original fixed
16-row seed20260921/Euler50/chunk8 plan. The CLI exposes no row, seed, ODE,
training, resume, gate-override, optimizer, or W&B controls. It deserializes the
trusted checkpoint but constructs and loads only the frozen/eval decoder; the
serialized optimizer is validated as evidence and never materialized as a live
optimizer. Its distinct
`nimloth_query_state_forensic_cfm_posthoc_rgb_inspection_v1` manifest is committed
last after a non-overwriting destination claim. Strict readers reject the
publication schema, undeclared/symlinked/hash-drifted files, and every incomplete
pre-manifest output; post-manifest durability failure remains a typed committed
but unconfirmed terminal state. This inspection never changes the failed
publication verdict or any byte in the original Job543457 output.

### Formal38 oracle-ladder owners

`forensic_query_state_oracle_cache.py` creates a separate immutable typed cache
for the exact SFT1 DINOv2-large 4×4 teacher target. It reuses the already audited
Stage B ordered rows and roles but reads each matching **original archived
observation** directly; it must not read the 128px CFM target tensor or a decoder
PNG. The owner records the pinned model/revision/processor/grid identity,
source-state-cache fingerprint, selection and ordered row/image identities,
finite float32 `[N,16,1024]` shards, and per-shard plus total hashes. Publication
is non-overwriting and commits `manifest.json` last. It never accepts a supplied
tensor, alternate teacher, grid size, resize or synthetic/fixed CoT path.

`cfm_forensic_oracle_ladder.py` trains exactly one of three fresh cells:
`token_oracle`, `spatial_state`, or `spatial_oracle`. The fourth cell,
`token_state`, is the immutable Stage B final-step4000 baseline and is rejected
by the trainer. Every fresh cell is decoder-only and shares the exact
12,836/1,413 rows, batch32, LR/WD 1e-4, clip1, seed20260921, eval/save1000 and
final-step4000 budget. Checkpoints bind the cell, condition/decoder family,
cache/selection/row identities, architecture, optimizer, RNG, fixed times,
noise seeds, sample selection and Euler50 contract; cross-cell/family resume and
best/intermediate checkpoint selection fail closed. The authoritative runner
requires explicit `WANDB mode=online`, project `nimloth-recon`, and exact run
ID/name; environment-selected offline/disabled tracking is rejected.

`nimloth.eval.query_state_oracle_ladder` then read-only evaluates the complete
2×2 matrix. It requires the immutable baseline SHA plus three complete matching
final4000 checkpoints; computes full external random-time and fixed-time
correct-vs-global-shuffled flow evidence; samples a deterministic identity-only
256-row plan under all three registered noise seeds; scores RGB L1/RMSE and
source-versus-generated DINO cosine/MSE; averages seeds inside each row before
paired row bootstrap; and emits the fixed 16-row original-plus-four-cell contact
sheet. That visual set is strict-read byte-for-byte from ID198's immutable
summary/external report hashes and ordered row/image pairs, not reconstructed
from indices alone. Source DINO is the oracle-cache tensor from canonical original-image
preprocessing. DINO rescoring of generated 128px sRGB is explicitly secondary
and is never relabelled as the teacher target. Reports are manifest-last,
hash-bound, forensic-only and nondeployable.

No oracle-ladder implementation, cache, checkpoint or visual result changes the
Formal38 actor failure, establishes SFT1 quality, authorizes SFT2, or proves that
DINO 4×4 is pixel-invertible. Building the oracle cache, training any cell, or
running the evaluator remains an experiment requiring a complete exact launch
contract and separate explicit launch approval.

### Direct-DINO spatial-grid ceiling

`dino_grid_ceiling_cache.py` does not accept or open an SFT1 state cache. It
metadata-only reads the immutable direct-DINO grid4 cache to recover its audited
ordered original-image rows, then sends each original observation through the
pinned frozen DINOv2-large owner exactly once. The fingerprinted processor uses
shortest-edge256 plus center-crop224 and therefore produces a native16×16 patch
map at patch14; `model.config.image_size=518` is not runtime geometry. Grid4 and
grid8 are independently adaptive-average-pooled from native16, while grid16 is
stored unpooled. Direct grid4 must be exactly equal to every immutable grid4 row
before the manifest-last cache may publish. Only float32 grid8 `[N,64,1024]`
and grid16 `[N,256,1024]` views are stored; chained pooling is excluded.

`cfm_dino_grid_ceiling.py` has only `spatial_dino8` and `spatial_dino16` cells.
It accepts no state-cache CLI argument and trains only equal-capacity
`SpatialConditionedFlowUNet` decoders under the matched Stage B image rows,
batch32, seed20260921 and exact final-step4000 budget. Cache/view/grid/source,
optimizer, RNG, exact W&B entity/project/run and output identities are checkpoint
invariants; W&B finish records explicit success/failure status. Cross-grid resume,
best/intermediate selection and post-result extension fail closed.

These owners are direct-DINO representation-decodability probes. They never load
SFT1, Query-State, actor, SFT2 or World Model tensors, and their cache/training
commands remain separately gated experiments rather than evidence that any
learned state can predict the DINO views.

The Stage B cache is rebuilt from the live audit without a caller row mask,
requires zero train/external image overlap, uses the same WS8 padded collective
schedule, and publishes fixed bounded 2,048-record shards with manifest-last NFS
semantics. Direct metrics cover all rows; only deterministic 16-row visual
samples are retained, with PCA/global scale fit on `all_train` and external rows
transform-only. Neither stage establishes safe/deployable state quality, and
implementation still does not grant cache/GPU launch approval.

Evidence is interpreted in this order: Formal38's actor safety failure remains
valid; direct frozen-DINO metrics describe the unsafe state's feature relation;
CFM sensitivity describes whether the decoder uses that state; sRGB strips and
contact sheets are human-readable examples only. None may resume Formal38,
promote update1605, override actor failure, select SFT1, mark a cache deployable,
or authorize SFT2.

Direct DINO feature maps and metrics remain the primary state diagnostic under
`nimloth.eval.query_state_features` (or the strict forensic adapter for the
unsafe cache). CFM RGB images are a secondary human-readable probe:
decoder/domain failure can make even oracle states reconstruct poorly, so image
quality cannot override direct feature evidence or select an SFT1 checkpoint.

CPU tests cover manifest/cache primitives, real-row wiring with fake model
owners, and fail-closed loader gates. They do not prove real Hugging Face owner
load, WS8 FSDP/NCCL, or GPU extraction. The deployable path requires an exact
human-gated terminal bundle; the forensic path instead requires the exact unsafe
Formal38 shards/failure evidence and remains nondeployable.

Actual feature extraction, cache building, CFM training, sampling, evaluation,
or W&B/remote/GPU execution is an experiment. It requires the applicable exact
owner above, a separate complete experiment launch contract, and explicit launch
approval; importing these modules or implementing their code grants none of
those permissions.

The decoder is not part of the SFT2/RL objective.  Use it to compare:

- `decoder(s_next)` vs next image: decoder/oracle upper bound.
- `decoder(wm_predictor(s_t, a_t))` vs next image: WM predictive reconstruction.
- `decoder(s_t)` or shuffled actions vs next image: baselines.

## Train

```bash
python -m nimloth.training.reconstruction.cli \
  --model /path/to/export_best_hf \
  --state-proj-checkpoint /path/to/best/state_proj.pt \
  --wm-checkpoint /path/to/best/wm_predictor \
  --train-jsonl /path/to/train.jsonl \
  --val-jsonl /path/to/val.jsonl \
  --output-dir outputs/experiments/training/reconstruction/<date>/<name>
```

## Eval an existing decoder

```bash
python -m nimloth.eval.reconstruction \
  --model /path/to/export_best_hf \
  --state-proj-checkpoint /path/to/best/state_proj.pt \
  --wm-checkpoint /path/to/best/wm_predictor \
  --decoder-checkpoint outputs/.../best \
  --val-jsonl /path/to/val.jsonl \
  --output-dir outputs/experiments/training/reconstruction/<date>/<name>/eval
```

## Train RCDM visualization model

RCDM is a heavier diffusion-based alternative to `WMImageDecoder`. It trains a
UNet from `external/RCDM` conditioned on SFT2 WM states. It is still post-hoc and
does not change SFT2/RL losses.

```bash
python -m nimloth.training.reconstruction.rcdm_sft2 \
  --model /path/to/export_best_hf \
  --state-proj-checkpoint /path/to/best/state_proj.pt \
  --wm-checkpoint /path/to/best/wm_predictor \
  --train-jsonl /path/to/train.jsonl \
  --val-jsonl /path/to/val.jsonl \
  --output-dir outputs/experiments/training/reconstruction/<date>/<rcdm_name> \
  --wandb-run-name <rcdm_name> \
  --state-cache-dir outputs/experiments/training/reconstruction/cache/<cache_name> \
  --build-state-cache
```

`--state-cache-dir` stores compressed shards of `StateProjector(Qwen
<|latent_state|>)` plus image paths.  This is the preferred full-run path: after
cache build, RCDM training no longer loads or runs Qwen.  The cache defaults to
`float16` state embeddings and `gzip` compression.

Resume the latest RCDM checkpoint in the same output directory:

```bash
python -m nimloth.training.reconstruction.rcdm_sft2 \
  ...same flags as above... \
  --resume
```

Use `--resume-checkpoint outputs/.../training_state_000001000.pt` to resume a
specific checkpoint. W&B uses `wandb_run_id.txt` in the output directory when
`--resume` is set.

## Train direct-state conditional flow matching

CFM is a lighter alternative to RCDM. It reuses a completed RCDM state cache,
trains only a token-conditioned UNet velocity field, and reports both validation
flow MSE and shuffled-condition sensitivity. Qwen, `StateProjector`, and the WM
predictor remain frozen.

```bash
python -m nimloth.training.reconstruction.cfm_sft2 \
  --state-cache-dir outputs/.../state_cache \
  --source-checkpoint outputs/.../sft2/epoch_002 \
  --wm-checkpoint outputs/.../sft2/epoch_002/wm_predictor \
  --output-dir outputs/.../cfm/<run_name> \
  --latent-token-count 8 \
  --epochs 10 --batch-size 32 --lr 1e-4 \
  --wandb-project nimloth-recon --wandb-run-name <run_name>
```

Use `--resume` to load the latest `checkpoint_*.pt` in the output directory.
The cache fingerprint, model config, data sizes, k, optimizer hyperparameters,
and seed must match. The final gate saves 5-step and 50-step ODE contact sheets
for current-state and WM-predicted-next reconstruction.

## Sample from RCDM

Sample from a trained RCDM checkpoint:

```bash
python -m nimloth.eval.rcdm_reconstruction \
  --model /path/to/export_best_hf \
  --state-proj-checkpoint /path/to/best/state_proj.pt \
  --wm-checkpoint /path/to/best/wm_predictor \
  --rcdm-checkpoint outputs/.../ema_0.9999_000100000.pt \
  --metadata outputs/.../metadata.json \
  --val-jsonl /path/to/val.jsonl \
  --output-dir outputs/.../rcdm_samples \
  --timestep-respacing 100
```
