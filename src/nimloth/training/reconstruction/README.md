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

Direct DINO feature maps and metrics remain the primary state diagnostic under
`nimloth.eval.query_state_features`. CFM RGB images are a secondary human-readable
probe: decoder/domain failure can make even oracle states reconstruct poorly, so
image quality cannot override direct feature evidence or select an SFT1
checkpoint.

CPU tests cover manifest/cache primitives and fail-closed loader gates; they do
not claim a successful real Hugging Face owner load. That requires future
integration against the exact terminal bundle under its approved experiment.

Actual feature extraction, cache building, CFM training, sampling, evaluation,
or W&B/remote/GPU execution is an experiment. It requires a terminal bundle that
passed the human SFT1 gate, a separate experiment contract, and explicit launch
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
