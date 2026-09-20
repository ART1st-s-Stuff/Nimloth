# Reconstruction and Evaluation

## Ownership

`nimloth.recon` contains post-hoc CFM/RCDM reconstruction models that consume frozen Nimloth state representations. `nimloth.training.reconstruction` owns decoder training/evaluation entry behavior. `nimloth.eval` owns model-dependent offline evaluation and reconstruction diagnostics; online environment rollout remains in the environment/Agent path.

Reconstruction must not silently enter SFT2/RL optimization, alter the state source, or substitute a service/decoder object for the real model graph. Record the exact source checkpoint/state representation and whether inputs are oracle, predicted, copied, shuffled-action, cached, or re-encoded.

For paired spatial CFM probes, train independent state-conditioned and DINO-conditioned
decoders on the same unique training-observation keys with matched model and optimizer
budgets. Preserve the native spatial grid; do not flatten spatial conditions into one
token. Validate sealed cache provenance and resolve RGB targets from the same observation
keys. Validation images never enter decoder fitting.

## Spatial + CLS CFM contract

### 1. Scope / Trigger

Use this contract when a post-hoc CFM consumes a state with an `8x8` spatial grid
followed by one DINO CLS-aligned global token. The decoder is an evaluation probe; it
does not update Qwen, the state projector, WM, ValueHead, or OutcomeHead.

### 2. Signatures

Training uses `experiments.training.sft.stage3.cfm_decoder_probe` with
`--decoder-family spatial_cls_grid_v1`, an explicit `--condition state|dino`, sealed
train/eval caches, and their matching JSONL assets. Evaluation uses
`experiments.training.sft.stage3.evaluate_cfm_decoder_probe` and checkpoints written by
that training command.

### 3. Contracts

- Cache conditions have shape `(N,65,1024)` and identity metadata contains the exact
  producer `state_layout`: `spatial_grid_size=8`, `global_tokens=1`, and
  `global_role=dino_cls`.
- Tokens `[0,64)` remain an ordered row-major `8x8` grid. Token `64` is normalized and
  projected by a separate global branch; it never receives or occupies a 2D coordinate.
- State-conditioned and DINO-conditioned decoders are trained independently from
  scratch with the same observation keys, optimizer budget, seed policy, and validation
  split. The checkpoint records `decoder_family=spatial_cls_grid_v1`.
- `best.pt` is selected only by the declared validation loss. `latest.pt`, `final.pt`,
  and resume logs must agree on identity and step.
- Paired CLS evaluation keeps spatial tokens, RGB target, noise, and sampler settings
  fixed while comparing correct, zero, and cross-sample shuffled CLS conditions.

### 4. Validation & Error Matrix

- K64 or non-1024 cache -> reject the spatial+CLS decoder request.
- Missing, malformed, or different `state_layout` -> reject; never infer CLS by shape.
- Cache/JSONL split hash mismatch or missing provenance -> reject.
- Shared RGB hashes between reconstruction train and eval -> reject.
- Checkpoint schema and decoder-family mismatch -> reject.
- Resume identity, checkpoint step, or metric-log continuity mismatch -> reject.

### 5. Good / Base / Bad Cases

- Good: sealed K65 caches with explicit layout, disjoint RGB images, and two matched
  state/DINO decoder runs.
- Base: historical K64 spatial checkpoints remain loadable only as
  `spatial_grid_v1`; they do not gain an inferred CLS branch.
- Bad: append an arbitrary 65th vector to a K64 cache, flatten K65 as one condition, or
  compare CLS variants with different sampling noise.

### 6. Tests Required

Tests assert spatial/CLS routing and gradients, correct/zero/shuffled paired conditions,
strict cache layout validation, legacy K64 checkpoint loading, schema/family mismatch
failure, best-checkpoint selection, and exact resume log/checkpoint agreement.

### 7. Wrong vs Correct

Wrong: reshape all 65 tokens or identify the last token as CLS only because `K=65`.

Correct: validate the producer layout, reshape only tokens `[0,64)` to `8x8`, and send
token `64` through the coordinate-free global branch.

Compare GT and predicted conditions with identical pure-noise tensors and sampler
settings. Label frozen Stage2 GT, online Stage3 GT and WM predictions separately.
Feeding a projected WM state to a DINO-trained decoder is a cross-distribution readout,
not proof that the WM predicts teacher DINO directly. Report repeated-window weighting
and unique-observation counts; image quality alone does not establish dynamics quality.

## Evaluation evidence

Every report names:

- exact checkpoint/component and configuration;
- dataset asset, verified split, overlap evidence, and sample/statistical unit;
- command/commit/output provenance;
- metric definition and aggregation;
- failures/exclusions and validity limits.

Static dataset success rates are not model evaluation. Results from one eval set do not generalize to another. Training rollouts, smoke samples, reconstruction image quality, WM MSE, action-value loss, average reward, and held-out success answer different questions and must not be relabeled.

## Sources

- [`recon/README.md`](../../../src/nimloth/recon/README.md)
- [`recon/cfm/README.md`](../../../src/nimloth/recon/cfm/README.md)
- [`recon/rcdm/README.md`](../../../src/nimloth/recon/rcdm/README.md)
- [`training/reconstruction/README.md`](../../../src/nimloth/training/reconstruction/README.md)
- [`eval/README.md`](../../../src/nimloth/eval/README.md)
- [`configs/eval/`](../../../configs/eval/)

Evaluation or reconstruction execution is an experiment and requires the experiment task/lifecycle contract.
