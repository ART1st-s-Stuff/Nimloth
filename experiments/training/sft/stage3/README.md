# Stage3 data preparation

`prepare_eval_records.py` consumes a hash-pinned VAGEN raw-record manifest and verifies
record bytes, each image's hash/mode/size, transcript/history alignment, executed actions,
rewards and outcomes. Image paths are relative to their raw record file when not absolute.
It uses the existing source action conversion followed by the unchanged B prompt rewrite
in `prompt_conversion.py`. The shared `rollout.tail_drop` converter owns finite-horizon
returns and removing the final transition with no retained next observation.

```bash
python -m experiments.training.sft.stage3.prepare_eval_records \
  --manifest records_manifest.json --latent-token-count 64 \
  --max-action-horizon 20 --check-only
```

Replace `--check-only` with `--output-root NEW_DIRECTORY` to write `data.jsonl`,
`manifest.json` and `COMMITTED`. The output directory must not exist. Failure writes
rejection evidence without a completion marker. The manifest preserves the original raw
record hashes and every verified image identity; record and image sources are not modified.

`prepare_transition_cache.py` builds one split with the production CPU cache API,
without loading Qwen weights. Specify source/output/processor, token count, max
length and max pixels explicitly. It refuses existing outputs and never filters
failed trajectories. Run separately for `preprocess/train` and `preprocess/val`.
Optionally pass `--reuse-image-cache OLD_SPLIT_DIRECTORY` to validate and hardlink
unchanged image shards into the new output. Text and labels are rebuilt with the
current encoding contract; old transitions are never reused. Source identity,
image settings, ordered indices, grids and shard layout must match. This saves
disk space but still reads the image shards for validation and SHA256 recording.
The source remains unchanged, and the two directories must not overlap.

## Training activation storage

`train.activation_offload` (CLI `--activation-offload` / `--no-activation-offload`)
selects PyTorch `save_on_cpu(pin_memory=True)` for the primary and SIGReg training
forwards. It defaults to false, including `action_outcome_k64_h1_t4.yaml` for both
arms. Tensors saved for backward are copied to CPU with their original dtype and
restored to their original device when needed. This includes saved parameter views
when autograd needs them; live model parameters and optimizer state remain on GPU.
The forward objectives, batch, gradients and distributed reductions are unchanged.

This reduces retained GPU activation storage at the cost of pinned host memory and
GPU/CPU transfers, which can slow each update. It complements gradient checkpointing;
it does not offload the optimizer or run backward on CPU. The enabled setting is
recorded in checkpoint training invariants. GPU memory headroom and runtime still
require validation on the actual training batch and distributed model.

## Frozen-WM diagnostic

`frozen_wm_diagnostic.py` isolates the production Stage3 world-model predictor
from representation learning. It does not change formal Stage3 defaults.

Run the production Stage3 entry point in `--eval-only` mode with
`--frozen-wm-cache-dir` and explicit `--frozen-wm-cache-split train|eval`
once for the official train split and once for the
official eval split. The export stores each trajectory's ordered Stage2 state
grids, real DINO grids, and actions once; overlapping T=4 windows are derived
later by index. Seal each fresh directory before training:

```bash
python experiments/training/sft/stage3/frozen_wm_diagnostic.py seal-cache \
  --directory /path/to/fresh/train-cache --expected-ranks 8
```

Run the two explicit diagnostics in separate fresh output directories:

```bash
python experiments/training/sft/stage3/frozen_wm_diagnostic.py train \
  --train-cache /path/to/train-cache --eval-cache /path/to/eval-cache \
  --output /path/to/fresh/output --mode stage2_state

python experiments/training/sft/stage3/frozen_wm_diagnostic.py train \
  --train-cache /path/to/train-cache --eval-cache /path/to/eval-cache \
  --output /path/to/other/fresh/output --mode dino
```

Defaults preserve H=1/T=4, the production predictor architecture, WM learning
rate `3e-4`, effective trajectory batch 64, gradient clipping at 1, and 46
updates. The original WM-loss cosine ramp from 0.1 to 1 over the first 30% of
updates is retained. A trajectory microbatch controls memory while window-count weighting
keeps each optimizer update equal to the global mean over its 64-trajectory
group. Only the predictor exists in the optimizer. Checkpoints at steps
1/5/10/final contain predictor, optimizer, RNG, immutable cache identities, and
resume position.

Metrics are reported per horizon against the fixed mode target and real DINO.
Baselines include the per-horizon train-window mean, current mode input copy,
and current real-DINO copy. Donor perturbations use a different trajectory ID
but are not task-matched because the frozen cache contains no task metadata.

## Spatial+CLS reconstruction decoder

`cfm_decoder_probe.py` can fit a post-hoc reconstruction decoder from a sealed
K65 cache without loading or updating Qwen, the state projector, world model,
ValueHead, or OutcomeHead. Select the explicit layout with
`--decoder-family spatial_cls_grid_v1`; the default remains the legacy K64
`spatial_grid_v1` family. The K65 family always interprets rows 0--63 as the
row-major `8x8` spatial grid and row 64 as the DINO CLS token. It rejects K64
caches instead of padding or pooling a replacement global token. The sealed
cache must also record the exact `row_major_spatial_then_global` state layout;
K65 shape alone is not accepted as proof of slot ordering.
If a train observation is byte-identical to any validation RGB target, the
loader removes that row from decoder fitting, records the exclusion count and
canonical hash-set digest, and verifies that no RGB overlap remains. The
validation split is never altered.

Train independent state and DINO decoders with the same `--steps`, `--batch`,
`--seed`, train/eval caches, and split JSONL files. Each run writes `latest.pt`,
`final.pt`, and the lowest validation correct-CLS flow-loss checkpoint as
`best.pt`; only the decoder parameters are optimized. Validation logs matched
correct, zero, and cross-sample shuffled CLS conditions under the same flow
noise and time.

`evaluate_cfm_decoder_probe.py` infers the decoder layout from both checkpoint
schemas and requires them to agree. For the K65 family it reports the ordinary
reconstruction metrics plus paired correct/zero/shuffled-CLS metrics for every
decoded state source. The image report shows these ablations for observed and
WM-predicted state while holding spatial state and sampling noise fixed. This
measures whether the post-hoc decoder uses CLS; it is not rollout success or
evidence that the world model itself improved.
