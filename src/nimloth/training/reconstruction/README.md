# Reconstruction decoder training

This package trains a **post-hoc** image decoder for WM diagnostics.  It freezes
Qwen, `StateProjector`, and the WM predictor checkpoint, then trains only
`WMImageDecoder` to reconstruct observations from projected WM states.

Reusable CFM and RCDM model code lives under `nimloth.recon`; this package owns
their training orchestration and command-line entrypoints.

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

For a DINO-grid checkpoint whose projector/backbone changed after the original
CFM was trained, first build a checkpoint-aligned cache from the immutable SFT2
compact preprocessing cache:

```bash
python -m torch.distributed.run ... \
  -m nimloth.training.reconstruction.cache_dino_grid_states \
  --sft2-checkpoint /path/to/sft2/epoch_002 \
  --train-jsonl /path/to/train.jsonl \
  --val-jsonl /path/to/val.jsonl \
  --preprocess-cache-dir /path/to/compact/preprocess \
  --preprocess-cache-processor-source /path/to/sft1/init \
  --output-dir /path/to/aligned_state_cache \
  --git-commit <exact_commit>
```

The distributed cache writer uses the checkpoint's online backbone weights and
creates two equally weighted pairs per transition: actual current state to
current image, and frozen-WM predicted next state to the actual next image. The
second pair intentionally lets CFM learn systematic decoder-side compensation
for WM prediction error, matching the predicted-state input used by ValueHead.
The backbone, projector, and WM predictor remain frozen. Atomic rank shards can
resume only under an identical world-size and artifact contract. `cfm_sft2`
detects the `[K,D]` cache shape. Use `--condition-dropout 0.15` when the
downstream probe uses CFG and `--init-cfm-checkpoint` only for a strict
identical-architecture warm start. `--skip-samples` omits the legacy
single-vector WM sample hook when a separate grid-WM evaluator owns the final
reconstruction.

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
