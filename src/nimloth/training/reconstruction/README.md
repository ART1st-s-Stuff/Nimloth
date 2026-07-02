# Reconstruction decoder training

This package trains a **post-hoc** image decoder for WM diagnostics.  It freezes
Qwen, `StateProjector`, and the WM predictor checkpoint, then trains only
`WMImageDecoder` to reconstruct observations from projected WM states.

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
  --wandb-run-name <rcdm_name>
```

Resume the latest RCDM checkpoint in the same output directory:

```bash
python -m nimloth.training.reconstruction.rcdm_sft2 \
  ...same flags as above... \
  --resume
```

Use `--resume-checkpoint outputs/.../training_state_000001000.pt` to resume a
specific checkpoint. W&B uses `wandb_run_id.txt` in the output directory when
`--resume` is set.

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
