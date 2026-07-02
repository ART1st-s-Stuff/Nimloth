# RCDM adapter

Nimloth RCDM adapter for post-hoc SFT2 latent-state reconstruction.

Purpose:

- Reuse `external/RCDM` without editing the submodule.
- Train an RCDM UNet conditioned on Nimloth SFT2 WM state embeddings (`StateProjector(Qwen <|latent_state|>)`).
- Sample images from true or WM-predicted SFT2 latent states for visualization.

Key files:

- `external.py`: locates `external/RCDM` and imports its `guided_diffusion_rcdm` package.
- `config.py`: RCDM model/diffusion configuration and factory.
- `image_utils.py`: image normalization helpers for guided-diffusion tensors.
- `checkpoint.py`: checkpoint save/load helpers for Nimloth-trained RCDM models.
- `state_cache.py`: compressed shard cache for `StateProjector(Qwen <|latent_state|>)` embeddings.

Training CLI:

```bash
python -m nimloth.training.reconstruction.rcdm_sft2 \
  --model /path/to/sft2/export_best_hf \
  --state-proj-checkpoint /path/to/sft2/best/state_proj.pt \
  --wm-checkpoint /path/to/sft2/best/wm_predictor \
  --train-jsonl /path/to/train_all.jsonl \
  --val-jsonl /path/to/val_all.jsonl \
  --output-dir outputs/experiments/training/reconstruction/<date>/<run_name> \
  --wandb-run-name <run_name>
```

Resume with `--resume` from the latest `training_state_*.pt` in the output
directory, or with `--resume-checkpoint /path/to/training_state_*.pt` for a
specific checkpoint. The W&B run id is stored in `wandb_run_id.txt`.

For full runs, prefer the compressed state cache so RCDM training does not rerun
frozen Qwen every step:

```bash
python -m nimloth.training.reconstruction.rcdm_sft2 \
  ...same model/checkpoint/jsonl flags... \
  --state-cache-dir outputs/experiments/training/reconstruction/cache/<name> \
  --build-state-cache \
  --state-cache-compression gzip \
  --state-cache-dtype float16
```

The cache stores only state embeddings and image paths, not image tensors.  A
1024-dim `float16` state costs roughly 2 KB per transition before gzip, much
smaller than the old SFT2 preprocess cache that stored Qwen image tensors.

Sampling CLI:

```bash
python -m nimloth.eval.rcdm_reconstruction \
  --model /path/to/sft2/export_best_hf \
  --state-proj-checkpoint /path/to/sft2/best/state_proj.pt \
  --wm-checkpoint /path/to/sft2/best/wm_predictor \
  --rcdm-checkpoint outputs/.../ema_0.9999_000100000.pt \
  --val-jsonl /path/to/val_all.jsonl \
  --output-dir outputs/.../samples \
  --timestep-respacing 100
```
