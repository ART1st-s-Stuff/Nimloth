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
