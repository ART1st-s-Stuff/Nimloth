# CFM reconstruction (`nimloth.recon.cfm`)

This package implements the post-hoc conditional flow-matching (CFM) image
visualizer used by Nimloth reconstruction diagnostics.

- `model.py`: token-conditioned UNet velocity fields. `spatial_grid_v1`
  consumes a square spatial grid. `spatial_cls_grid_v1` consumes an explicit
  row-major `8x8` spatial grid followed by one non-spatial DINO CLS token. The
  CLS token has a separate normalization/MLP path and only contributes to the
  global residual/time condition; it is never reshaped into the grid.
- `flow.py`: straight-path flow loss, shuffled-condition diagnostics, paired
  correct/zero/shuffled-CLS diagnostics, and Euler ODE sampling.

CFM is not part of SFT2 or RL optimization. The SFT2 trainer and world model are
frozen before state embeddings are cached; CFM trains only from those cached
states and their observation image paths.

The two decoder families have separate checkpoint schemas and fail closed when
the cache layout, checkpoint identity, or model family disagrees. A K65 cache
cannot be passed to the spatial-only decoder, and a K64 cache cannot be padded
to train the spatial+CLS decoder. CLS ablations keep the spatial state, target
image, flow time/noise, and reconstruction sampling noise fixed.
