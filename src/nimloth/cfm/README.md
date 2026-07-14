# Conditional flow matching

This package implements the post-hoc conditional flow-matching (CFM) image
visualizer used by Nimloth reconstruction diagnostics.

- `model.py`: token-conditioned UNet velocity field.
- `flow.py`: straight-path flow loss, shuffled-condition diagnostics, and Euler
  ODE sampling.

CFM is not part of SFT2 or RL optimization. The SFT2 trainer and world model are
frozen before state embeddings are cached; CFM trains only from those cached
states and their observation image paths.
