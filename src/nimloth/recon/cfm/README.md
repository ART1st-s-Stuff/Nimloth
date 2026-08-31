# CFM reconstruction (`nimloth.recon.cfm`)

This package implements the post-hoc conditional flow-matching (CFM) image
visualizer used by Nimloth reconstruction diagnostics.

- `model.py`: token-conditioned UNet velocity field.
- `flow.py`: straight-path flow loss, shuffled-condition diagnostics, and Euler
  ODE sampling.

CFM is not part of SFT2 or RL optimization. The SFT2 trainer and world model are
frozen before state embeddings are cached; CFM trains only from those cached
states and their observation image paths.

For the direct SFT1 Query-State probe, the same generic model is configured with
`token_count=16` and `token_dim=1024`. Training requires the complete audited
`all_train` cache and evaluation requires only the image-disjoint audited
`external_validation` selection (never raw validation). The cache preserves
`[N,16,1024]`; only
the model-call boundary views it as `[N,16384]`, after which
`TokenConditionedFlowUNet.encode_condition` restores the 16 tokens for token
projection and cross-attention. This flattening is not pooling and does not
create another deployment state transform.

The Query-State CFM predicts three channels in `[-1,1]`; approved sampling
converts them explicitly to sRGB PNG only after a preregistered full-split
multi-noise sensitivity gate. Checkpoint comparisons reuse fixed validation
seeds; any final robustness seeds are preregistered in checkpoint invariants.
Publication loads the checkpoint, checks its schema and exact CFM config, matches
validation cache/split/row-set/item count, and recomputes every per-seed mapping,
metric, aggregate and evidence identity. Each artifact then records the exact
decoder checkpoint/hash, validation provenance, verified source-image/output-PNG hashes,
image preprocessing, ODE steps, noise seed, row identity, and color conversion.
These color reconstructions are secondary post-hoc readability aids and cannot
replace direct DINO feature metrics.

Building the real cache, training the decoder, or generating formal color images
is an experiment requiring its own reviewed contract and explicit launch
approval.
