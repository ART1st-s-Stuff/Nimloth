# CFM reconstruction (`nimloth.recon.cfm`)

This package implements the post-hoc conditional flow-matching (CFM) image
visualizer used by Nimloth reconstruction diagnostics.

- `model.py`: schema-distinct token-set and spatial-grid conditioned UNet velocity fields.
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

The Formal38 forensic CFM adapter is deliberately separate from this deployable
Query-State owner. It accepts only
`nimloth_query_state_forensic_reconstruction_cache_v1` from Job540589's
actor-failed, non-resumable, non-authoritative, nondeployable update1605 owner;
there is no unsafe compatibility flag on the deployable CLI. Its state was
extracted read-only from original observations plus matching real archived
responses/CoT through the frozen final-current K16/direct-head path. The decoder
still sees 16×1024 tokens and trains only `TokenConditionedFlowUNet`; no Qwen,
direct head, optimizer/RNG from Formal38, `StateProjector`, WM, Value, or SFT2
owner enters the decoder checkpoint.

Forensic Stage A is a mechanics/overfit probe on 48 `mechanics_train` and 16
exact-image-disjoint `mechanics_validation` train-derived rows. Correct and
global shuffled conditions share noise/time for every preregistered seed. Only
the final decoder checkpoint's mechanics-train gate controls pass;
mechanics-validation is report-only and not held out. A pass does not authorize
Stage B. Full `all_train`/`external_validation`, 128px training, thresholds,
resources, and launch identity must be replanned and separately approved, and a
fresh decoder is the default.

The separate `stage_b_diagnostic` forensic owner requires a fresh 128px decoder
and the complete image-disjoint 12,836/1,413 cache. Its only publication
checkpoint is step4000; seeds 20260931/32/33 must each reach external normalized
velocity-MSE delta 0.01 and their aggregate shuffled/correct ratio must reach
1.05 before 16 deterministic external Euler50 RGB examples are emitted. Stage A
cache/checkpoints cannot initialize or resume this owner.

A failed Stage B publication gate remains final scientific evidence. The
separate forensic post-hoc inspection owner can, only after a new approval,
replay the original deterministic 16-row correct-condition seed20260921,
128px/Euler50/chunk8 sample for human viewing. It loads no live optimizer,
performs no update/resume or shuffled publication, writes a fresh manifest-last
non-publication schema, and binds the exact failed-gate/checkpoint/cache/summary
identities without modifying the original run.

The forensic oracle ladder keeps the legacy `TokenConditionedFlowUNet`
(`decoder_family=token_set_v1`) byte-compatible and adds a separate
`SpatialConditionedFlowUNet` (`decoder_family=spatial_grid_v1`). The legacy
family normalizes and cross-attends an unordered token set. The spatial family
requires exactly 16 row-major slots, reshapes them to 4×4, adds fixed normalized
coordinates, and injects resized spatial condition maps at every UNet resolution.
Token and spatial checkpoints are schema-bound and cannot initialize or resume
each other.

The preregistered matrix is immutable `token_state` plus fresh `token_oracle`,
`spatial_state`, and `spatial_oracle`. `state` always means the matching
Formal38 K16 canonical state; `oracle` means the frozen DINOv2-large 4×4 target
computed from the original archived observation through exact SFT1 teacher
preprocessing. It never means DINO computed from an already resized decoder
image. All fresh cells train decoder-only with the same Stage B rows, optimizer,
step budget, seed and evaluation schedule. Fixed-time flow diagnostics report how
much target RGB is already present in the interpolated model input, while
pure-noise Euler50 samples are the generation evidence.

Formal38's actor failure remains above all reconstruction evidence. Direct DINO
metrics are primary, condition sensitivity is secondary, and three-channel sRGB
strips/contact sheets are visual inspection aids that can also fail because of
the decoder/domain. Oracle-ladder results are forensic representation-
decodability evidence only: they cannot promote update1605, select or resume
SFT1, establish deployability, authorize SFT2, or reconstruct information absent
from the 4×4 teacher.

Building either real cache, training either decoder, or generating formal or
forensic color images is an experiment requiring its own reviewed contract and
explicit launch approval.
