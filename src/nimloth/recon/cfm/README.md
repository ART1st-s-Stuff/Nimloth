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

Formal38's actor failure remains above all reconstruction evidence. Direct DINO
metrics are primary, condition sensitivity is secondary, and three-channel sRGB
strips/contact sheets are visual inspection aids that can also fail because of
the decoder/domain. They cannot promote update1605, select or resume SFT1,
establish deployability, or authorize SFT2.

Building either real cache, training either decoder, or generating formal or
forensic color images is an experiment requiring its own reviewed contract and
explicit launch approval.
