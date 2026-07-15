# Frozen-State matched WM-head ablation

## Purpose

Freeze the old SFT2 epoch2 Query representation and the trained best@7500
`8×2048 → 8×1024` encoder. Compare two WM heads over exactly the same cached
State scalars:

- vector: a flatten view `1×8192`, hidden width 896;
- token: the native view `8×1024`, hidden width 1024.

Both use depth 4, 8 heads, the same action-token conditioning, 10,000 optimizer
steps, batch 128, AdamW `lr=1e-4`, weight decay `1e-4`, and
`MSE + 0.1 × (1-cosine)`. The parameter counts are 53,281,664 and 52,503,552
(1.48% difference). No branch-specific tuning is allowed.

## Frozen inputs

Canonical configuration:
`configs/training/reconstruction/frozen_wm_head_shape_ablation.json`.

- Query cache train: 59,389 rows, fingerprint `fe3076b60cc96fe2`.
- Query cache val: 6,054 rows, fingerprint `d06f4adf47846d52`.
- Encoder: query bottleneck probe `best.pt`, step 7500.
- Visual adapter: the adapter portion of the same probe checkpoint.
- Renderer: frozen proven 16×512 Qwen-token CFM.
- Turn set: exactly the six records in
  `configs/eval/reconstruction/rcdm_rollout5_turns_val.json`.

The train cache contains only the training split. Full validation and all six
visual rollouts use the held-out validation split. The six five-action lists
each contain both `4=turn_right` and `5=turn_left`; actions 2/3 are not accepted
as substitutes.

## Trainable and frozen modules

Only the two independent WM heads are trainable. The source Qwen model is not
loaded by cache transformation, head training, or evaluation. The Query
encoder, State-to-vision adapter, and CFM are frozen. The vector branch is only
a view of the same immutable `8×1024` cache used by the token branch.

## Output and recovery

Exclusive server output:

`/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/query_state_ablation/19_frozen_wm_heads_vector8192_vs_token8x1024_s10000_b128`

The cache writes atomic shards and a manifest last. Training saves unique
`step_002000` ... `step_010000` checkpoints plus `best` and `final`.
Optimizer, deterministic sampler position, CPU RNG, and CUDA RNG are included.
A preempted run resumes with:

```bash
STAGE=train RESUME_CHECKPOINT=<output>/train/step_NNNNNN sbatch \
  experiments/training/reconstruction/frozen_wm_head_ablation.slurm
```

Evaluation can be rerun independently with `STAGE=eval`. Existing completed
outputs must not be overwritten; use a fresh directory if invariants differ.

## W&B and resources

- project: `nimloth-wm` (confirmed absent before this experiment);
- run ID: 1;
- comment: `frozen`;
- run name: `1_frozen_vector8192_vs_token8x1024_s10000_b128`;
- params: vector8192 vs token8x1024, 10k steps, batch128.

The single Slurm job requests one GPU, eight CPUs, and at most two hours. This
is within the user-approved limit of at most two GPUs and two GPU-hours.
Training logs per-head loss/timing every 10 steps and full validation every 500
steps; checkpoints are every 2,000 steps.

## Launch and verification

Use the explicit pinned interpreter; never use `torchrun`, venv activation, or
mise:

```bash
sbatch --output=<output>/logs/slurm-%j.out \
  experiments/training/reconstruction/frozen_wm_head_ablation.slurm
```

Before launch, the output README records the exact committed hash and command.
After completion, run:

```bash
bash experiments/validation/verify_wm_head_shape_ablation.sh
```

The verifier requires finite full-cache metrics, best/final reload and five-step
rollout, six contact sheets covering 30 rows, and a completed semantic review.
Pixel L1 is auxiliary and cannot determine the winner.
