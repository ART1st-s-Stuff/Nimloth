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

## Result

Experiment commit `1eee7c5373234069724a808b452bddc783ea3f88` ran as Slurm
job `476723` on one H800 and completed exit0 in `00:06:24`. W&B project/run is
`nimloth-wm/ned9k9vf`. Cache transformation took 13.96s; training took 308.78s
and 0.0858 GPU-hour. Output size is 11 GiB.

Both branches completed 10,000 matched steps and strict finite reload/rollout.
Best vector is step3500, val MSE `0.16083155`; best token is step2000,
`0.16159483`. Vector/token throughput is 134.57/70.26 branch steps/s. Full-val
MSE at horizons1..5 is vector
`0.160832,0.208456,0.240766,0.266622,0.287718` versus token
`0.161595,0.218074,0.253824,0.280754,0.302603`. Vector also has a larger
one-step shuffled-action penalty: 19.9% versus 9.1%.

All 30 fixed visual rows were reviewed. Vector often becomes a smooth same-color
wall; token often has sharper but wrong doors/corridors/fixtures. Neither branch
reliably follows the actual right/left viewpoint sequence, so no overall visual
winner or new default is declared. Vector is the better fast latent-dynamics
head under this matched budget, subject to that visual limitation.

Post-processing/verifier commit `7bd6939` produced per-horizon auxiliary visual
metrics and passed the deterministic artifact verifier. Full evidence is in the
exclusive output README, `eval/dynamics_metrics.json`,
`eval/turns/semantic_review.json`, and `eval/turns/visual_horizon_metrics.json`.
The direct release command passed on cleanup commit `b51e4d6`: 13 server tests,
all artifact gates, and `release_suite=PASS` without mise.
