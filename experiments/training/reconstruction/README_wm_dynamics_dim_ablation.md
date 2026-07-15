# Frozen-State SFT2 dynamics-dimension ablation

## Question

Before starting complete SFT2, test whether its `wm_dynamics_dim=2048`
factorization damages dynamics relative to `wm_dynamics_dim=8192` when both
predict the same external `1×8192` State.

This is a WM-only cache experiment. Qwen, the 8×1024 State encoder, visual
adapter, and CFM are frozen and Qwen is not loaded.

## Compared predictors

Both use existing `LatentWMPredictor` semantics: LeWM AR predictor hidden1024,
depth6, heads16, MLP4096, history4, and eight actions.

| Branch | External State | dynamics_dim | Parameters |
|---|---:|---:|---:|
| full | 8192 | 8192 | 408,345,672 |
| factorized | 8192 | 2048 | 160,648,264 |

The parameter budgets are intentionally unequal. This experiment isolates the
practical effect of the 2048 dynamics bottleneck used by the planned SFT2; it
must not be reported as a matched-parameter architecture comparison.

## Data and budget

Canonical config:
`configs/training/reconstruction/wm_dynamics_dim_ablation.json`.

Shared immutable ID19 State cache:

- train: 59,389 rows / 56,172 adjacent dynamics transitions, fingerprint
  `b0802d7c6dae1639`;
- held-out val: 6,054 rows / 5,699 transitions, fingerprint
  `520b27798fb28c1c`;
- each State is FP16 `8×1024`, exposed to both predictors as the same flatten
  view `1×8192`.

Training is exactly five independently shuffled epochs. Every train transition
is visited once per epoch. Batch128 gives 439 steps/epoch and 2,195 total.
Both use the same seed20260716, batch indices, AdamW lr3e-4/wd1e-4,
`MSE + 0.1×(1-cosine)`, BF16 autocast, and grad clip1.

Each epoch saves model, optimizer, sampler position, CPU RNG, and CUDA RNG.
Best branch weights are selected independently by held-out one-step MSE and
combined into `train/best`; `train/final` preserves the epoch5 pair.

## Evaluation

Use the same full-val definition and six fixed turn-both rollouts as ID19:

- one-step MSE/cosine and shuffled-action controls;
- every valid full-val horizon1–5 window;
- exactly 30 visual rows with the frozen adapter/CFM and matched noise;
- columns: GT, Qwen positive, Frozen State GT, full8192 WM, factorized2048 WM;
- semantic review takes priority over auxiliary pixel L1.

## Production-shape smoke

Job `476783` ran commit `b90d53668450b17ace193caa44156cd0d9dfde97`
on one H800 and completed exit0 in `00:01:51`.

- W&B: `nimloth-wm/65w2wpv8`, ID2 run
  `2_smoke_dynamics8192_vs2048_state8192_ep1_b128`;
- batch128, two optimizer steps over an explicit real-State subset;
- gpumem 11,214 MiB; MaxRSS about8.68GiB;
- full/factorized throughput 0.922/33.4 step/s;
- epoch checkpoint, best/final strict reload, and five-step rollout passed;
- no OOM, NaN, traceback, or Qwen load.

The smoke metrics are mechanics-only and have no quality interpretation.

## Formal run

- W&B project: `nimloth-wm`;
- ID: 3;
- run: `3_dynamics8192_vs2048_state8192_ep5_b128`;
- resource: one normal H800, eight CPUs, two-hour hard limit;
- estimated runtime: 45–60 minutes, below one GPU-hour;
- expected output: about40–50GiB including five resume checkpoints.

Exclusive output:

`/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/query_state_ablation/20_state8192_dynamics8192_vs2048_ep5_b128`

Launch:

```bash
sbatch --output=<output>/logs/slurm-%j.out \
  experiments/training/reconstruction/frozen_wm_dynamics_dim_ablation.slurm
```

Resume a preempted run with a completed epoch checkpoint:

```bash
STAGE=train RESUME_CHECKPOINT=<output>/train/epoch_NNN sbatch ...
```

Do not start complete SFT2 until this experiment and its visual review finish.
