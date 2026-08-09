# 2026-08-09: ID147 PlannerPolicyHead PPO node-switch retry

## Purpose and retry boundary

- Retry the formal PlannerPolicyHead experiment from corrected SFT2 epoch 1 at
  RL global step 0, train through committed global step 20, and then run the
  standard held-out 120-episode evaluation.
- ID146 Job `510253` ended before the outer controller because the exact
  single-visible renderer probe on `dgx-17` exceeded its 150-second cold-home
  wall-clock gate. ID146 produced no rollout, W&B run, optimizer step,
  consumption state or RL checkpoint and is not resumed.
- The human explicitly chose “switch node”. ID147 therefore keeps the existing
  150-second renderer gate and all model/data/training semantics unchanged, but
  adds `dgx-17` to the node exclusions. Any allocated node must still pass the
  exact renderer probe and the formal navigation prewarm; no old renderer result
  is reused.

## Identity and source

- ID/comment: `147`, `retry_node`.
- W&B project/run:
  `nimloth-rl/147_retry_node_plannerpolicy_sft2ep1_rl16_eval20x120_policyh1_k16_dino05_ppo4_iter20_2n4r2g_2xtp4_preempt44`.
  Evaluation uses the same name with suffix `-eval`.
- W&B API reported current numeric max 143 because IDs 144--146 stopped before
  creating W&B runs; ID147 is used to preserve the already assigned experiment
  sequence. Exact training and evaluation names each had zero matches.
- Formal output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-09/147_retry_node_plannerpolicy_sft2ep1_rl16_eval20x120_policyh1_k16_dino05_ppo4_iter20_2n4r2g_2xtp4_preempt44`.
  The output, adjacent iteration-progress log and adjacent renderer-preflight
  root were absent at login preflight.
- Runtime worktree:
  `/project/peilab/atst/nimloth/.worktree/dev-7f12c79f`, tracked-clean at
  `082e24f1a966781c2f57ff30a415e143a4c76c59`.
- Code/config commit:
  `42e29ee25efae03ce89909f76fc2a954b7231782`.
- VAGEN and LeWM pins:
  `192c35a91f3941b72d5e1272af6603ef7a7d93e0` and
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- Config:
  `configs/training/rl/planner_policy_h1_full_16rollout_8gpu_44_step20.yaml`.
- Entrypoints: `experiments/training/rl/train_8gpu_44.slurm`,
  `run_vllm_online_ppo_full.sh`, and
  `run_vllm_online_ppo_parallel_slurm.sh`.
- Python:
  `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`.

## Initialization, data and objective

- Corrected SFT2 epoch-1 initialization:
  `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-08-02/sft2/74_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16n3g844lw844_px100352/train_ws16/epoch_001`.
  Login preflight rechecked non-empty Qwen config/shards, StateProjector, WM
  predictor, ValueHead and training state; the root has no `rl_state.pt`.
- Revalidated immutable seed-42 PlannerPolicyHead artifact:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-07/146_plannerpolicy_head_seed42_from_sft2ep1`.
  Weight SHA256 is
  `f8577c9a25d8587208b08024a1b0fd7c13e60f012a4a545335dbd9aefc9e9a42`;
  metadata SHA256 is
  `ca919e00eafc330327a7a9fc207a157f634ee42aa3a1d0134c16807108a1f60c`.
  It is an untrained initialization artifact, not an ID146 RL checkpoint.
- Training uses `base_train` and `common_sense_train`, eight fresh trajectories
  per split per iteration, bounded retries, max 20 steps; per-split seeds 1..8
  at iteration 1 through 153..160 at iteration 20.
- Held-out evaluation runs only after committed step 20 on `base` and
  `common_sense`, seeds 1..60 each, exactly 120 episodes. Held-out assets never
  enter optimization.
- Trainable: Qwen language body, WM predictor, ValueHead and PlannerPolicyHead.
  Frozen: Qwen vision, `lm_head`, StateProjector, DINO teacher and direct-Qwen
  token actor/reference-KL path.
- PlannerPolicyHead PPO uses behavior action log-probabilities, clip 0.2,
  entropy coefficient 0.01, temperature 1.0 and four epochs. ValueHead learns
  MC returns; DINO-grid weight is 0.5; gamma is 1 with zero truncated bootstrap.
  No fixed or invented CoT is used.

## Resource and recovery contract

- Request `preempt/preempt_qos`, two flexible physical nodes x four H800 GPUs,
  64 CPUs and 48 GiB per node, eight hours, `--requeue`; four synchronized
  two-GPU training ranks and two node-local TP4 rollout workers.
- Exclude `dgx-17,dgx-32,dgx-37,dgx-51`. The retry does not pin a replacement
  node. At the 2026-08-09 01:36+08 snapshot, preempt had 26 free GPUs;
  `dgx-38` exposed four and `dgx-44` eight, but the scheduler chooses the actual
  allocation and all nodes are revalidated.
- Expected runtime remains about 5--7 hours, or 40--56 GPU-hours. The human was
  informed before submission and selected the node-switch retry.
- Each successful iteration writes a complete checkpoint before committing its
  matching fresh-rollout consumption. A preempted job resumes only from the
  newest complete committed checkpoint. Failed/uncommitted rollout is never
  consumed.

## Pre-launch status

- Login preflight passed runtime cleanliness and commit/submodule pins, Python,
  config parsing, checkpoint file gates, PlannerPolicyHead hashes, empty output
  paths and exact W&B uniqueness.
- Config parsed as 20 iterations, 16 episodes/iteration, max 20 steps, two
  nodes, world size four, two GPUs/rank, eight total GPUs, TP4, external interval
  20 evaluation with 120 held-out episodes, PlannerPolicyHead enabled, PPO four
  epochs and clip 0.2.
- No user Slurm job was active at the login preflight snapshot.
- Exact `sbatch --test-only` accepted the full request and estimated
  `dgx-38,dgx-44`. The immediate pre-submit resource refresh still showed 26
  free preempt GPUs, including four on `dgx-38` and eight on `dgx-44`.

## Submission status

- Formal Job `511059` was submitted at `2026-08-09T01:40:02+08:00` with job name
  `id147-pp-sft2-node`. Its Slurm contract is two nodes x four GPUs, 64 CPUs and
  48 GiB per node, eight hours, `Requeue=1`, and exclusions
  `dgx-17,dgx-32,dgx-37,dgx-51`.
- The first observation was `PENDING(Priority)` with empty `NodeList` and
  `AllocTRES`. Ten seconds later Slurm allocated `dgx-38,dgx-44`; the job began
  at `01:40:12+08:00` with the full eight GPUs/128 CPUs. The absolute batch
  command and exported `REPO` contract are used even though Slurm recorded the
  submission working directory as `/home/csejzhang`.
- Runtime/dependency and W&B uniqueness gates passed. Both exact cold-home
  renderer probes then passed within the unchanged 150-second gate: `dgx-38`
  emitted `AI2THOR_RENDER_OK` in 74.200 seconds and `dgx-44` in 76.360 seconds;
  both frames were 255x255 with dynamic range 246. Slurm recorded both renderer
  steps `COMPLETED 0:0`.
- The outer controller started iteration 1 at `01:42:24+08:00` from the corrected
  SFT2 checkpoint with seed offset 1. It created the formal README/train/rollout
  skeleton and launched one TP4 rollout step on each node. This proves allocation
  and renderer health only; no claim of healthy rollout or training is made
  until both TP4 rollouts, strict merge, synchronized PPO update, consumption
  commit and complete checkpoint are verified.

## Terminal result

- Job `511059` finished `COMPLETED 0:0` at `2026-08-09T06:53:30+08:00`, elapsed
  `05:13:18`. Every formal iteration completed with 16 fresh trajectories; all
  20 consumption sidecars are committed and form exact global-step lineage
  `0 -> 20`. Per-dataset training seeds are exactly `1..160`.
- Step 20 used 259 transitions and produced finite four-epoch metrics:
  `wm_mse=0.2463575894`, `dino_grid_mse=0.8953118226`,
  `value_loss=26.5014698212`, `planner_policy_loss=-1.6011937615`, entropy
  `1.4336167960`, clip fraction `0.0289575284`, mean ratio `1.0033764586`, and
  total loss `25.5799533766`. Its train-rollout success `7/16=0.4375` is only a
  batch diagnostic.
- `train/latest`, `train/final`, and `train/iter_0020` expose complete
  `13,098,478,473`-byte `rl_state.pt` checkpoints and required components.
- Standard held-out evaluation is `ALL_OK`: exactly 120 trajectories / 2,157
  transitions; `base/common_sense` seeds `1..60` each. Overall success is
  `21/120=0.175`, average reward `-0.6419166667`, average steps `17.975`.
  `base` is `11/60=0.1833333333`; `common_sense` is
  `10/60=0.1666666667`.
- Training W&B `i1g3w8b7` and eval W&B `n929fhah` both printed successful sync
  completion. Login-node API credentials resolve to another default entity, so
  independent API state lookup was not claimed.
- The requested corrected-SFT2-epoch1 -> RL-step20 -> held-out120 mechanics goal
  is complete. There is no matched step-0 held-out evaluation, so the final
  success rate does not prove PPO quality improvement. ID143 also measured
  `21/120`, but it is not a matched initialization/objective control.
- No resume is needed. A future continuation must use a new identity, start from
  committed `train/final`, and use fresh non-overlapping training seeds.
