# ID189 source20 Base+Common120 full-browser evaluation contract

Status: normal 4x2 retry2 cancelled for proven archive bottleneck; retry3 implementation ready

## Attempt 0 — Job 525905

- Allocated `preempt/dgx-42` with 8×H800 and ended `FAILED 1:0` after `00:01:15`.
- Allocation, dynamic FloorPlan1 render, Ray `[8]`, runtime environment/import probe, and cleanup passed.
- The runner clean-tree gate rejected `external/le-wm/__pycache__/module.cpython-312.pyc`. It was created by server preflight pytest in the production worktree without disabling Python bytecode.
- Failure occurred before formal `RUN_OUT`, W&B/model load/rollout/validation/browser. No optimizer update or checkpoint exists.
- Attempt0 is not resumable and neither its output nor W&B identity may be reused. See `E0136`.

## Retry 1 — Job 527282

- Fresh production worktree: `/project/peilab/atst/nimloth/.worktrees/id189-bc120-r1-37489767` at Parent `37489767cb3393092867123a37ee7cf91ca9a04c`.
- Fresh output suffix: `_preempt_retry1`; fresh W&B suffix: `-preempt-r1`.
- No tests or imports were run in this production worktree. Parent, VAGEN, VERL, le-wm, and RCDM were all clean immediately before submission.
- Submitted to preempt 1×8 as Job `527282`; it allocated `dgx-47` and ended `FAILED 1:0` after `00:01:12`.
- Worktree/allocation/render/Ray/source checkpoint/assets/generated 120-row validation config/cleanup passed. A stale copied assertion required 300 unique validation IDs although this filtered evaluation correctly has 120. See `E0137`.
- No W&B/model load/rollout/validation/browser/update/checkpoint exists. Partial output contains preflight diagnostics and `progress.md`; retry1 cannot resume or reuse identities.

## Retry 2 — normal 4x2

- Human directed immediate retry on available normal 4×2 resources.
- Fresh output: date `2026-08-22`, run suffix `_normal_4x2_retry2`.
- Fresh W&B identity suffix: `-normal-4x2-r2`.
- Allocation: normal, 4 nodes × 2 H800, 16 CPU and 64 GiB per node, total 8 GPU/64 CPU/256 GiB, 5 hours.
- VAGEN ID189 topology is 4×2 with no one-node `joint_process_on_nodes`; rollout remains TP8/DP1 and actor restore remains DP8 across Ray's four 2-GPU nodes.
- Dynamic navigation head authentication and 10.23 fabric/Ray address validation remain mandatory.
- Job `527287` started at `2026-08-22T15:14:41+08:00` on `dgx-[26,28-29,31]`. Dynamic render, four-node Ray `[2,2,2,2]`, source20 checkpoint, 120-row manifest, Base/Common environment prewarm, DP8 actor loading, distributed TP8 vLLM startup, and all 40 first-batch rollouts passed.
- It was cancelled at `02:58:05` after the TaskRunner spent over `02:21` CPU at about 111 GB RSS constructing serial deflated archives while atomic output remained at zero rows/files. Three such batches could not complete before the 5-hour limit. No update/checkpoint/browser batch exists; W&B was finalized failed with zero rows.
- VAGEN now uses ZIP-stored `np.savez` rather than `np.savez_compressed`; all float32 key/shape/finite/SHA256 contracts remain unchanged and a ZIP compression-type regression test passes. See `E0138`.
- TERM/INT now preserve exit143 so cancellation cannot produce a false passed phase marker; retry2 metadata was corrected. See `E0139`.
- Retry3 uses fresh `_normal_4x2_retry3` output and `-normal-4x2-r3` W&B identities.

- Nimloth implementation commit: `860062e4a37e6e847828e089f69b4905eeaccc78`.
- VAGEN implementation commit: `14d862e816f6f598c0f2eeb3383ac2df6b894e84`.
- VERL commit: `494f264494b2525f2c13595f63ac4912963e6d2f`.

## Purpose

Evaluate the frozen ID184 step20/source796 K4 Scheme-B policy on all held-out Navigation Base 60 and Common Sense 60 tasks. Persist the complete Rollout Browser evidence for every rollout and every real turn.

## Identity

- W&B project: `vagen`.
- Run name: `189_eval_rollout_browser_k4_dp8_tp8_source20_base_common120_t20_s100_preempt`.
- W&B run ID: `nimloth-id189-eval-rollout-browser-k4-source20-base-common120-preempt`.
- Output: `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-21/189_eval_rollout_browser_k4_dp8_tp8_source20_base_common120_t20_s100_preempt`.
- Fresh output and W&B identity only; failed attempts are never overwritten or resumed.

## Data and split

- Source asset config: VAGEN `val_navigation_joint_id185.yaml`, filtered by parsed `config.eval_set` to exactly `base` and `common_sense`.
- Each source has `n_envs=60`; effective explicit seeds are `1..60`.
- Expected rows: exactly 120, with 60 unique rows per source and 120 unique rollout identities.
- Both assets are held-out evaluation scenes and remain disjoint from `*_train` scenes; source asset hashes and scene disjointness are revalidated before model startup.

## Model and execution

- Actor/source checkpoint: immutable ID184 `global_step_20`, source796.
- Planning checkpoint: corrected ID74 epoch1; active snapshot `sha256:6648780b3791cb4b937974b151b9e119ed9bf74602d1bc21dabfc30a3914d969`.
- All modules are frozen. `val_only=true`; no backward, optimizer update, snapshot publication, or checkpoint write.
- Scheme-B alpha1/beta85.78297006578457, K4 horizon4, 100 UCT simulations, exploration1.0.
- Actor restore is DP8; rollout is one vLLM TP8/DP1 replica with `mm_encoder_tp_mode=data`.
- Each real turn stores image, raw response/CoT, action distribution/action/Q/value, latent `[16,2048]`, true current state `[16,1024]`, all unique predicted states `[N,16,1024]`, and all 100 chronological MCTS simulations.

## Runtime and recovery

- Partition `preempt`; one node × 8 H800, 64 CPU, 256 GiB, 5-hour limit.
- Expected runtime: approximately 1–3 hours. Estimated output: approximately 40–60 GiB.
- Validation batch size 40 produces three append-only journal batches and three atomic browser batches.
- The current runner does not resume a failed formal identity. A failure preserves completed diagnostics but requires a fresh output/W&B identity and complete 120-row rerun.

## Entrypoints

- Slurm: `experiments/training/rl/id189_source20_base_common120.slurm`.
- Launcher: `experiments/training/rl/launch_vagen_k4_1x8_browser_on_hold.sh`.
- Runner: `experiments/training/rl/run_vagen_k4_id189_source20_base_common120.sh`.
- VAGEN config: `vagen/configs/joint_id189_source20_base_common120.yaml`.

## Completion gates and monitoring

- Slurm `COMPLETED 0:0`; phase/final validator `passed`.
- Restore marker: `ID189_K4_SOURCE20_BASE_COMMON120_RESTORE_OK global_step=20`.
- Journal and browser each contain exactly 120 rows in 3 batches.
- Every archive passes key/dtype/shape/finite/SHA256 checks; every turn has exactly 100 simulation indices `0..99` and horizon4.
- Report per-source success, reward sum/mean, and turn count.
- W&B finishes with the single validation history step; checkpoint list remains empty.
- Monitor Slurm state, Ray/vLLM startup, DP8 restore, environment health, journal batch progress, filesystem capacity, browser finalization, W&B, and cleanup.
