# ID189 source20 Base+Common120 full-browser evaluation contract

Status: implementation ready; not submitted

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
