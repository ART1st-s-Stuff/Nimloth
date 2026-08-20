# ID187 rollout browser production canary contract

Date: 2026-08-21
Status: switched to available preempt 1x8; retry17 prepared

The original normal Job `525468` was cancelled pending. Preempt retries `525568`, `525570`, `525572`, and `525575` exposed and fixed the allocation-state/Slurm-client bootstrap. Retry5 Job `525581` passed exact4x2, dynamic head20, Ray, environment, all prewarms, checkpoint hashes, and model startup, then correctly failed before rollout because it reused the experiment-bound ID185 config with an ID187 W&B name. It created a partial diagnostic output but no browser, validation row, checkpoint, or W&B run. Retry6 added the dedicated `id187_k4_source20_browser_v1` gate, but Job `525594` was cancelled with zero allocation/output/W&B after the human required full current/predicted states and the chronological 100-simulation process. Retry7 Jobs `525617/525618` were cancelled unallocated when the human directed use of one currently free 6-GPU node plus one 2-GPU node. Retry8 Job `525644` successfully allocated `dgx-52=6` plus `dgx-20=2`, then failed in one second before output/W&B/Ray/environment because nested `srun --jobid=<base> --het-group=0,1` scoped itself to only component0. Retry9 Job `525646` showed that comma-selected `--het-group=0,1` still does not flatten two one-node components for a nested two-node step. Retry10 Job `525648` proved per-component node steps work on group0, then exposed inherited component0 `cpus-per-task=48` on the 2-GPU/16-CPU group1 step. Retry11 Job `525651` showed that `--het-group=1` from the component0 batch still cannot schedule the second component on this cluster, even with the correct 16-CPU request. Retry12 Job `525654` confirmed real component JobIds but group1 still inherited component0's 192-GiB step memory request. Retry13 Job `525658` confirmed component sibling JobIds cannot be addressed as independent allocation leaders. Dedicated hold Job `525661` then proved the exact command on both nodes: `srun --jobid=<heterogeneous leader> --het-group=<offset>` plus explicit per-node CPU/memory/GRES. Retry14 Job `525666` showed the externally proven command is still overridden by inherited component0 `SLURM_*` batch variables. Retry15 Job `525670` showed additional inherited `SLURM_*` fields still affected nested scheduling. Retry16 Job `525672` failed before output/W&B because nested group1 steps remained unschedulable. The human then directed use of the available single 8-GPU preempt node. Retry17 uses one Ray node with 8 H800 GPUs; actor world size remains DP8 and rollout remains TP8/DP1.

## Purpose

Validate that one real frozen K4 Scheme-B Navigation evaluation rollout produces a complete unified Evaluation Rollout Browser on the production Ray/vLLM path. The canary validates behavior-time images, raw response/CoT, all action scores, direct Q/Frozen-V, all MCTS candidates, same-generation latent/current state, every unique predicted node state, all 100 chronological UCT simulations and backups, identity joins, binary/manifest hashes, and the global complete marker without any second model/WM/critic forward.

## Immutable code and entrypoints

- Parent state-process implementation commit: `4f76c926d1f41c62bb309d59f8d2a561a8cb5053` (runtime pin supplied through `EXPECTED_PARENT_COMMIT`).
- VAGEN: `af06b9288a504ccc739749a0e643dd01a28effcc`.
- VERL: `494f264494b2525f2c13595f63ac4912963e6d2f`.
- Slurm entry: `experiments/training/rl/id187_rollout_browser_canary.slurm`.
- Allocation launcher: `experiments/training/rl/launch_vagen_k4_id185_visualize_base_failure_on_hold.sh`.
- Phase runner: `experiments/training/rl/run_vagen_k4_id185_visualize_base_failure.sh`.
- VAGEN config/gate: `external/VAGEN/vagen/configs/joint_id187_source20_visualize_one.yaml`, `id187_k4_source20_browser_v1`.
- Python: `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`; activation scripts and console-script shebangs are not used.

## Data and selection

- Environment: VAGEN Navigation.
- Split: heldout `base` test tasks from `navigation_base_test_id187`.
- Semantic task selection: `data_source=navigation_base_test_id187`, seed `2`.
- Exactly one newly generated rollout; this is not a reconstruction of the historical ID185 stochastic trajectory.
- The runtime runner verifies the full five-category test asset composition, explicit seeds 1--60, and zero overlap with train scenes before selecting the canary row.

## Checkpoint and frozen modules

- Exact read-only restore: ID184 `global_step_20`, actor source step `796`.
- Checkpoint: `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-17/184_continue_k4schemeb_jointupdate_dp8_tp8_u20_from10_train3x60_b24_t20_s100_c1_a1_b85p78297006578457_t1_cot07p095_val5x8_retry1/checkpoints/global_step_20`.
- Active snapshot: `sha256:6648780b3791cb4b937974b151b9e119ed9bf74602d1bc21dabfc30a3914d969`, source step `796`.
- Actor, vision tower, SharedSlotProjector, WM predictor, ValueHead/critic, and reference behavior are all frozen.
- `val_only=true`; no optimizer update and no new checkpoint are permitted.

## Policy and planner

- Scheme-B `alpha=1`, `beta=85.78297006578457`, prior temperature `1`.
- K4 history `1`, horizon `4`, `100` UCT simulations per nonterminal turn, exploration constant `1.0`.
- Rollout TP8/DP1 with `mm_encoder_tp_mode=data`; actor restore topology DP8.
- Maximum environment actions: `20`; policy sampling temperature `0.7`, top-p `0.95`.

## Identity, output, and resume

- W&B project: `vagen`.
- W&B run name: `187_smoke_rollout_browser_k4_dp8_tp8_source20_base_seed2_t20_s100_preempt_retry17`.
- W&B run ID: `nimloth-id187-smoke-rollout-browser-k4-source20-seed2-preempt-r17`; `resume=never`.
- Output: `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-21/187_smoke_rollout_browser_k4_dp8_tp8_source20_base_seed2_t20_s100_preempt_retry17`.
- The output and W&B identities must not exist before launch. No resume or overwrite is allowed; a failed retry would require a new ID/name/output.
- Unified browser target: `evaluation_browser/global_step_20/index.html`.

## Slurm and runtime ownership

- Partition `preempt`; one node × 8 H800, 64 CPU, 256 GiB; 5-hour limit.
- Expected healthy runtime: approximately 10--15 minutes after allocation.
- Allocation exclusions: `dgx-09,dgx-13,dgx-32,dgx-51`.
- Navigation/Ray head exclusions: `dgx-09,dgx-10,dgx-13,dgx-23,dgx-32,dgx-37,dgx-51`.
- The current allocation must dynamically pass a real FloorPlan1 direct-render probe within 150 seconds before Ray/environment launch.
- A Slurm-owned top-level controller owns Ray, environment, rollout, log capture, and cleanup. Runtime-root process ownership gates apply on all four nodes.

## Completion gates and monitoring

A successful canary requires all of the following:

1. Source checkpoint, snapshot, topology, dataset, W&B uniqueness, and output nonexistence preflights pass.
2. One real rollout completes with exactly one unique `(rollout_sample_id, rollout_repeat_index)`.
3. Browser `manifest.json`, per-rollout `rollout.json`, PNG/HTML files, per-turn float32 state `.npz`, and `complete.json` exist; tensor key/dtype/shape/finite checks and all SHA256 checks pass.
4. Browser and legacy single-rollout audit agree on identity, success, turn count, and complete MCTS capability.
5. Validation batch journal reports exactly one batch and one row.
6. No optimizer update and no checkpoint directory are produced.
7. W&B has only the evaluation step and finishes normally.
8. Four-node Ray logs are captured before cleanup; runtime-owned process audit is empty afterward.
9. `final_status.json` reports `passed`.

The rollout outcome itself may be success or failure; `ID185_VIS_EXPECTED_OUTCOME=any` changes only final validation and never policy behavior.
