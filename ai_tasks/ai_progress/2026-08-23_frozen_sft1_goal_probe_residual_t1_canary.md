# Frozen-SFT1 goal probe and Residual-T1 canary

## Authorization

Human approved both experiments:

- T1 design: freeze ID176 actor/backbone/vision and the SFT1 `SharedSlotProjector`; train only a zero-initialized one-step residual WM; raw DINO is evaluation-only and is not a training loss.
- Resources: Slurm `normal`, one H800 total, at most one hour, probe then canary sequentially.

No projector calibration, T2/T4, ValueHead, MCTS or RL training is authorized by this task.

## Goal

1. Determine whether frozen deployed-actor + SFT1-projector state contains goal information beyond a matched frozen-DINO visual baseline using a low-capacity learned probe.
2. Determine whether a zero-initialized residual T1 WM trained in that fixed state space beats repeated-copy for each well-supported major action.

## Data and split

- Pre-RL ID52 terminal-CoT archive only; no ID189, post-RL or RL data.
- States use ID176 deployed actor hidden and the frozen SFT1 projector with each archived observation's actual recorded CoT.
- Use exact early transitions at steps 0--3 and their exact next decision state/terminal CoT.
- Actual dataset identity is parsed from each referenced source row's `config_id`; migrated and source `env_seed` must match exactly.
- Preflight verified train seeds 1--1080 and validation seeds 1081--1200, with zero overlap in `(actual_eval_set, env_seed)` and source UID.
- The migrated category mismatch (587 train, 67 validation) is recorded but never used as actual task identity.

## ID60 goal probe contract

- Extract one immutable float32 cache of unique early decision states and original-observation DINO grids.
- Aggregate step-0 records by actual task identity so duplicate archive rows do not overweight a task.
- Train matched low-capacity linear probes on slot-mean ID176+SFT1 state and slot-mean DINO features.
- Hyperparameter selection uses only a deterministic task-grouped inner split of the pre-RL training data; external validation is evaluated once.
- Report micro/macro top1, top5, NLL, represented/unseen labels, majority baseline and paired bootstrap state-minus-DINO accuracy interval.
- Goal gate requires state micro and macro top1 each exceed DINO by at least 0.02, state micro exceed majority, and the paired bootstrap 95% lower bound exceed 0.
- Trainable: diagnostic linear readout only. Frozen: actor, Qwen, vision, SFT1 projector and DINO. Probe weights are diagnostic artifacts and cannot become the state representation.

## ID75 T1 canary contract

- Consume the immutable ID60 state cache; do not replay or update Qwen/projector.
- Model: existing temporal-spatial grid predictor as a feature body plus a new zero-initialized delta head; prediction is exactly `z_t + delta(z_t,a_t)` before the first update.
- Loss: fixed-state next-state MSE only. No raw DINO, goal, ValueHead, SIGReg, CE or RL loss.
- Use deterministic task-grouped inner train/selection split and action-balanced training weights; external pre-RL validation is evaluated once.
- Report natural-distribution overall and per-action copy-relative skill, RMSE, predicted/actual std ratio and next-DINO metrics.
- Primary action gate applies only to actions with at least 20 external validation transitions; preflight counts make actions 0,2,3,4 primary. Actions 1 and 5 remain reported but cannot be acceptance gates.
- Canary gate: all primary-action state skills `>0`, macro primary skill `>0`, overall skill `>0`, predicted/actual std ratio in `[0.9,1.1]`, and predicted next-DINO RMSE no worse than copy.
- The stronger candidate signal `overall skill>0.2` is reported separately and is not silently substituted for the canary gate.
- Trainable: residual T1 predictor only. Frozen/absent: actor, Qwen, vision, SFT1 projector, DINO, ValueHead, planner and policy.
- Save a fresh canary checkpoint and optimizer-free inference metadata. It is not approved for downstream use unless gates pass and the human separately approves continuation.

## Outputs and identities

- ID60 W&B: project `nimloth-recon`, run `60_id176_sft1_frozen_goal_probe_early4_k16`, ID `nimloth-recon-id60-frozen-state-goal-probe`.
- ID75 W&B: project `nimloth-sft2`, run `75_frozen_sft1_residual_t1_canary_early4_k16`, ID `nimloth-sft2-id75-frozen-sft1-residual-t1-canary`.
- Fresh outputs under server `outputs/experiments/evaluation/state_alignment/.../60_*` and `outputs/experiments/training/sft2/.../75_*`.
- Neither experiment may overwrite or resume ID74, ID59 or any previous output.

## Progress

- [x] Human approved design and resource boundary.
- [x] Actual source seed alignment and train/validation task-key disjointness checked.
- [ ] Write RED tests and experiment contracts.
- [ ] Implement immutable state cache and matched goal probe.
- [ ] Implement zero-copy-initialized residual T1 predictor and canary trainer.
- [ ] Complete local and clean remote CPU gates.
- [ ] Run sequential normal 1xH800 job and monitor to completion.
- [ ] Validate artifacts, update progress and report decision.
