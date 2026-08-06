# 2026-08-07: ID141 detached diagnostic-gate PPO step 16 retry

## Identity and launch

- Purpose: complete the same formal PPO ValueHead step-16 smoke after ID140's
  attached `srun` was cancelled solely by SSH-session closure.
- W&B entity/project:
  `art2nd-hong-kong-university-of-science-and-technology/nimloth-rl`.
- ID/run:
  `141_smoke_diagcorpus_gate_detached_resume15_rl16_fresh16_greedyh1_k16_dino05_ppo4_1n4r2g_2xtp4`;
  exact-name query returned zero matches.
- Output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/141_smoke_diagcorpus_gate_detached_resume15_rl16_fresh16_greedyh1_k16_dino05_ppo4_1n4r2g_2xtp4`.
- Runtime commit: `dbcadc53938d05e3ada56a3a2e6006164c502dcc`;
  server worktree
  `/project/peilab/atst/nimloth/.worktree/ppo-value-critic-9ef56fc9`;
  config
  `planner_greedy_h1_smoke_iter16_16rollout_8gpu_1x8.yaml`.
- Existing normal hold `508866`, `dgx-26:8`. Launch one 8-GPU `srun` client
  with `nohup`, stdin `/dev/null`, durable client PID/log, and a 5,400-second
  controller timeout. A preceding detached probe, step `508866.15`, survived SSH
  exit and completed `0:0` in 16 seconds with `DETACHED_SRUN_OK`.

## Data and objective

- Formal input is ID141 fresh rollout only: `base_train` and
  `common_sense_train`, eight episodes each, per-dataset seeds 145 through 152,
  max 20 steps and at most three attempts. ID138/139/140 rollouts are forbidden
  from formal training reuse.
- Gate input remains the read-only ID138 behavior-checkpoint-matched diagnostic
  corpus with the real 16,184-token final prefix. Controller `dbcadc53` rejects
  diagnostic paths under formal `RUN_OUT` and logs both paths.
- Resume from ID134 `train/policy_inputs/iter_0016`, committed global step 15.
  Objective is `receding_horizon_decision_state_ppo_value_v1`, executed-action
  `Q(s_t,a_t)`, gamma 1, zero bootstrap, value clip 0.2, four PPO critic epochs,
  greedy H=1/K=16, history 1 and DINO weight 0.5.
- Train Qwen language body, WM predictor and ValueHead. Freeze Qwen vision,
  StateProjector, lm_head, DINO teacher and actor/token PPO.

## Terminal contract

- Stages: fresh two-TP4 strict rollout/manifest; non-consuming 14k real-prefix
  single/DDP gate; formal train only after gate passes.
- Success requires exact global step 16, ID141 consumption committed, complete
  `train/final/rl_state.pt`, and finished W&B. No held-out 120-episode evaluation
  is due at iteration 16; this is not policy-quality evidence.
- Any stage failure terminates ID141 and forbids formal reuse of its rollout.
  Monitor durable srun/stage logs, Slurm step, rollout counts, all gate JSONs,
  formal train logs, consumption, final metadata, W&B and cleanup.

## Completed result

- Detached staged step `508866.16` completed `0:0` in 8 minutes 32 seconds and
  the stage log ends `complete/all_passed`. Rollout step `508866.17` completed
  in 3 minutes 32 seconds. The fresh strict merge contains 16 trajectories and
  283 raw environment steps; the trainer reported 204 eligible actor
  transitions.
- The gate contract read the separate ID138 diagnostic trajectory/manifest.
  Single/DDP modes used the real 16,184-token prefix; both two-GPU ranks
  completed four PPO/AdamW epochs with synchronized witnesses and a ValueHead
  parameter delta of `0.00039300043135881424`.
- Formal training completed iteration/global step 16/16 in 95.2 seconds. Final
  metrics were finite: `wm_mse=0.20356984884710982`,
  `dino_grid_mse=0.8987376613076776`,
  `value_loss=48.13194083637665`, and
  `total_loss=48.78487959849189`; critic PPO epochs were 4 and actor/token loss
  plus policy tokens remained zero.
- The ID141 consumption sidecar is committed from starting global step 15 to
  committed step 16. `train/final` contains a 13,090,012,345-byte `rl_state.pt`,
  full Qwen weights, StateProjector, ValueHead and WM predictor. Checkpoint
  metadata records objective `receding_horizon_decision_state_ppo_value_v1`,
  value clip 0.2/four epochs, zero bootstrap, training world size 4, replicated
  optimizer state, Qwen full tune and frozen vision.
- W&B run `ifzt62xg` is `finished`:
  `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/nimloth-rl/runs/ifzt62xg`.
  Its train-rollout success rate 0.6875 is not held-out policy-quality evidence;
  no 120-episode held-out evaluation was due at iteration 16.
- Post-run cleanup found no GPU compute process or listener on ports 9760, 9761
  or 32860. The server output README is marked completed. Hold `508866` was
  then released; it is absent from `squeue`, while completed ID141 steps remain
  recorded `COMPLETED 0:0` in Slurm accounting.
