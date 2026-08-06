# 2026-08-07: ID140 diagnostic long-prefix gate plus fresh PPO step 16

## Identity and purpose

- Purpose: complete one formal PPO ValueHead update from committed global step
  15 to 16 after removing the random requirement that each fresh training batch
  contain a 14k-token trajectory. This is mechanics/training evidence, not
  policy-quality evidence.
- W&B: entity/project
  `art2nd-hong-kong-university-of-science-and-technology/nimloth-rl`;
  ID `140`; comment `smoke`; params
  `diagcorpus_gate_resume15_rl16_fresh16_greedyh1_k16_dino05_ppo4_1n4r2g_2xtp4`.
- Full name:
  `140_smoke_diagcorpus_gate_resume15_rl16_fresh16_greedyh1_k16_dino05_ppo4_1n4r2g_2xtp4`.
- Output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/140_smoke_diagcorpus_gate_resume15_rl16_fresh16_greedyh1_k16_dino05_ppo4_1n4r2g_2xtp4`.
- Exact W&B name has zero matches before launch. Output, staged log and srun log
  must be absent.

## Runtime and allocation

- Runtime commit: `dbcadc53938d05e3ada56a3a2e6006164c502dcc`.
- Server worktree:
  `/project/peilab/atst/nimloth/.worktree/ppo-value-critic-9ef56fc9`;
  VAGEN `192c35a91f3941b72d5e1272af6603ef7a7d93e0`.
- Entry:
  `experiments/training/rl/run_ppo_value_gc_gate_then_train_on_hold.sh`;
  config
  `configs/training/rl/planner_greedy_h1_smoke_iter16_16rollout_8gpu_1x8.yaml`.
- Existing normal hold Job `508866` on `dgx-26:8`; one attached 8-GPU staged
  `srun`, no new allocation. The controller timeout is 7,200 seconds and the
  hold has about 1 hour 45 minutes remaining at preparation.

## Strictly separated data paths

- Formal training input: new ID140 rollout only, `base_train` and
  `common_sense_train`, eight episodes per dataset, independent per-dataset
  seeds 137 through 144, at most 20 environment steps and three attempts per
  requested trajectory. It must produce its own strict merge and fresh-policy
  manifest under ID140.
- Non-consuming gate input: ID138 diagnostic trajectory and matching manifest,
  produced by the same immutable ID134 behavior checkpoint. It contains the
  real 16,184-token final prefix that already passed the corrected two-rank
  GPU gate. It is read-only and can never be passed to formal training.
- Controller `dbcadc53` requires the diagnostic paths explicitly, rejects them
  if they are under formal `RUN_OUT`, logs both diagnostic and formal paths, and
  passes only ID140's own output to `run_parallel_phase train`.
- ID138/ID139 formal rollouts remain forbidden from training reuse.

## Checkpoint, objective, and module ownership

- Resume checkpoint: ID134
  `train/policy_inputs/iter_0016`, the committed global-step-15 boundary with
  optimizer state.
- Objective:
  `receding_horizon_decision_state_ppo_value_v1`, executed-action
  `Q(s_t,a_t)`, gamma 1, zero truncated bootstrap, value clip 0.2 and four PPO
  critic epochs. Planner is greedy H=1, K=16, history size 1, DINO weight 0.5.
- Train: Qwen language body, WM predictor, ValueHead.
- Freeze: Qwen vision, StateProjector, lm_head, DINO teacher, actor/token PPO.

## Fail-closed and success contract

- Stages: new two-TP4 rollout/strict merge, fixed real long-prefix non-consuming
  gate, formal train only after gate success.
- Gate requires at least 14,000 state tokens on each single/DDP rank, 37 active
  Qwen checkpoint modules, nonzero Qwen/ValueHead gradients, absent frozen and
  lm_head gradients, four finite PPO/AdamW updates, and synchronized rank
  gradient/parameter witnesses.
- Formal success requires exactly global step 16, committed ID140 trajectory
  consumption, complete `train/final/rl_state.pt` and finished W&B. No held-out
  120-episode evaluation is due at iteration 16.
- Any failure terminates ID140; its new rollout cannot be reused for training.
  The immutable retry boundary remains ID134 step 15.

## Monitoring

- Monitor Slurm/stage state, both rollout counts and errors, diagnostic gate
  contract/result JSONs, formal train/pipeline logs, `train_step_log.csv`,
  consumption sidecar, final checkpoint metadata and W&B run state. After any
  terminal state, update the output README and progress files and clean GPU
  processes/ports before releasing the hold.

## Terminal result

- Staged step `508866.12` and rollout step `508866.13` started. The fresh ID140
  rollout completed in 4 minutes 4 seconds and strictly merged 16 trajectories
  / 283 transitions for seeds 137 through 144.
- Runtime evidence confirmed strict path separation: the stage log records the
  ID138 diagnostic and ID140 formal trajectory paths, and the gate contract
  actually reads the ID138 trajectory/manifest. The single-GPU 16,184-token
  gate passed.
- While the two-rank gate was loading checkpoint shards, the interactive SSH
  session that owned the attached `srun` closed. Slurm marked step `508866.12`
  `CANCELLED by 3738`, exit `0:9`, at
  `2026-08-07T03:38:36+08:00`. This was not an OOM, data validation failure, or
  PPO failure.
- Formal training never started. There is no DDP result, train step, W&B run,
  checkpoint or consumption sidecar, and GPUs were clean after cancellation.
  ID140 is terminal; its rollout is forbidden from training reuse.
- A retry must retain the hold-then-srun workflow but detach the `srun` client
  from SSH with durable PID/log files. A short detached probe must first prove
  that the step survives SSH exit; the formal retry then requires a new ID and
  fresh rollout.
