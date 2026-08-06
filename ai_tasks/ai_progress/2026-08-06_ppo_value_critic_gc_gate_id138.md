# 2026-08-06: checkpointed long-prefix PPO ValueHead gate and retry ID138

## Status

- The resource-only hold expired before exact human confirmation. Job `508346`
  reached its eight-hour limit at `2026-08-07T01:27:42+08:00` and ended
  `TIMEOUT` after 8:00:12 with allocation exit code 0:0.
- No ID138 GPU probe, rollout, gate, W&B run, consumption, optimizer step or
  checkpoint ever started. The exclusive output, adjacent staged-controller
  log and iteration-progress path are all absent, so ID138 itself did not fail
  and remains an unused identity subject to a fresh W&B/output check.
- Functional runtime commit:
  `66a7afde822547a4517a2c5b7e18c2e2a9ef62b9`. The server worktree is detached
  at this commit; VAGEN/LeWM are pinned to `192c35a9`/`8edfeb33`.
- A new 1x8 hold and a repeated live preflight are required before launch.
  The post-timeout resource snapshot found idle normal 8-GPU nodes
  `dgx-26/31/35/37/52/54`; known exclusions `dgx-37/51` remain, and any newly
  selected node still requires the exact renderer probe.

## Purpose and objective

- Verify on real long prefixes that the planner Qwen actually enters gradient
  checkpointing after `from_pretrained()` and that ValueHead PPO critic
  gradients reach the Qwen language body without supervising action-token
  logits or `lm_head`.
- Only after that non-consuming gate passes, execute exactly one resumed
  production-shaped update from global step 15 to 16.
- Objective is checkpoint-recorded
  `receding_horizon_decision_state_ppo_value_v1`: executed-action
  `Q(s_t,a_t)`, gamma 1, zero truncated bootstrap, ValueHead clip 0.2 and four
  critic epochs. Actor/token PPO and reference KL remain disabled.
- Trainable in the formal update: Qwen language body, WM predictor and
  ValueHead. Frozen: Qwen vision tower, StateProjector, `lm_head`, DINO teacher
  and direct-Qwen actor/token policy. The gate isolates the critic by freezing
  WM predictor while retaining the same differentiable Qwen-to-ValueHead path.

## Initialization, data and evaluation

- Immutable initialization and optimizer resume checkpoint:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/134_ppo_value_retry3_parentfix_sft2ep1_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8/train/policy_inputs/iter_0016`.
- `rl_state.pt` is 13,090,012,345 bytes and records iteration/global step 15,
  replicated optimizer state, training world size 4, full Qwen tuning, frozen
  vision, trainable WM, H1 greedy planner and the exact PPO value objective.
- Config:
  `configs/training/rl/planner_greedy_h1_smoke_iter16_16rollout_8gpu_1x8.yaml`.
  It stops after iteration 16: strict 16 fresh trajectories, maximum 20 actions
  and at most three attempts per episode.
- Training rollout uses the actual VAGEN `base_train` and
  `common_sense_train` assets, eight episodes per dataset and per-dataset seeds
  121--128. Live validation found 1,200 tasks in each train asset, 60 in each
  held-out `base`/`common_sense` asset, and zero same-category train/eval scene
  overlap. No held-out evaluation is due at iteration 16.
- ID136/137 trajectories are forbidden from reuse. ID138 first creates a new
  merged manifest under its exclusive output. The GPU gate validates but does
  not begin consumption; only the formal optimizer/checkpoint path may commit
  it.

## Identity and outputs

- W&B entity/project:
  `art2nd-hong-kong-university-of-science-and-technology/nimloth-rl`.
- ID/comment/params:
  `138_smoke_gc_longprefix_gate_resume15_rl16_fresh16_greedyh1_k16_dino05_ppo4_1n4r2g_2xtp4`.
  `smoke_gc_longprefix_gate` denotes the bounded checkpointing/memory gate;
  remaining fields encode resume step, single update, fresh batch, objective and
  1-node/4-rank/2-GPU-replica/two-TP4 topology.
- Exclusive output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/138_smoke_gc_longprefix_gate_resume15_rl16_fresh16_greedyh1_k16_dino05_ppo4_1n4r2g_2xtp4`.
- Adjacent controller log is the same path plus
  `.staged_controller.log`; the gate output is
  `gpu_gate_longprefix/` inside the run. Live preflight found the W&B name has
  zero matches and all three paths absent.
- A successful update must atomically commit manifest consumption at global
  step 16 and produce complete `train/final` model, aux-module, optimizer and
  `rl_state.pt` artifacts. Any failure makes ID138 non-resumable and forbids
  reusing its rollout.

## Entry point and exact staged command

- Python:
  `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`.
- Top-level batch-step controller:
  `experiments/training/rl/run_ppo_value_gc_gate_then_train_on_hold.sh`.
  It runs, in one attached Slurm step, two-TP4 fresh rollout/strict merge,
  long-prefix non-consuming GPU gate, then train-only if and only if the gate
  exits successfully. The default parallel runner's existing `all` behavior is
  unchanged.
- Exact execution is one `srun --jobid=508346 --overlap -w dgx-52
  --gres=gpu:8` with the paths/identity above, `EXPECTED_COMMIT=66a7afde...`,
  iteration 16, initial step 15, seed offset 121, environment ports 9730/9731,
  gate rendezvous port 29346 and training rendezvous port 32830. The outer
  controller is capped by `timeout --signal=TERM --kill-after=30s 7200s`.
- Before this command, the same allocation runs concurrent exact-visibility
  AI2-THOR probes for physical GPUs 0 and 4, each capped at 150 seconds. Both
  must emit `AI2THOR_RENDER_OK`; otherwise ID138 is not started.

## Resource and monitoring bound

- One existing H800 node with eight GPUs; no additional queued allocation.
  Expected fresh rollout is 10--20 minutes, gate at most 20 minutes, and the
  one resumed update is expected in 10--90 minutes because the long-prefix
  checkpointed path has no real-GPU timing yet.
- The combined staged controller is capped at two hours: at most 16 GPU-hours.
  The preflight snapshot left about 2 hours 49 minutes on the outer hold, so the
  cap fits without extending or replacing the allocation.
- Monitor both formal navigation prewarms, two distinct TP4 EngineCore warmups,
  strict 16/16 merge, per-rank gate JSON, GPU peaks, Qwen/ValueHead gradients,
  four critic epochs, synchronized replicas, finite formal metrics,
  consumption and the final checkpoint. Do not call rollout success or a gate
  pass policy-quality evidence.

## Pass/fail and cleanup

- Gate pass requires selected real prefixes of at least 14,000 state tokens in
  single and both DDP ranks, at least one active checkpoint-enabled Qwen module,
  nonzero Qwen-language and ValueHead gradients, absent `lm_head`/frozen-module
  gradients, four finite critic PPO epochs/AdamW steps, synchronized DDP
  gradient and parameter witnesses, and no CUDA/NCCL failure.
- Formal pass requires exactly one finite synchronized global-step-16 update,
  committed fresh consumption and a complete final checkpoint. It establishes
  mechanics and a new resume boundary, not policy improvement.
- Rollout, gate or train failure stops later phases, preserves the original
  traceback and artifacts, and prohibits a same-identity retry. After either
  success or failure, run the on-experiment-end hook, finalize README/progress
  and W&B status as applicable, and verify no Unity/VAGEN/vLLM/Ray/training
  process, GPU allocation or selected-port listener remains inside the hold.

## Completed preflight evidence

- Local diff/shell/Python checks passed. Server focused tests passed the four
  staged/gate contracts and all seven distributed-mode tests. The server
  tracked worktree is clean at exact `66a7afde`; LeWM retains only its existing
  untracked `__pycache__/`.
- Live checkpoint/config/data/output/W&B checks passed. `dgx-52` exposed no GPU
  compute application, selected ports 9730/9731/29346/32830 had no listeners,
  and exact-name process checks found no Python, Ray, vLLM or Unity residue.
- Remaining pre-launch actions are the concurrent renderer probes, immediate
  resource/output/W&B refresh, and explicit human confirmation of this exact
  two-hour/16-GPU-hour contract.

## Hold terminal update

- The `508346` timeout was only the terminal state of the resource reservation;
  it produced no ID138 output and consumed no trajectory. There is no ID138
  README/metadata under server `outputs/` to update.
- The recovery boundary remains the immutable ID134 global-step-15
  `policy_inputs/iter_0016` checkpoint. Launching later requires a new hold but
  not a new experiment ID, because the exact ID138 W&B/output identity was
  never created.
- The current live resource snapshot reports 68/88 normal GPUs free, including
  five non-excluded idle 8-GPU candidates (`dgx-26/31/35/52/54`). This is a
  volatile availability observation, not a reservation; no new job has been
  submitted.

## Confirmed relaunch allocation and renderer gate

- Human explicitly confirmed this exact ID138 contract. A fresh live W&B query
  again returned zero exact-name matches and all ID138 output/controller paths
  remained absent. The only submitted hold is Job `508866`: normal one node,
  eight GPUs, 128 CPUs, 96 GiB and 2:30 walltime, excluding
  `dgx-32/37/51`. It started on `dgx-26:8` at
  `2026-08-07T02:48:38+08:00`.
- Renderer attempt 1 stopped before importing AI2-THOR because the one-off
  command omitted `PYTHONPATH=${REPO}/src`; its logs were preserved and it
  created no ID138 output, W&B, rollout or GPU conclusion. This was a launch
  command environment error, not a node/render failure.
- Corrected attempt 2 used isolated homes, the shared verified AI2-THOR release
  cache and the exact formal visibility contract. Physical GPU0 and GPU4 each
  exposed only itself as `CUDA_VISIBLE_DEVICES=0/4` with relative
  `gpu_device=0`; both emitted `AI2THOR_RENDER_OK` in 16.862/17.115 seconds with
  255x255 frames and dynamic range 246. Evidence is under
  `outputs/experiments/training/rl/preflight/2026-08-07/138_dgx26_exact_visibility_508866/attempt2`.
- ID138 itself is still not started at this checkpoint. Immediately before the
  staged step, refresh hold state, exact runtime commit, GPU/process/port state,
  W&B uniqueness and output absence once more.

## Terminal ID138 result

- After human confirmation, staged step `508866.3` started under hold `508866`
  on `dgx-26:8` from immutable runtime
  `66a7afde822547a4517a2c5b7e18c2e2a9ef62b9`.
- Two TP4 workers completed a fresh strict 16-trajectory merge: eight
  `base_train` and eight `common_sense_train` episodes with independent
  per-dataset seeds 121 through 128. The fresh manifest exists and no
  consumption sidecar was committed.
- The single-GPU gate passed on a real 16,184-token final prefix. It observed 37
  active gradient-checkpointed Qwen modules, nonzero Qwen and ValueHead
  gradients, absent vision/StateProjector/lm_head gradients, and a
  17,095,888,384-byte peak allocation.
- The two-rank gate failed closed before its first optimizer step. Rank 1's
  rank-owned trajectory subset had a longest final prefix of only 11,332
  tokens, below the 14,000-token memory contract. This was neither an OOM nor a
  PPO gradient-path failure. The staged step ended at
  `2026-08-07T03:05:03+08:00`; formal training, W&B, global step 16 and
  trajectory consumption never started.
- ID138 is terminal and cannot be resumed. Its rollout remains diagnostic-only
  and is forbidden from training reuse. A retry requires a new identity, a
  corrected non-consuming gate selection contract, and fresh rollout from the
  unchanged ID134 committed-global-step-15 boundary.
