# 2026-08-07: ID139 fresh rollout, long-prefix gate, and PPO ValueHead step 16

## Purpose and identity

- Purpose: retry the complete step-16 online RL pipeline after the
  global-qualified long-prefix gate passed its real-GPU diagnostic. This is a
  mechanics/training smoke, not policy-quality evidence.
- W&B entity/project:
  `art2nd-hong-kong-university-of-science-and-technology/nimloth-rl`.
- ID: `139`; comment: `smoke`; params:
  `gc_longprefix_gate_resume15_rl16_fresh16_greedyh1_k16_dino05_ppo4_1n4r2g_2xtp4`.
- Full run name:
  `139_smoke_gc_longprefix_gate_resume15_rl16_fresh16_greedyh1_k16_dino05_ppo4_1n4r2g_2xtp4`.
- Output:
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-06/139_smoke_gc_longprefix_gate_resume15_rl16_fresh16_greedyh1_k16_dino05_ppo4_1n4r2g_2xtp4`.
- Exact-name W&B query returned zero matches before launch. Output, adjacent
  staged-controller log and srun log must all be absent before launch.

## Immutable runtime and entrypoint

- Runtime commit: `fddbaef867ed9656538c8e6fff140d3851dd6813`.
- VAGEN submodule: `192c35a91f3941b72d5e1272af6603ef7a7d93e0`.
- Server worktree:
  `/project/peilab/atst/nimloth/.worktree/ppo-value-critic-9ef56fc9`.
- Staged entrypoint:
  `experiments/training/rl/run_ppo_value_gc_gate_then_train_on_hold.sh`.
- Config:
  `configs/training/rl/planner_greedy_h1_smoke_iter16_16rollout_8gpu_1x8.yaml`.
- Allocation: existing hold Job `508866`, `dgx-26:8`, normal partition. The
  staged controller runs as one attached 8-GPU `srun` step; no new allocation is
  submitted.

## Data, checkpoint, and objective

- Resume checkpoint:
  `134_ppo_value_retry3_parentfix_sft2ep1_rl16_eval10x120_greedyh1_k16_dino05_ppo4_iter60_1n4r2g_2xtp4_normal1x8/train/policy_inputs/iter_0016`.
  It is the committed global-step-15 boundary with complete optimizer state.
- Fresh rollout only: `base_train` and `common_sense_train`, eight episodes per
  dataset, independent per-dataset seeds 129 through 136, at most 20 environment
  steps and at most three attempts per requested trajectory. ID138 rollout is
  diagnostic-only and forbidden from training reuse.
- Objective:
  `receding_horizon_decision_state_ppo_value_v1`, executed-action
  `Q(s_t,a_t)`, gamma 1, zero truncated bootstrap, PPO value clip 0.2 and four
  critic epochs.
- Planner: greedy H=1, K=16, history size 1, DINO supervision weight 0.5.

## Train/freeze and staged fail-closed contract

- Train: Qwen language body, WM predictor and ValueHead.
- Freeze: Qwen vision, StateProjector, lm_head, DINO teacher and actor/token PPO.
- Rollout: two TP4 workers produce a strict 8+8 fresh merge and fresh-policy
  manifest. A failed trajectory attempt is isolated and retried; it does not
  terminate the batch unless its requested trajectory exhausts all attempts.
- Non-consuming gate: every single/DDP rank must use a real final prefix with at
  least 14,000 state tokens; all 37 Qwen checkpoint modules must be active;
  Qwen/ValueHead gradients must be nonzero; frozen/lm_head gradients must be
  absent; two model-parallel DDP ranks must complete four finite PPO/AdamW steps
  with synchronized gradients and parameters. Distinct qualifying samples are
  preferred and deterministic reuse is recorded when the real qualifying pool
  has fewer samples than ranks.
- Formal training starts only after the gate passes. Success requires exactly
  global step 16, committed trajectory consumption and a complete `train/final`
  checkpoint. No held-out 120-episode evaluation is due at iteration 16.
- Any rollout, gate or train failure terminates ID139. Its rollout may not be
  reused for training under another identity; retry starts again from the
  immutable ID134 step-15 checkpoint with a new ID and fresh rollout.

## Monitoring and budget

- Hold walltime is 2:30. At preparation it has more than two hours remaining;
  expected rollout is about 10--20 minutes, gate about 1--5 minutes, and formal
  train about 10--90 minutes. The combined staged controller has a 7,200-second
  timeout.
- Monitor stage log, both rollout shards, strict merge/manifest, all three gate
  result JSONs, train step log, consumption sidecar, final `rl_state.pt`, W&B
  run state and Slurm/process/port cleanup.

## Terminal result

- Staged step `508866.9` started from the documented runtime. The two-TP4
  rollout step `508866.10` completed in 4 minutes 5 seconds and produced a
  strict fresh 8+8 merge for seeds 129 through 136.
- The corrected global-qualified gate failed closed in `single_grad` selection:
  every fresh trajectory was short and the maximum real final-prefix length was
  4,120 tokens, below the 14,000-token memory contract. This was neither an OOM
  nor a PPO gradient-path failure.
- The staged step ended with exit 1 at `2026-08-07T03:27:30+08:00`. Formal
  training never started. `train/` is only an empty directory skeleton; no W&B
  run, optimizer step, checkpoint, or consumption sidecar exists.
- ID139 is terminal and its rollout cannot be reused for training. The next
  controller revision must use an explicit behavior-checkpoint-matched long
  diagnostic corpus for the non-consuming mechanics gate while keeping the new
  identity fresh rollout as the only formal training input. Tests must prevent
  the two paths from being mixed.
