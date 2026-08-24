# ID191 — State-interface direction canary

## Authorization and purpose

The human approved a controlled canary to determine the next clean retraining
direction. ID191 is not authorized to repair or resume ID74/ID75, and its
checkpoint cannot enter WM, ValueHead, MCTS or RL.

The canary answers two causal questions:

1. Does frozen same-generation ID176 K16 hidden retain goal and lateral
   collision information that the SFT1 projected state weakens?
2. Can a bounded low-capacity correction recover both signals without damaging
   the SFT1 visual anchor?

## Data and state semantics

- Reuse immutable pre-RL ID60 state/DINO cache and exact ID52 train/validation.
- Recompute ID176 hidden from each observation's actual archived CoT; fixed or
  placeholder CoT is forbidden.
- Reprojection through the frozen SFT1 projector must reproduce every ID60
  state within a strict numerical tolerance before training.
- Preserve one unified `[16,1024]` visual-goal state. Goal and action-success
  heads are supervision/readouts, not independent state branches.
- Exact `last_action_success` is a training label only; it is never an input.
- External validation retains ID60 exact initial/current/next-image exclusion.
  Row-level task identity remains unavailable, so no task-generalization claim.

## Intervention

Freeze ID176 actor/Qwen/vision and the complete SFT1 projector. Train only:

- a shared per-slot hidden `2048 -> rank64 -> 1024` residual adapter;
- a training-only goal head;
- a training-only action-specific movement-success head;
- fresh diagnostic linear readouts.

The residual output layer is exactly zero initialized and every sample is
bounded to at most 10% of the frozen SFT1 state's Frobenius norm:

`z_candidate = z_sft1 + bounded_adapter(h_same_generation)`.

Visual cosine, goal CE and outcome BCE are normalized to dimensionless
reference losses; a residual-energy anchor limits state drift. Inner selection
keeps exact initial-image groups intact. The selected epoch is retrained fresh
on all pre-RL train rows before one external evaluation.

## Gates

- Hidden direction gate: hidden must improve the failed goal interface and not
  trail projected state on either lateral action.
- Visual gate: external candidate DINO RMSE/cosine cannot regress and the 10%
  residual bound must hold.
- Goal gate: candidate must satisfy the existing ID60 micro/macro, majority and
  paired-bootstrap requirements against DINO.
- Outcome gate for every movement action: candidate AUC lower CI above chance,
  not more than 0.02 below DINO, lateral actions significantly improve the
  original state readout, and Brier score beats the constant train-rate baseline.
- Any failed component fails the canary. A completed job with `gate=false` is a
  valid diagnostic result, not permission to enlarge the model.

## Runtime and identity

- One H800 on `normal`, 45-minute limit; excluded nodes follow project policy.
- Fresh output:
  `outputs/experiments/training/sft2/2026-08-24/191_state_interface_direction_canary`.
- W&B project `nimloth-sft2`, run ID
  `nimloth-sft2-id191-state-interface-canary`, resume forbidden.
- Shared `.env` is sourced before locked W&B values are exported; initialized
  project/run identity is verified fail-closed.
