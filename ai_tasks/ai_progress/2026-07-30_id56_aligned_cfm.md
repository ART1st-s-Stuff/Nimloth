# ID56-aligned CFM retraining

## Goal

Train a CFM on actual DINO-grid states from the exact ID56 epoch-2 online
backbone/projector, then rerun the matched-noise actual versus autoregressive
WM-predicted reconstruction.

## Fixed contract

- Source checkpoint: ID56 `train_ws16/epoch_002`; online backbone weights.
- Data: ID52 migrated train/val JSONL; 59,269 train states and 6,054 disjoint
  validation states.
- Input acceleration: immutable ID53 compact preprocess cache.
- Cache build: frozen Qwen/projector, `[16,1024]` float16 states, atomic shards,
  exact-contract resume.
- CFM: only `TokenConditionedFlowUNet` trains; old ID45 CFM is a strict
  identical-architecture initialization, with a fresh optimizer.
- Recipe: 128px, batch 32, 15 epochs, lr `3e-5`, weight decay `1e-4`, condition
  dropout `0.15`, best checkpoint selected by disjoint validation flow MSE.
- Final eval: 40 held-out trajectories, t+1 through t+4, Euler50, CFG2,
  matched noise; old SFT1 CFM remains a control and the new CFM decodes only
  ID56 actual/predicted states.

## Progress

- Created isolated branch/worktree `exp/id56-cfm-retrain`.
- Implemented distributed resumable state-cache builder.
- Extended direct CFM trainer for multi-token grid caches and strict warm start.
- Extended evaluator with an optional aligned-CFM path while preserving legacy
  ID47 CLI behavior.
- Added Slurm lifecycle scripts for 16-GPU cache, 1-GPU CFM, and 1-GPU eval.
- Added targeted unit tests; local syntax and full remote tests are pending.

## Pending

- Receive explicit partition/GPU confirmation.
- Commit/push, create clean remote worktree, initialize `external/le-wm`.
- Run remote tests and real-artifact CPU/GPU preflight.
- Launch and monitor cache, CFM, and aligned reconstruction.
- Run on-experiment-end audit and record final metrics/artifacts.
