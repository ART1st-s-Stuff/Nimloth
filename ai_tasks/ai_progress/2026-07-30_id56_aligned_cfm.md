# ID56-aligned CFM retraining

## Goal

Train a CFM on both actual-current and frozen-WM-predicted-next DINO-grid states
from the exact ID56 epoch-2 checkpoint, then rerun the matched-noise actual
versus autoregressive WM-predicted reconstruction.

## Fixed contract

- Source checkpoint: ID56 `train_ws16/epoch_002`; online backbone weights.
- Data: ID52 migrated train/val JSONL; 59,269/6,054 transitions become
  118,538/12,108 balanced CFM pairs.
- Input acceleration: immutable ID53 compact preprocess cache.
- Cache build: frozen Qwen/projector/WM; each transition contributes actual
  current -> current image and WM predicted next -> actual next image. States
  are `[16,1024]` float16 with atomic shards and exact-contract resume.
- CFM: only `TokenConditionedFlowUNet` trains; old ID45 CFM is a strict
  identical-architecture initialization, with a fresh optimizer.
- Recipe: 128px, batch 32, 8 epochs, lr `3e-5`, weight decay `1e-4`, condition
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
- Human clarified that CFM should compensate frozen-WM errors like ValueHead;
  replaced the interim actual-next design with predicted-next/actual-next pairs.
- Local Python/shell syntax and staged diff checks pass.
- Corrected predicted-next pair tests pass remotely: `20 passed, 1 warning`.
- Real artifact preflight passes: 59,269/6,054 exact rows, ID53 fingerprints
  `ac7835348d6eade1`/`d857dc4ef51a70be`, ID56 `[16,1024]` projector output,
  and strict ID45 CFM architecture/weight initialization at source step 29,000.
- ID56 H=1 WM predictor strict-load preflight passes: 92,418,120 parameters,
  finite `[1,16,1024]` single-step output.
- Live resources: preempt 36 free H800, normal 1. ID48/ID49 are unused in both
  server outputs and live W&B.

## Pending

- Launch and monitor cache, CFM, and aligned reconstruction.
- Run on-experiment-end audit and record final metrics/artifacts.

## Failed lifecycle 498307-498309

- Commit: `e8db6222`; intended cache/CFM/eval contract was the fixed contract
  above, using preempt 16/1/1 H800 and IDs 48/49.
- Cache 498307 allocated `dgx-[55-56]` but bare `srun` resolved to the cluster
  warning wrapper. It exited zero after three seconds without creating the
  cache directory or `summary.json`.
- CFM 498308 then failed before output/W&B initialization because the cache
  summary was absent. Eval 498309 was cancelled. There are no checkpoints or
  resumable artifacts, so IDs 48/49 will not be reused.
- Fix: use absolute Slurm binaries plus authoritative `SLURM_CONF`, and fail the
  batch job unless the cache completion summary exists. Retry identities are
  ID50/ID51.
