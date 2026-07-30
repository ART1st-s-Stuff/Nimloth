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

- Run the fixed matched-noise reconstruction sweep over the completed ID50
  checkpoints and the unchanged ID45 initialization.
- If a trained checkpoint beats the initialization, run one final visual audit
  for that checkpoint; otherwise retain the failed-quality conclusion and do
  not claim compensation success.

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

## Completed lifecycle 498310-498312

- Commit `93fe4bb6`; preempt 16-GPU cache 498310 completed on `dgx-[55-56]`
  in 11:28 with exit `0:0`.
- Train manifest: 59,269 transitions, 118,538 pairs, split equally between
  actual-current and WM-predicted-next; fingerprint `22c62903361fda39`.
- Disjoint val manifest: 6,054 transitions, 12,108 equally split pairs;
  fingerprint `5eb34b5263a0afbd`. Completion summary and exact row semantics pass.
- ID50 CFM 498311 completed on `dgx-03` in 47:17 with exit `0:0`. It trained
  29,640 steps; the recorded best is step 10,000 / subset flow MSE `0.0326791`,
  while the final checkpoint full-val flow MSE is `0.0409250`. The best payload
  has 180 finite model tensors and SHA256
  `1390b39fb30e94a696f167bf961bc14d0bbc3fd3cb1d74a4d8db778f8029c9c5`.
  W&B run `g5wsfasm` is finished.
- ID51 498312 completed on `dgx-03` in 1:48 with exit `0:0`; W&B `p832xdxr`
  finished. The exact 160-row matched protocol produced actual/predicted image
  L1 `0.281274/0.298919`, worse than ID47 `0.240510/0.255730`. Predicted-better
  frame fraction also fell from `0.4625` to `0.38125`. The training mechanics
  succeeded, but the requested decoder compensation did not.
- Audit found checkpoint selection used `seed + step` for validation flow noise
  and time, so the stored best compares different random statistics. `E0076`
  records the failure; future training now uses one fixed validation seed in
  metadata and resume invariants. A frozen diverse40 checkpoint sweep is the
  next bounded diagnostic before considering any second training run.
