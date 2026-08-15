# VAGEN-Lite K4 MCTS joint policy

## Goal

Integrate the corrected ID74 `history_size=1 / prediction_horizon=4` temporal-spatial WM predictor into the upstream-based VAGEN-Lite joint policy. Each real turn must generate one real CoT+K16 state and one same-forward 8-action prior, run frozen K4 UCT-MCTS, sample one Scheme-B root action, execute only that action, then replan from the next real observation.

The first experimental endpoint is an optimizer-free TP8/rank-0-co-located beta calibration gate. It must stop for human approval after measuring beta; it must not start the approved 10-update canary automatically.

## Human-approved contract

- Search: deterministic UCT-MCTS, fixed K4 on every nonterminal real turn.
- Search budget: 100 simulations, UCT exploration constant 1.0.
- Leaf score: outgoing `Q(predicted_state_3, action_3)`; no reward/done head in the first version.
- Scheme-B: raw backed-up root mean values only, combined with Qwen action logits; direct root Q is not mixed into guidance.
- Actual environment action: coordinator-keyed sample from Scheme-B, never a direct MCTS argmax override.
- Frozen-V: guided behavior distribution expectation over direct root Q, not MCTS root means.
- Replay: persist behavior-time direct Q and MCTS root scores; never recompute old guidance with a newer WM.
- WM update after calibration: continue training ID74 projector/predictor with all valid nonterminal-crossing 1--4 step targets from the behavior snapshot projector; WM loss does not backpropagate into Qwen.
- Online WM auxiliary objectives: state MSE 1.0, DINO-grid 0.5, SIGReg 0.1.
- World/critic optimizer: one AdamW with projector/predictor/ValueHead groups, each LR 1e-4; betas (0.9,0.95), eps 1e-8, WD 0.01, grad clip 1; selected-action Huber delta 1.
- Actor: LR 1e-7, PPO clip 0.2, one epoch, token KL 0.01, guided entropy 0.01, AdamW betas (0.9,0.95), eps 1e-8, WD 0.01, grad clip 1; vision/reference frozen.
- Reward: per-turn format 0.01, terminal format 0, success 1.
- Temporal credit: gamma 1.0, GAE lambda 0.95.
- Rollout: three train splits balanced; 20 real turns maximum; global batch 24; CoT temperature 0.7, top-p 0.95, full response limit 512.
- Placement: frozen planner co-located on TP8 vLLM rank 0.
- Beta: optimizer-free 24-trajectory balanced calibration, then fixed from a 1:1 median action-spread rule and presented to the human.
- Canary after beta approval: 10 updates, fresh-runtime resume after step 5, full checkpoints at steps 5 and 10, held-out 5x8 validation before/after, then held-out 5x60 evaluation before any long-training decision.

## Invariants

- Preserve the existing ID171 direct-Q path and its schemas/tests; production K4 uses explicit new schemas.
- Frozen planning snapshot identity covers projector, predictor, ValueHead, architecture, source step, contract, score dtype, and MCTS contract.
- MCTS tail actions are imagined evidence only; they never enter the executed-action ledger, reward rows, PPO actions, or critic targets.
- No second Qwen transformer replay during rollout planning.
- Terminal trace has real CoT+K16 only; no action, draw, MCTS, Q scoring, or environment step.
- Infrastructure truncation produces no training row.
- General production remains fail closed until the human approves measured beta and the complete production config.

## Plan

1. Audit existing snapshot, planner, rollout-worker capture, behavior schema, training compiler, publication, and checkpoint boundaries.
2. RED: add parent tests for full ID74 planning snapshot, K4 final-edge MCTS, direct-Q/root-mean separation, exact export/restore, and rank-0 worker behavior.
3. GREEN: implement immutable full planning snapshot and rank-0 scoring primitives while preserving direct-Q v1.
4. RED/GREEN: add VAGEN planning behavior/action-draw schemas and execution wiring that use planner root means for Scheme-B while persisting direct Q.
5. RED/GREEN: install and score the frozen planner inside the TP8 rank-0 vLLM worker; retain a CPU lifecycle/provenance owner that never performs planning compute.
6. RED/GREEN: add optimizer-free balanced calibration config, output schema, beta statistic, validator, and strict launcher/Slurm identity gates.
7. Run local and server CPU regressions, independent review, commit/push feature branches, and prepare a fresh exact-SHA server worktree.
8. Run the on-experiment-start hook, submit only the approved optimizer-free calibration, run the on-experiment-end hook, record results, and stop for human beta approval.
9. After beta approval only: implement/validate DP8 online projector/predictor/ValueHead training with real 1--4 step targets, DINO, SIGReg, publication, and exact resume before the 10-update canary.

## Current status

- Contract approved.
- Candidate implementation and ID172 calibration gate committed and pushed:
  - Parent `b758063efc6c89a1edff21178629622acf696944`.
  - VAGEN `b7c45d9c085a3076bfe416f598bccd21e2c166e5`.
  - VERL remains the exact `494f264494b2525f2c13595f63ac4912963e6d2f` gitlink.
  - le-wm remains `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- Added a full immutable projector/predictor/ValueHead K4 planning snapshot, exact file transport, direct-Q/MCTS scoring separation, K4 behavior/draw/execution/ledger schemas, TP-rank-zero worker install/score methods, a CPU lifecycle owner, and the optimizer-free balanced calibration entrypoint.
- Added the batch-owned ID172 normal/1-node/8-H800/64-CPU/256-GiB/60-minute launcher with exact-SHA, clean-tree, asset-hash, ID74-hash, fixed render/prewarm, process cleanup, all-or-nothing output, and explicit no-optimizer/no-checkpoint/no-canary gates.
- Legacy direct-Q schemas and ID171 paths remain separate.
- Fresh server worktree `/project/peilab/atst/nimloth/.worktrees/k4-mcts-c936682c` is now at exact candidate SHAs and clean at parent/VAGEN/VERL/le-wm/RCDM layers.
- ID172 Job`519634` ran on `normal/dgx-23` and failed `124:0` after2m50s at the unchanged 150-second FloorPlan1 direct-render gate. Exact SHA/clean/allocation/three-asset/ID74 hash gates passed; env service, prewarm, Ray, vLLM, model load, generation, MCTS and beta calculation never started. Planned run output was never created; control evidence is `outputs/experiments/training/rl/slurm/id172-k4-519634`, cleanup audit is empty, and dgx-23 returned 8/8 GPUs free.
- ID172 is consumed and non-resumable. ID173 kept every calibration value and run seed unchanged, used a new empty identity, and temporarily excluded dgx-23 while preserving the 150-second gate.
- ID173 Job`519648` on `normal/dgx-39` reached real TP8 ID74 generation, rank-zero K4 scoring and 24 complete balanced trajectories. MCTS median spread was finite and above`1e-8`, while median std across the eight behavior LLM action logits was exactly0; the ratio therefore produced beta0 and the implementation rejected it as non-positive. Job failed`1:0` after15m18s.
- ID74 untied LM-head action rows are nearly identical (maximum pairwise L2`0.0001990821`, max element difference one BF16 step), consistent with same-forward BF16 action logits collapsing. No positive beta is approved; beta0 would remove MCTS guidance.
- No optimizer, update, training checkpoint, W&B run or canary exists. ID173 cleanup is empty and non-resumable. Further GPU work is stopped pending the human's zero-spread/precision decision.

## Validation log

- Local AST/shell parsing, embedded Python compilation, and `git diff --check` passed.
- Server focused K4/planning/capture/legacy-schema regressions passed (`136 passed, 19 subtests`; later same-generation/rank-zero additions `23 passed, 2 subtests`; final launcher/planner/K4 set `13 passed`).
- Expanded VAGEN/VERL regression passed `199 passed, 115 subtests`.
- Expanded parent RL/WM set had `353 passed` and 19 failures, all from legacy `planner_verl_adapter.py` hard-coding old VERL `084f042b` while this branch intentionally pins `494f2644`; no K4/ID172 target test failed. This pre-existing legacy planner path is not used by calibration.
- Corrected ID74 real CPU load/export/restore plus one 100-simulation K4 score passed: direct shape `[1,8]`, root visits sum100, candidate shape `[1,100,4]`, snapshot fingerprint stable; CPU score took 10.828 seconds.
- Live server asset checks confirmed all three `*_train` assets contain 1200 tasks and exact SHA256 values recorded in the launcher. CLI import and full composed calibration config preflight passed with TP8, 24 workers, K4/100/c1, beta0, CoT0.7/top-p0.95.
- Every import/test used `PYTHONDONTWRITEBYTECODE=1`; all five server source layers remained clean afterward.
- ID172 end hook updated its adjacent metadata and server RL progress. The fixed render gate failed before any scientific measurement, so no conclusion about K4 scale or quality can be drawn.
- ID173 end hook updated run README/metadata and server RL progress. Because the positive-beta check ran before output persistence, exact spread/latency/turn records were lost despite full rollout; this is registered as`E0109`. The validation order proves all trajectories completed, MCTS spread exceeded threshold and median prior spread was0, but a new calibration must persist diagnostics before accepting/rejecting beta.
