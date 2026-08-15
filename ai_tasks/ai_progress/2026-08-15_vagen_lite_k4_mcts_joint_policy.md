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
- First implementation slice committed and pushed:
  - Parent `6ba45207910e3e7a46af00b5fd2c472a33d194fd`.
  - VAGEN `3efb0368aeb43f48301c232c1220664ddfe5ff52`.
  - VERL remains the exact `494f264494b2525f2c13595f63ac4912963e6d2f` gitlink.
- Added a full immutable projector/predictor/ValueHead K4 planning snapshot, exact file transport, direct-Q/MCTS scoring separation, K4 behavior/draw/execution/ledger schemas, TP-rank-zero worker install/score methods, a CPU lifecycle owner, and the optimizer-free balanced calibration entrypoint.
- Legacy direct-Q schemas and ID171 paths remain separate.
- Parent worktree is clean except this progress update and the known nested `external/le-wm` untracked marker; VAGEN worktree is clean.
- No calibration, Slurm, Ray, model load, environment rollout, optimizer, or GPU work has started.
- Remote CPU validation is currently blocked: two SSH attempts reached the proxy then closed with `Connection closed by UNKNOWN port 65535`; this was not a timeout and no remote command ran.

## Validation log

- Local AST parsing passed for every new/modified Python file.
- `git diff --check` passed before both commits.
- Dependency-light K4 config/draw and K4 decision-ledger manual checks passed under `PYTHONDONTWRITEBYTECODE=1`.
- Full PyTorch/pytest validation is pending because the project worktree has no local runtime with its dependencies and superpod SSH is currently unavailable.
