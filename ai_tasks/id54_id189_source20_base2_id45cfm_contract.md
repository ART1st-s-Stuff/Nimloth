# ID54 — ID189 source20 Base seed2 ID45 CFM guided-successor page

Status: ready for production submission

## Purpose

Create an immutable derived page for one ID189 rollout. Every turn compares:

1. real current observation;
2. ID45 CFM reconstruction of behavior-time `current_state [16,1024]`;
3. ID45 CFM reconstruction of the MCTS depth-1 state for the actually executed guided action;
4. real next observation after that action.

Current/successor reconstructions use matched Gaussian noise, Euler50 and CFG2.

## Data and split

- Evaluation input: ID189 source20 `navigation_base_test_id187`, seed2. This is heldout evaluation data and is used only for frozen sampling, never training.
- ID45 CFM training source: pre-RL SFT1 epoch5 DINO-grid state cache, 59,389 train transitions.
- ID45 validation source: disjoint pre-RL validation cache, 6,054 transitions.
- ID45 condition is exact `16x1024`; there is no slot truncation, pooling or placeholder.
- No RL or post-RL data is used to train or update CFM. No training occurs in ID54.

## Checkpoints and modules

- Frozen CFM: `.../45_sft1e5_dinogrid16x1024_cfm_ep30_b32_drop015/train/best.pt`.
- Checkpoint step: 29000.
- Checkpoint SHA256: `5f029ba4cdf1077d49377100c43d9ac836d89386e0ac049c4b92e0b0a7744dfa`.
- CFM full-val correct/shuffled ratio: `1.084771351264476`.
- All modules are frozen. There is no optimizer, backward pass, training checkpoint or resume.

## Runtime and output

- Entry: `python -m nimloth.eval.id189_cfm_browser`.
- Runtime code commit: `76b0fab1d5d8f0c986522ef258051d1629eb7ad7` (adds required `peilab` Slurm account after the first pre-allocation submission rejection).
- Resource: normal 1xH800, 16 CPU, 96 GiB, 30 minutes.
- Output: `outputs/experiments/evaluation/reconstruction/2026-08-23/54_id189source20_base2_id45cfm_guidednext_euler50_cfg2`.
- Existing source Browser is read-only; output directory must be fresh and atomic at the derived-browser level.
- W&B project: `nimloth-recon`.
- W&B run name: `54_id189source20_base2_id45cfm_guidednext_euler50_cfg2`.
- W&B id: `nimloth-recon-id54-id189source20-base2-id45cfm-guidednext`.

## Monitoring and gates

- Verify exact source Browser manifest and CFM checkpoint hashes before GPU work.
- Verify one unique Base seed2 rollout, contiguous 20 turns and archive/image SHA256.
- For every turn, require exactly one depth-1 MCTS node matching executed action.
- Require current and successor states to be finite float32 `[16,1024]`.
- Require 20 rendered comparison strips, HTML, metadata and W&B table.
- Preserve `training_uses_rl_data=false`; produce no checkpoint.

## Decision history

The human initially approved a new pre-RL K16 CFM training run. Before launch, W&B and checkpoint inspection found the already-trained exact-shape ID45 checkpoint. The human then selected “先用现有ID45”; therefore the expensive retraining plan was not launched.
