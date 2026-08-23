# ID56 — ID189 WM versus frozen ID45 CFM decoder diagnostic

Status: implementation/preflight

## Question

Determine whether the poor guided-successor images arise primarily from the K4 WM state prediction, the frozen ID45 CFM decoder/domain transfer, or both.

## Read-only design

For every nonterminal ID189 source20 transition:

- `current`: exact behavior-time `current_state [16,1024]`;
- `actual_next`: next turn's exact behavior-time `current_state [16,1024]`, which uses that observation's actual same-generation CoT;
- `predicted_next`: unique depth-1 WM state for the actually executed guided action;
- `wrong actions`: the other seven exact behavior-time depth-1 MCTS states;
- `real_next`: actual next environment observation.

The final turn of each rollout is excluded because the Browser has no following behavior-time current state to serve as this diagnostic's exact actual-next target. No terminal state or CoT is fabricated.

Cardinality is fixed at 120 rollouts, 1,862 turns and 1,742 nonterminal transitions: Base892 plus Common Sense850.

## Metrics

### Direct WM, no decoder

- predicted/actual-next RMSE and flattened cosine;
- current/actual-next copy baseline;
- predicted-over-copy relative error and better-than-copy fraction;
- executed-action rank among all eight depth-1 states, top1 fraction and margin to best wrong action;
- predicted/current state RMSE against the frozen DINO grid of the real next image, matching the WM's direct DINO supervision target.

### Frozen decoder oracle

With matched Gaussian noise per transition, compare against real next:

1. `D(current)` — copy baseline;
2. `D(actual_next)` — decoder oracle without WM;
3. `D(predicted_next)` — WM plus decoder.

Pixel L1 is averaged over four deterministic matched noise seeds. Frozen DINOv2-large cosine is measured for seed index0. One four-panel example per rollout is retained.

## Frozen identities

- Source Browser manifest SHA256: `6d555cd81141f280d3b7b1de5ad1972cea5456c13c2c0334ac4861dabb27de60`.
- ID45 CFM best step29000 SHA256: `5f029ba4cdf1077d49377100c43d9ac836d89386e0ac049c4b92e0b0a7744dfa`.
- ID45 was trained before RL on SFT1 data; no RL/post-RL data trains or updates it.
- DINOv2-large revision: `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`.
- All models are frozen. No optimizer, backward, checkpoint or resume.

## Runtime and output

- normal 1xH800, 16 CPU, 96 GiB, two-hour limit.
- Euler50, CFG2, CFM chunk4, four matched noise seeds.
- Output: `outputs/experiments/evaluation/reconstruction/2026-08-23/56_id189source20_wm_vs_id45cfm_oraclenext_all1742_s4_euler50_cfg2`.
- W&B project `nimloth-recon`, ID `nimloth-recon-id56-id189source20-wm-vs-id45cfm-oraclenext`.
- Runtime commit: pending implementation commit.

## Interpretation

- Poor decoder-oracle images with good direct WM metrics indicate decoder/domain transfer dominates.
- Good decoder-oracle images with poor direct WM metrics indicate WM dominates.
- Poor results in both paths indicate both components contribute.
- State action-rank near chance indicates insufficient action-specific WM dynamics even if aggregate reconstruction appears plausible.

## Human decision

The human explicitly approved execution after reviewing this separation design: “可以，你来执行”.
