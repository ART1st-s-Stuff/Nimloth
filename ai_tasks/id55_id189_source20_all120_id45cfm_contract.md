# ID55 — full ID189 source20 Base/Common120 ID45 CFM guided successors

Status: completed, validated, downloaded and merged

## Purpose

Generate CFM current-state and executed-action depth-1 successor reconstructions for all ID189 source20 heldout rollouts, then merge them into a derived copy of the prior 120-rollout interface. The canonical Browser remains immutable.

## Frozen data and model contract

- Evaluation source: exact ID189 source20 Base seeds1–60 plus Common Sense seeds1–60.
- Cardinality: 120 unique rollouts and 1,862 behavior-time turns.
- Inputs per turn: exact float32 finite `current_state [16,1024]` and the unique MCTS depth-1 `predicted_state [16,1024]` matching the actually executed guided action.
- Frozen decoder: ID45 best step29000, SHA256 `5f029ba4cdf1077d49377100c43d9ac836d89386e0ac049c4b92e0b0a7744dfa`.
- ID45 was trained before RL on SFT1 train states; it did not use RL or post-RL data.
- No optimizer, backward pass, checkpoint, resume or parameter update is allowed.

## Sampling and output

- Euler50, CFG2, chunk size4.
- A deterministic per-rollout noise seed is derived from base seed20260823 and immutable `rollout_sample_id`.
- Within every turn, current and successor use matched Gaussian noise.
- Server output: `outputs/experiments/evaluation/reconstruction/2026-08-23/55_id189source20_basecommon120_id45cfm_guidednext_all_euler50_cfg2`.
- The output must be fresh and atomic, with a global manifest, 120 per-rollout metadata files/pages and exactly 1,862 strips.
- A checksummed `view_payload.tar.gz` contains only derived reconstruction material for local transfer and merger.

## Runtime

- Resource: normal 1xH800, 16 CPU, 96 GiB, 30 minutes.
- W&B project: `nimloth-recon`.
- W&B run name: `55_id189source20_basecommon120_id45cfm_guidednext_all_euler50_cfg2`.
- W&B ID: `nimloth-recon-id55-id189source20-all120-id45cfm-guidednext`.
- Runtime commit: `1ae7dc32ae570a2b42ca8c20eaf69dee2888ba39`.

## Validation gates

- Exact canonical manifest SHA256 `6d555cd81141f280d3b7b1de5ad1972cea5456c13c2c0334ac4861dabb27de60`.
- Exact source coverage 60+60, 120 unique identities and 1,862 turns.
- Every source image/archive SHA256, state shape, finite value and unique executed depth-1 node passes.
- Every per-rollout metadata hash passes and every declared strip exists.
- No training checkpoint anywhere below output.
- The local merger must preserve the prior source manifest hash and expose all 120 rollouts in the old selector.

## Human decision

After reviewing the one-rollout canary integrated into the old interface and the 1–2 hour end-to-end estimate, the human approved full reconstruction with “可以”.

## Result

- Job `528324` completed on `normal/dgx-10` in `00:10:37`, exit `0:0`, peak RSS about 1.52 GiB.
- Produced and validated 120/120 per-rollout Browsers and exactly 1,862/1,862 comparison strips; global manifest SHA256 `84b9a8aafcb18b235f5f5b2b14744a9b4d9a79120a23f08111d44deed6131081`.
- View payload is 139 MiB, SHA256 `fb054f1552b440ecb195ae33ba01858f83135223cea121cce1efeceebc8677b7`.
- W&B ID55 is `finished`, history step0. No optimizer update or checkpoint exists.
- Resumable transfer took about 37 minutes over the current ProxyJump path; archive hash and all 1,862 local strips passed.
- Merged into the previous 120-rollout selector without changing the original Browser. Local entry: `nimloth-artifacts/id189_source20_base_common120_retry4_cfm_all_derived/extracted/index.html`.
- Derived merger manifest records 120 reconstructed rollouts, 1,862 turns and original manifest SHA256 `6d555cd8...de60`; all 120 selector entries and all 120 rollout pages carry reconstruction markers.
- Aggregate 8-bit strip diagnostics over 1,862 turns: current reconstruction→real current L1 `0.17478`; predicted successor reconstruction→real next `0.17488`; current reconstruction→real next `0.17545`; predicted/current reconstruction difference `0.05099`; predicted successor is closer than current reconstruction on `47.31%` of turns. Decoder/domain error remains large, so these images are qualitative aids rather than faithful action-effect ground truth.
