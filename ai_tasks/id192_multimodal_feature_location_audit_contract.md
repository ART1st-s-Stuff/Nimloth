# ID192 — Frozen multimodal feature-location audit

## Authorization and question

The human approved a frozen feature-location audit before any new state encoder
or WM training. ID192 asks where current-image collision evidence and exact
instruction-goal semantics disappear inside ID176.

No output from this audit is a deployable checkpoint and it does not authorize
state fusion, backbone/projector training, WM, ValueHead, MCTS or RL.

## Same-forward features

For each exact ID60 early state, make the same frozen Qwen forward used for its
K16 hidden and capture:

- `vision_pre_llm`: the current/last image output of the Qwen visual transformer
  and merger before the language model, pooled row-major to 4x4;
- `fused_image_final`: final-norm LLM hidden at the current/last image-token
  positions from that same forward, pooled to 4x4;
- `instruction_embedding`: mean input embedding over the exact archived
  instruction token span;
- `instruction_final`: mean final-norm hidden over the same causal span;
- final K16 hidden, which must numerically reproduce the immutable ID191 cache.

Prefixes use the observation's actual archived CoT. Fixed, canonical or
placeholder CoT and a second Transformer replay are forbidden. Historical
images may remain in the prefix, but spatial features always select the last
image corresponding to the current observation.

## Data and probes

- Exact pre-RL ID52 train/validation records and immutable ID60/ID191 caches.
- Preserve ID60 exact initial/current/next-image decontamination and grouped
  inner selection. Row-level task identity remains unavailable.
- Train only fresh diagnostic linear readouts.
- For `vision_pre_llm`, `fused_image_final`, K16, SFT1 state and DINO, report
  action-specific outcome ROC-AUC, PR-AUC, Brier/ECE and paired visual-source
  minus DINO bootstrap intervals.
- For instruction, spatial, K16, state and DINO features, report the existing
  exact-instruction goal probe and paired goal intervals.
- Every saved feature is float32, finite, exact-shape checked and SHA256 bound.

## Direction decision

A Qwen visual source supports direct unified fusion only if both lateral-action
AUC lower intervals exceed chance and its paired AUC is not more than 0.02 below
DINO. An instruction source must pass the existing ID60 micro/macro, majority
and paired-DINO goal gate.

- Both source types pass: propose a separately authorized unified same-forward
  fusion canary.
- No visual source passes: prefer visual-encoder repair or DINO-teacher
  distillation rather than another hidden-only projector.
- Visual passes but instruction fails: redesign the instruction/goal encoder.

## Preflight evidence

Job `529749` completed `0:0` in 33 seconds on one H800. A two-state actual-prefix smoke captured finite float32 `k16_hidden`, `vision_pre_llm`, and `fused_image_final` arrays of shape `[2,16,2048]`, plus `instruction_embedding` and `instruction_final` arrays of shape `[2,2048]`. No optimizer or W&B run was created.

## Attempt0 and corrected runtime

- Attempt0 Job `529767` failed `1:0` after `00:02:14`: isolated instruction tokenization did not preserve a Qwen BPE punctuation/newline boundary. No feature cache, probe, optimizer update or result was produced; W&B finished failed. It is not resumable.
- Retry1 fixed token boundaries and completed all 2,229 forwards, but retained thousands of Torch-backed NumPy chunks for post-loop concatenation. Job `529788` was manually cancelled at `00:41:26` before cache creation; peak RSS was about 23 GiB. No result/readout/checkpoint/optimizer update exists and its interrupted W&B run must not be resumed.
- Retry2 tokenizes the complete archived instruction field with offsets, preallocates each final float32 feature array, copies every selected batch directly by source-state index, and logs forward/model-release/cache/probe stages.
- Entrypoint: `nimloth.eval.multimodal_feature_location_audit`.
- One H800 on `normal`, 45-minute limit, 128 GiB host memory.
- Corrected fresh output:
  `outputs/experiments/evaluation/state_alignment/2026-08-24/192_frozen_multimodal_feature_location_audit_retry2`.
- W&B project `nimloth-recon`, retry2 run ID
  `nimloth-recon-id192-feature-location-audit-retry2`, resume forbidden.
- A failed attempt cannot overwrite or resume; it requires fresh output and W&B identity.

## Actual retry2 result

Job `529879` completed `0:0` in `00:25:13` on `normal/dgx-27`; W&B finished. Same-forward K16 identity RMSE/max is `0/0`.

### Goal location

Goal micro/macro top1:

- exact instruction input embedding: `0.99711/0.99000`;
- exact instruction final hidden: `0.99133/0.98266`;
- DINO: `0.05780/0.04042`;
- K16 hidden: `0.05491/0.03575`;
- SFT1 state: `0.06069/0.04261`.

Both instruction sources pass every goal gate; paired-minus-DINO lower CIs exceed `+0.90`. Goal semantics are available explicitly in the prompt and are lost by the K16/state interface.

### Visual/outcome location

Outcome AUC pre-LLM vision / fused-image final / K16 / DINO:

- forward `0.83497/0.86799/0.86537/0.87294`;
- right `0.52535/0.73952/0.76432/0.71889`;
- left `0.71831/0.73119/0.59910/0.73831`.

`fused_image_final` has the strongest consistent same-forward visual point estimates and is near DINO for all movements; raw pre-LLM vision is near chance on right. The strict 0.02 non-inferiority gate remains false because external right/left N=`142/193` gives wide fused-minus-DINO CIs `[-0.08063,+0.12466]` and `[-0.08142,+0.06734]`. This is unresolved uncertainty, not evidence of a large point deficit.

### Decision

- The exact instruction embedding is the preferred goal source.
- `fused_image_final` is the preferred visual candidate by point estimate, but direct fusion is not formally authorized until larger exact-image-grouped out-of-fold diagnostics tighten its non-inferiority evidence while retaining archive-external results as primary heldout evidence.
- Do not jump directly to a second deployed DINO encoder solely because the conservative gate is false. If grouped confirmation fails, then evaluate DINO-teacher distillation or visual-encoder repair.
- Feature cache SHA256=`a46278d43fafdd964708af988123d8898c3990a5fb3a3d9f97d935c5645f813c`; result SHA256=`4592b68d6d039423b439bb9af0343818a615c7a82be1f9b55130b0ec58e7717a`.
