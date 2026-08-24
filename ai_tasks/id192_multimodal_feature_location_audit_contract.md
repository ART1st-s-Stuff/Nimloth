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

## Planned runtime

- Entrypoint: `nimloth.eval.multimodal_feature_location_audit`.
- One H800 on `normal`, 45-minute limit, 128 GiB host memory.
- Fresh output:
  `outputs/experiments/evaluation/state_alignment/2026-08-24/192_frozen_multimodal_feature_location_audit`.
- W&B project `nimloth-recon`, run ID
  `nimloth-recon-id192-feature-location-audit`, resume forbidden.
- A failed attempt cannot overwrite or resume; it requires fresh output and W&B identity.
