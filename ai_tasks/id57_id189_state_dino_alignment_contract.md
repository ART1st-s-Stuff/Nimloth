# ID57 — ID189 actual/predicted state versus DINO alignment

Status: completed, validated and downloaded.

## Question

Test the clarified state semantics: state should retain the visual and goal-related planning subspace, DINO should pull its visual component toward a visual space, and WM should predict only that constrained state. Determine whether the observed mismatch is primarily in the actual behavior-time projector state, the WM prediction, slot/normalization alignment, or their interface.

## Read-only source and split

- Immutable source: ID189 source20 canonical Browser manifest SHA256 `6d555cd81141f280d3b7b1de5ad1972cea5456c13c2c0334ac4861dabb27de60`.
- Split: the Browser records `split=validation` and contains heldout `navigation_base_test_id187` seeds1--60 plus `navigation_common_sense_test_id187` seeds1--60.
- Cardinality: 120 rollouts, 1,862 unique behavior-time turns and 1,742 exact nonterminal transitions.
- Each actual next state is the next turn's archived behavior-time `current_state [16,1024]`, generated with that observation's actual same-generation CoT.
- The final turn of each rollout is excluded from transition comparison because no exact following K16 behavior-time state exists. The diagnostic does not fabricate, replay or fill a terminal state or CoT.

## Comparisons

For every unique turn:

1. actual current state versus frozen same-image DINO grid;
2. raw scale, RMS and slot-diversity statistics;
3. raw, cosine, token-centered and token-standardized alignment;
4. a global fixed 16-slot minimum-cost permutation diagnostic;
5. canonical DINO from the original archived observation and a separately labeled legacy-ID56 comparison that first bicubic-resizes to 128×128.

For every exact nonterminal transition:

1. current/copy state versus next-image DINO;
2. actual next state versus next-image DINO;
3. executed-action WM predicted next state versus next-image DINO;
4. WM prediction and copy versus exact actual next behavior-state;
5. overall, per-source and per-action copy-relative skills.

The fixed slot permutation is fitted and reported only on this same frozen evaluation set. It is not saved as a model, adapter or deployment recommendation.

Goal-specific probing is explicitly unavailable in this run: the archived Browser has free-form task text but no validated goal labels or controlled same-observation/different-goal pairs. No heuristic labels will be fabricated.

## Frozen identity and mutation boundary

- Frozen DINOv2-large revision: `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`.
- Canonical comparison passes the original archived observation directly to the frozen DINO processor, matching the WM training/runtime teacher path. ID56 instead passed the decoder's 128×128-resized real image into DINO; ID57 retains that path only as an explicitly labeled sensitivity comparison.
- Browser states and images are read only.
- No policy, projector, WM, ValueHead, decoder or environment replay is loaded.
- No optimizer, backward, parameter update, checkpoint, rollout or environment action is allowed.
- ID189/source20 data is used only as heldout evaluation and never as training data.

## Entry, output and resume

- Entry: `python -m nimloth.eval.id189_state_dino_alignment`.
- Runner: `experiments/training/reconstruction/run_id57_id189_state_dino_alignment.sh`.
- Proposed Slurm: `normal`, one H800, 16 CPU, 48 GiB, 45-minute hard limit.
- Expected elapsed time: approximately 5--15 minutes; DINO inference over 1,862 archived images dominates.
- Output: `outputs/experiments/evaluation/state_alignment/2026-08-23/57_id189source20_state_dino_alignment_all1742`.
- Output is atomic and must not exist before launch.
- Resume is disabled. A failed attempt must use a fresh output and W&B identity.
- W&B project: `nimloth-recon`.
- Run name: `57_id189source20_state_dino_alignment_all1742`.
- W&B ID: `nimloth-recon-id57-id189source20-state-dino-alignment`.
- The launcher requires exact `EXPECTED_COMMIT` equality and writes that runtime commit into the output README before final validation.

## Monitored outputs

- 120-rollout progress and exact turn/transition counts;
- actual-state/DINO raw and normalized metrics;
- predicted/copy/actual-next DINO errors;
- behavior-state and DINO copy-relative skills;
- state/DINO distribution statistics;
- fixed slot permutation and identity-cost reduction;
- all metrics finite, source/seed cardinality exact, no checkpoint files.

## Human authorization

The human authorized continuing the read-only comparison: “你可以继续进行只读比较”, then explicitly selected `Normal 1×H800（推荐）` after being shown the 5--15 minute estimate and frozen/no-checkpoint boundary. This authorizes no training or model modification.

## Result

- Job `528490` completed on `normal/dgx-37` in `00:03:10`, exit `0:0`, peak batch RSS `2,398,264 KiB`.
- Runtime commit: `a8b34accd8ddad533143ef1c3667a79bf95a8b0a`.
- All 120 rollouts, 1,862 unique turns and 1,742 exact nonterminal transitions validated. W&B finished with one step0 row. No model replay, optimizer, update or checkpoint occurred.
- Actual same-image state has DINO RMSE `0.97769`, flattened cosine `0.36629` and token-centered cosine `0.35176`: actual projector states are weakly aligned to the intended visual teacher.
- Against canonical original-observation next DINO, copy RMSE/cosine is `0.98095/0.36005`, actual-next is `0.97866/0.36391`, and predicted-next is `0.83053/0.62777`.
- Prediction has copy-relative canonical-DINO skill `+0.28099` and actual-next-relative skill `+0.27750`; it beats copy on `100%` and actual-next on `99.885%` of transitions.
- Behavior-state prediction remains poor: RMSE `0.52598` versus copy `0.11923`, skill `-9.44815`, with `0%` better-than-copy.
- Mean state standard deviation is actual-next `0.47804`, prediction `0.82286`, DINO `1.04802`; slot-deviation RMS is `0.25585/0.46168/0.76248`. WM output lies between the behavior-state and DINO distributions.
- A best global fixed slot permutation reduces identity assignment cost by only `0.169%`; slot ordering is not the main issue.
- The corrected original-observation result is close to legacy128 (`0.83053` versus `0.83801`; skill `+0.28099` versus `+0.27888`). Thus the prior visual-signal direction survives E0144 correction, but ID57 is the canonical metric.
- Base and Common Sense both show approximately `+0.28` DINO skill. Every recorded action class is positive, with low confidence for rare actions.
- Goal retention remains unmeasured because no validated labels or matched goal-counterfactual observations exist in the archive.

Conclusion: actual projected state and WM predicted state implement incompatible interfaces. Actual state is weakly DINO-aligned, while WM is strongly pulled toward DINO and leaves the behavior-state manifold consumed by recursive WM/ValueHead. The next optimization should anchor actual current/next projector states with symmetric visual/goal supervision and a frozen or EMA canonical target before changing search depth.

- Server output: `outputs/experiments/evaluation/state_alignment/2026-08-23/57_id189source20_state_dino_alignment_all1742`.
- Summary SHA256: `fa6e8fa90ea6268ecfbb14ee360a5fd561245f58af7d6561a51738348fc32ea7`.
- Payload SHA256: `e59d89c11da786d99e07a0945c16b4b32f8740b2c3066f8d0f4194151b79c232`.
- Local report: `/workspace/remote2/nimloth-artifacts/id57_id189_state_dino_alignment/extracted/index.html`.
