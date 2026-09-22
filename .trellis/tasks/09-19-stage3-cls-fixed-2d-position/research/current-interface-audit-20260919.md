# Current interface audit: DINO CLS and Stage3 spatial position

## Source findings

- `FrozenDINOGridTargets._encode_images` reads the full DINO `last_hidden_state`, then selects
  only the last `patch_count` rows before spatial pooling. The current cache contains no CLS
  vector and cannot reconstruct it after the fact.
- `QueryAlignedModel` gathers exactly `grid_tokens` ordered Qwen query positions and requires
  its projector output shape to equal the DINO target. The selected Stage2 checkpoint therefore
  has no independent global state slot.
- `SharedSlotProjector` applies one shared MLP independently to all slots. It preserves the
  row-major axis but owns no position parameter.
- `TemporalSpatialGridPredictor` currently adds a learned absolute `spatial_position` per slot.
  The residual wrapper reuses that body and adds a zero-initialized delta to the input state.
- The recent Stage3 artifact records a residual K64/H1 predictor. The frozen reconstruction
  decoder is `spatial_grid_v1`, which consumes only a square K64 grid and already adds fixed
  coordinate channels at each UNet scale.

## Design consequence

A true additional global observation token is not a checkpoint-only flag. The experiment must
choose one of two interfaces:

1. Add a new Qwen query token. It can be appended to the existing 64 ordered query tokens,
   initialized from the existing query-token embedding mean, masked from LM labels, and aligned
   to real DINO CLS during Stage3. This changes prompt/tokenizer and trains one new selected
   embedding row, but preserves the existing 64 spatial slots and avoids a learned pooling proxy.
2. Pool the existing K64 state inside Stage3. This avoids changing the prompt but introduces a
   new pooling mechanism. Its output is a derived summary, not a direct Qwen global query, and
   the pooling architecture becomes a second experimental variable.

The first option is the recommended interpretation of “add CLS token” because the global slot
remains observation-conditioned through Qwen and is supervised by the actual DINO CLS. It does
not require retraining the whole Stage2 experiment: Stage3 can fresh-start from epoch16, extend
the tokenizer/model by one selected input token row, and train that row under the established
small Query LR. It does, however, change Qwen sequence length and must pass fixed-prefix/LM-format
regression checks.

The user selected the first option. The new token should be serialized after all 64 spatial
queries and before `action_start`. Qwen's causal mask then preserves every pre-existing spatial
query hidden state, lets the new global query read all spatial queries, and prevents the global
state from reading the not-yet-executed action. The later action logits do see the new token, so
format and rollout-success checks remain mandatory even if the old model parameters are frozen.

The representation should be established in a new Stage2 continuation from epoch16 before any
Stage3 WM training. Starting in Stage3 would couple a moving current-state representation with a
new dynamics model and make state-MSE targets move during the same experiment. It would not be a
clean test of global state or fixed spatial encoding.

The user classified this continuation and its downstream Stage3 as evaluation training only.
For this evaluation, only the newly added CLS token row is trainable in Stage2; the accepted K64
queries, Qwen backbone, and shared projector remain frozen. A future formal run must enable CLS
from Stage2 epoch1 and train it jointly under the formal Stage2 contract. The implementation must
therefore support fresh construction as well as the evaluation continuation, while checkpoint and
run metadata must prevent the evaluation-only artifact from being mistaken for a formal model.

## Fixed 2D encoding recommendation

Use a deterministic 2D sine-cosine table of shape `(1,64,1024)`, registered as a persistent
buffer. Give the global token a separate fixed type vector and keep it outside the 8x8 coordinate
table. Replace, rather than add to, the learned spatial position to keep the ablation interpretable.
The residual delta head remains zero initialized, so fixed position changes do not break exact-copy
initialization.

## Readout and reconstruction scope decision

The current `GridWorldModel.predict_action_values` and `ActionOutcomeHead` mean-pool the spatial
slot axis before their nonlinear readouts. This cannot distinguish grids with identical channel
means but different spatial layouts. Action-conditioned attention pooling is a stronger future
design, especially for per-action Q values, but the user chose not to make those heads the focus of
this evaluation.

ValueHead and OutcomeHead therefore remain trainable with the recent Stage3 objective and retain
their exact K64 mean-pooling input. The new global slot is explicitly excluded from their pool so
the additional token cannot silently change the old readout distribution. Their losses are health
signals rather than acceptance metrics, and this task does not claim the pooling defect is fixed.

The frozen `spatial_grid_v1` CFM also continues to receive exactly K64. It cannot directly measure
whether a decoder can use DINO CLS. Its useful comparison is narrower: whether the K65 WM uses the
global slot and fixed 2D positions to produce spatial future states that reconstruct better through
the same fixed decoder. Observed K64 from epoch16 and the CLS-aligned continuation should remain
identical because the appended causal Query is later in the sequence and all old representation
parameters are frozen.
