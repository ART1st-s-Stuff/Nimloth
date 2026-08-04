# RL decoded stop and state-token budget

## Goal

- Match decoded literal `</think>` before injecting latent query tokens in the
  single-request vLLM turn state machine.
- Bound the complete processor-expanded Qwen state prefix before both rollout
  action execution and training recomputation.

## Evidence and decision

- ID122 iteration 11 completed with a largest measured training state of 6,765
  tokens.
- The failed iteration 12 episode `rl_common_sense_train_000091` measured
  16,677 tokens at step 14, 18,134 at step 15, and 23,227 at step 19. The first
  observed OOM region began with the 18,134-token late prefixes.
- The formal 16-rollout configs now use `actor.max_state_tokens=16384`. This is
  1,750 tokens (9.7%) below the first observed OOM state, while clean 20-step
  episodes were about 4,100 tokens and the longest completed ID122 state was
  6,765. The cap is an empirical safety boundary for the current two-GPU/rank
  Qwen topology; it cannot rule out unrelated OOM causes.

## Implementation

- `TurnResponseLogitsProcessor` loads the artifact tokenizer and decodes the
  generated continuation before each next-token sample. The first literal
  `</think>` match records the token boundary and immediately forces the latent
  query block, independent of BPE segmentation.
- Rollout measures `len(prompt_token_ids) + tokens through action_start`. If
  that complete state exceeds the cap, `EpisodeRunner` truncates before the
  action; terminal-state generation remains available for the preceding real
  transition target.
- `RLModelRuntime` repeats the cap check on processor-built `input_ids` before
  Qwen forward, so stale or externally supplied over-budget trajectories fail
  closed rather than reaching CUDA.
- All formal H=1 two-GPU/rank configs, including the 12/20/22/24-GPU
  16-rollout variants, and both rollout launch paths propagate the same cap.

## Validation status

- Python compile of 16 changed source/test files, `bash -n` for both launch
  scripts, and `git diff --check` pass.
- Local focused pytest was unavailable because neither local Python environment
  contains pytest. The first remote focused run collected after initializing
  the pinned LeWM submodule and found two test-fixture/config-consistency
  failures; both are fixed and the focused rerun is pending.
- No GPU experiment, Slurm submission, checkpoint update, or RL restart has
  been performed in this task.
