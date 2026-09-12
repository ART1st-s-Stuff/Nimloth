# Stage 1 epoch 5 zero-success diagnosis

## Run identity and terminal state

- Output: `/mnt/nimloth/outputs/experiments/sft-stage-evaluation/20260911T091220Z_stage1_epoch5_test120_gpu7`
- Source: `e15faf31c9bc5abdb75d9c68efcca399bf0165a6`
- Checkpoint: `/mnt/nimloth/outputs/experiments/sft2-rollout2000/20260911T072059Z_epoch5_query_converge/base`
- Requested scope: Base 60 plus Common Sense 60 test episodes.
- Stopped with human authorization after 12/120 Base episodes, 0 successes. Evaluation PID 606504 and environment PID 605810 were verified by exact command and process group before SIGTERM. Both exited and GPU 7 was released. Existing output was preserved.

## Observed mechanism

- Across the 220 initially aggregated steps, every raw response contained `<|action_end|>`, every response reached `max_tokens=512` with finish reason `length`, and every strict parse returned `invalid_response_envelope`.
- A representative token suffix was `action_end, action_start, newline, endoftext, im_start, ...`; actual EOS `<|im_end|>` did not occur.
- The evaluator therefore sent the Stage 1 invalid-output no-op. Every observed step returned -0.2, episodes reached `max_steps=20`, and environment success remained false.
- The persisted training cache was separately inspected and its labels end with `<|action_end|><|im_end|>`. The defect is generation behavior, not missing EOS labels.

## Source mismatch exposed by the run

- `stage1.loss.resolve_action_token_ids()` includes action start, action end, and all eight action-number tokens; standard weight is 8 while EOS remains 1.
- `stage1.trainer.nimloth_format_correct()` uses regex `search`, so a correct-looking substring passes even when the model continues after `action_end`.
- `EarlyProtocol.parse()` uses full-response matching and intentionally returns an empty service response for invalid Stage 1 output. The production evaluator therefore exposes a failure that the training metric missed.

The weighting imbalance is the strongest current causal hypothesis because the repeated generation begins immediately after `action_end`, but the run alone does not prove it. The new training is an ablation with the prompt, initialization, data and optimization settings held fixed.
