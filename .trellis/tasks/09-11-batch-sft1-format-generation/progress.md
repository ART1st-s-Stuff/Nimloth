# Progress

## 2026-09-11 implementation and review

- Added formal `format_eval_batch_size`; standard Stage 1 value is 4.
- Batched the fixed 32 prompt generation while preserving order, FSDP synchronized calls, and strict sampled-token validation.
- Added per-batch progress plus separate validation and format-generation timing.
- Independent review fixed text-only multimodal input handling and the EOS trailing-content strictness bug.
- Verification: focused 10 passed; all SFT1 172 passed with 9 pre-existing warnings; compileall and diff check passed. Stage 2 had 152 passed, 1 skipped, and 3 sandbox-only Gloo address-resolution failures. Ruff and static type tools were unavailable.
- Remote old-code run remains active in epoch 4 validation. Do not interrupt before a complete `epoch_004/COMMITTED` or later boundary. After the code commit, refresh remote state and resume from the newest complete epoch for the real batch/FSDP timing gate.

## a100-1 resume contract

- Purpose: continue the already approved convergence run and measure whether batch size 4 preserves strict format generation while reducing epoch-end time.
- Source: committed branch `codex/fix-sft1-termination-retrain`; implementation commit starts at `1e640ae4`. The exact remote resume commit must be recorded after the experiment-task metadata commit.
- Entry: `/mnt/nimloth/venv/bin/python3 -m nimloth.training.sft.stage1.workflow` from a new clean remote worktree at that commit.
- Inputs: existing prepared train/heldout data, semantic-initialized base, and caches under `/mnt/nimloth/outputs/experiments/sft1-rollout2000/20260911T115012Z_success_only_termination_w8`; no data or cache rebuild.
- Resume/output: reuse the same run directory with `--resume`, starting only from its newest complete `epoch_N/COMMITTED`; optimizer, scheduler, per-rank RNG, data cursor, and convergence history must validate before launch.
- Model/objective: unchanged 8-rank FSDP LoRA Stage 1, BF16 model, action-number weight 8, answer CE, complete validation LM loss, and the approved two-epoch 1% convergence patience. The only runtime change is `format_eval_batch_size=4` plus timing/progress logs.
- Resources and duration: direct a100-1, all 8 GPUs, one resumed long run until convergence. This exceeds ten minutes and is covered by the user's approval of the reviewed plan. No W&B.
- Acceptance: first resumed epoch must finish without OOM/collective error, save 32 strict samples, expose 8 batch progress events, and record separate validation/format durations. Compare output reasons and timing with epoch 3/4 without treating this diagnostic as environment success rate.
- Monitoring: reuse the existing status heartbeat and controller/process checks. Stop on OOM, non-finite loss, collective failure, checkpoint validation failure, or output-contract mismatch.
