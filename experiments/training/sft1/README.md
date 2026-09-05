# Phase 1 — format SFT (SFT1)

Canonical location for SFT1 per `ai_tasks/sft1_exp.md`.

| File | Purpose |
|------|---------|
| `train.py` | Qwen2.5-VL SFT on Nimloth rollout records |
| `train_8gpu.slurm` | 8-GPU DDP train (`SFT1_TUNE_MODE=lora\|embedlr`) |
| `build_preprocess_cache.slurm` | CPU-only BF16 preprocess-cache build |
| `submit_cache_then_train_8gpu.sh` | Submit cache, then dependency-gated training |
| `convert_rollouts.py` | VAGEN rollout JSONL → Nimloth SFT records |
| `vagen_step60_data.py` | Pinned step60 source partition, overlap, conversion and complete-shard contracts |
| `vagen_step60_checkpoint.py` | Non-overwriting step60 shard audit, legacy FSDP merge plan and HF load validation |
| `extract_vagen_step60_evidence.py` | Non-overwriting W&B prompt/reward extractor that excludes assistant CoT and emits the hash-bound reconstruction fixture |
| `vagen_step60_runtime_contract.py` | Non-overwriting Git-computed reconstruction contract producer; prints the payload hash for approval |
| `hash_vagen_step60_runtime_contract.py` | Independent runtime-contract payload hash recomputation/check CLI |
| `vagen_step60_collect.py` | Evidence-backed reconstructed legacy service client, frozen-policy rollout, EOS/terminal audit, per-trajectory durable checkpoints, explicit interrupted-shard resume and reserved-directory/COMPLETE-last v3 shards; unavailable exact source commit remains provenance only |
| `vagen_step60_shard_state.py` | Fail-closed fresh/complete/matching-in-progress classifier and strict complete-shard gate for orchestration |
| `vagen_step60_gate_8gpu_preempt.slurm` | One-node preempt orchestrator: four environment GPUs plus four TP1 collectors, source-index-0 smoke gate, then four parallel 100-row shards |
| `vagen_step60_convert.py` | Complete batch1 shards → linked K16 SFT1/SFT2 views, v3 rejections and hash manifest; validates reserved-directory publication only after `conversion_manifest.json` appears last |
| `validate_vagen_step60_conversion.py` | Independent published-conversion hash/count/envelope validator |
| `derive_rollout_images_255.py` | Preserve sources and derive RGB 255×255 images with rewritten JSONLs |
| `derive_rollout_images_255.slurm` | CPU wrapper for the non-destructive image derivation |
| `merge_lora_ckpt.py` | LoRA adapter → `hf_merged` for VAGEN eval / SFT2 init |
| `rollouts_greedy_parallel.slurm` | Greedy rollout collection (Slurm array) |
| `eval_greedy_valtest.slurm` | Val/test rollout eval for a checkpoint |
| `env_external_4gpu.slurm` | Shared 4-GPU AI2-THOR env for rollouts/eval |
| `ckpt_eval_watcher.slurm` | Per-epoch eval during training |
| `summarize_eval_rollouts.py` | Aggregate eval JSONL success rates |
| `summarize_before_after_rollouts.py` | Before/after training comparison |
| `compare_eval_summaries.py` | Compare eval summary CSVs |
| `compare_rollout_resolution_probe.py` | Paired comparison for dumps with verified stable metadata; fails on visible runtime/metadata mismatch |
| `recover_rollout_resolution_pairs.py` | Diagnostic recovery for E0030-corrupted dumps via batch/runtime/instruction/initial-frame identity |
| `validate_rollout_train120_dump.py` | Exact 120-key, stable metadata/UID, runtime-config and RGB PNG completion gate |
| `submit_*.sh` | Thin sbatch wrappers (no hardcoded nodes by default) |

Config: `configs/training/sft1/qwen25vl_lora.yaml`; k=8 run manifest: `configs/training/sft1/qwen25vl_lora_k8.yaml`.

Latent query token count can be set with `LATENT_TOKEN_COUNT=<k>` in Slurm wrappers or `--latent-token-count <k>` in `train.py`. Select the protocol with YAML `latent.query_mode` or `--latent-query-mode inject|generate`: `inject` masks query-token CE labels and uses staged format evaluation, while `generate` supervises and freely generates the query-token block. `--[no-]mask-latent-query-labels` remains a deprecated compatibility alias; conflicting settings fail fast.

Library (planned): `src/nimloth/training/phase1_sft/`

## Paths

- **Scripts**: `experiments/training/sft1/`
- **Slurm logs**: `outputs/experiments/training/sft1/slurm/`
- **New train outputs**: `outputs/experiments/training/sft1/<date>/<name>/`
- **Legacy runs** (records, rollouts, eval): `experiments/navigation_baseline/runs/` — override via `SFT1_RUNS_ROOT`

Default init checkpoint: VAGEN `retry2` `global_step_79` actor HF export.

## Quick start

```bash
cd /project/peilab/atst/nimloth

# Recommended: build cache on CPU, then start LoRA training after cache succeeds.
# TRAIN_OUT, TRAIN_JSONL, VAL_JSONL, and INIT_HF must be exported.
SFT1_TUNE_MODE=lora bash experiments/training/sft1/submit_cache_then_train_8gpu.sh

# Rollout collection
ENV_NODE=dgx-13 bash experiments/training/sft1/submit_env_external_4gpu.sh
bash experiments/training/sft1/submit_rollouts_greedy.sh

# Per-epoch eval watcher
TRAIN_OUT=.../sft1_train_lora BASE_MODEL=.../global_step_79/actor/huggingface \
  bash experiments/training/sft1/submit_ckpt_eval_watcher.sh
```

For the fixed 120-task resolution probe, set `ROLLOUT_TRAIN120=1`; the dataset is exactly `base_train` seeds 1–60 plus `common_sense_train` seeds 1–60. `VAGEN_DIR` selects the old or corrected VAGEN worktree, and `EXPECTED_ROLLOUT_PNG_SIZE=512|255` makes the job fail if its persisted image path is wrong. The probe always uses greedy `temperature=0`, `top_p=1`, `top_k=-1`, `n=1`, 20 turns, one action per turn, and 512 response tokens per turn.

Validation dumps produced before the E0030 stable-identity fix may have trajectory metrics paired with the wrong `data_source/env_seed`. Direct paired comparison now fails on visible `config_id/eval_set` mismatch. `recover_rollout_resolution_pairs.py` is diagnostic-only: it can recover task pairs from control-batch membership, runtime config, instruction, and initial-frame similarity, but cannot restore exact seed labels.

SFT1 stores cached `pixel_values` as BF16 by default (`CACHE_PIXEL_DTYPE=bfloat16`), which matches the GPU visual encoder input dtype and halves their disk/read bandwidth versus FP32. The dependency-gated wrapper sets `REQUIRE_PREBUILT_CACHE=1`, so the GPU allocation never performs image preprocessing.

## Step60 interrupted-shard resume

A fresh collection command must omit `--resume`. It exclusively creates the stable direct-sibling `<output-dir>.inprogress` directory and fails if either that staging path or the final output exists. To resume the same interrupted shard, rerun the **identical command** with only `--resume` appended:

```bash
python experiments/training/sft1/vagen_step60_collect.py <all-approved-fresh-arguments>
python experiments/training/sft1/vagen_step60_collect.py <the-identical-approved-arguments> --resume
```

Resume validates the ordered source specs, runtime/policy identities, max-step and format contracts, every completed record hash, and every referenced image before environment health/identity requests or vLLM GPU-engine construction. CPU-only policy inspection binds package, model/tokenizer, EOS, sampling and engine configuration first; activation must match it exactly. It skips only validated completed rows. Unfinished rows restart from their original source spec with an attempt-unique environment-service ID and image namespace, while the persisted record/source identity remains stable; previous attempt and unreferenced image evidence is retained.

A crash-released exclusive sibling lock covers staging validation/creation, rollout, finalization and marker-last publication. Checkpoint files use atomic no-overwrite creation. A changed contract requires a fresh unique output identity rather than editing the in-progress directory.

The eight-GPU concurrency gate is intentionally not a self-contained launcher. `vagen_step60_gate_8gpu_preempt.slurm` requires the exact Nimloth/runtime/model/partition/contract/entrypoint paths, approved identity literals including `EXPECTED_NIMLOTH_COMMIT`, and explicit `RUN_MODE=fresh|resume`. Its Python helper verifies a clean exact Nimloth HEAD before run-root handling and again before service startup. Fresh mode atomically reserves an absent `RUN_ROOT`, writes the hashed immutable identity, creates its fixed layout, and fsyncs the identity file, directories, and parent; resume requires that exact identity and layout and rejects a durable `NON_RESUMABLE.json`. It requests one preempt node with eight GPUs, 224 CPUs, and 480G memory, binds the first four allocated GPUs only to the existing environment-service entrypoint in its own process group, and binds one of the remaining GPUs to each TP1 collector. The bash-invoked environment entrypoint must be a readable regular non-symlink file but need not have executable mode; `PYTHON_EXECUTABLE` and both Python entrypoints remain executable requirements. `PYTHON_ENV` is derived from the exact `PYTHON_EXECUTABLE` for the reused environment service. One batched helper process deeply classifies the smoke and all four gate outputs before environment startup, inspecting common actor/runtime/reconstruction/partition identity once. It atomically publishes and fsyncs a canonical hash-bound inspection handoff containing the validated common context and exact selected specs. The orchestrator captures its file SHA256 and verifies schema, CLI/input/item bindings before environment startup; every collector and later output validator requires that same hash rather than repeating common or partition inspection. GPU policy construction still verifies the live policy runtime contract against the handed-off contract after engine construction. The source-index-0 smoke uses `SMOKE_FORMAT_FAILURE_POLICY=fail_shard` and `SMOKE_COLLECTOR_CONCURRENCY=1`; after it completes and validates, four parallel 100-row collectors use `GATE_FORMAT_FAILURE_POLICY=exclude_trajectory` and the explicit production `GATE_COLLECTOR_CONCURRENCY`. Every scheduler attempt owns a new non-overwriting `RUN_ROOT/attempts/<job-and-attempt-id>` log/control/status tree and environment-service run directory. Ordinary nonzero orchestrator exits durably mark the run non-resumable without masking the original exit status; TERM/INT exit 143 and abrupt termination without that marker remain resumable. Cleanup targets recorded process groups even when their leaders have exited, while preserving logs, checkpoints, and attempt evidence.

## Legacy

SFT1 scripts in `experiments/navigation_baseline/` are frozen. Do not add new files there.
