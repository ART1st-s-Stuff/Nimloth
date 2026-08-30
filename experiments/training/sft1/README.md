# Phase 1 — format SFT (SFT1)

Canonical location for SFT1 per `ai_tasks/sft1_exp.md`.

| File | Purpose |
|------|---------|
| `train.py` | Historical Qwen2.5-VL format SFT on Nimloth rollout records |
| `state_interface_v2_canary.py` | Historical non-launching SFT1-v2 config/manifest/source preflight |
| `state_interface_v2_identity_audit.py` | CPU-only complete ID176 processor/token/template/K16-action identity audit |
| `state_interface_v2_resolve_config.py` | Publish one immutable launch-locked config from explicit approved values |
| `state_interface_v2_controller.py` | Strict phase preflight/transaction for cache→smoke→resume-smoke→formal→report |
| `state_interface_v2_cache.py` | Launch-locked distributed fresh ID176/DINO target generation or CPU cache inspection |
| `state_interface_v2_train.py` | Launch-locked production FSDP smoke, exact resume-smoke, or formal three-epoch runtime |
| `state_interface_v2_canary.slurm` | Resource-unspecified sequential wrapper; executes only the separately approved resolved phase command |
| `query_state_smoke.py` | Non-submitting `torchrun` child for one separately approved Query-State fresh/resume phase |
| `query_state_train.py` | CPU-only strict pilot/formal resolved-contract preflight; it does not launch CUDA/Slurm |
| `query_state_export.py` | CPU-only human-gate/terminal-primary export preflight; actual full-state materialization remains a separate all-rank library call |
| `train_8gpu.slurm` | 8-GPU DDP train (`SFT1_TUNE_MODE=lora\|embedlr`) |
| `build_preprocess_cache.slurm` | CPU-only BF16 preprocess-cache build |
| `submit_cache_then_train_8gpu.sh` | Submit cache, then dependency-gated training |
| `convert_rollouts.py` | VAGEN rollout JSONL → Nimloth SFT records |
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

Reusable state-interface-v2 code lives in `src/nimloth/training/sft1/`.
The unchanged code/interface fixture is
`configs/training/sft1/state_interface_v2_code_canary.yaml`. The separate
experiment template `state_interface_v2_early4_report_first.yaml` records the
approved early-4 data and optimizer/objective values, but deliberately retains
`launch_locked=false`,
`LOCK_BEFORE_LAUNCH` outputs, and zero processor/token/prompt digests. It cannot
pass launch preflight until the later identity/topology/output approval updates
all of those fields in a committed clean worktree. The historical manifest
preflight additionally requires a separately generated identity-bound manifest:

```bash
python experiments/training/sft1/state_interface_v2_canary.py \
  --config configs/training/sft1/state_interface_v2_code_canary.yaml \
  --manifest /path/from/a/later/approved/experiment/manifest.json
```

The Query-State path has a separate unresolved preparation file at
`configs/training/sft1/query_state_smoke_prep.yaml`. It cannot execute CPU
preflight or CUDA and is not a launch contract. A future externally immutable
`preflight_locked=true, launch_locked=false` artifact must bind exactly
`2 * world_size` real train rows (one per rank for fresh and resume), source and
ID176/DINO identities, two optimizer groups, FULL_SHARD topology, unique output
and resources; it enables only the read-only `preflight` phase. A subsequent
approval-bound artifact additionally binds the two-line canonical child-command
manifest and may run CUDA. Both stay outside the clean exact-commit worktree
rather than claiming to contain their own Git commit. The child performs no
`sbatch`; fresh and resume remain separate `torchrun` processes and W&B is disabled.
Every invocation also fails before project imports unless bytecode is disabled,
pycache is outside the worktree, `HF_HOME` is absolute/unambiguous, and the
launch `PYTHONHASHSEED` equals the resolved config seed.

The pilot/formal and export entrypoints added here are intentionally thin,
read-only preflights because this worktree contains no launch-locked config and
has no GPU/export approval. The production libraries own segment transactions,
validation, strict W&B state, official FSDP greedy probing and gated full-state
materialization; a later approved exact torchrun command must compose those
owners rather than turning these preflights into an implicit launcher.

A passing local test or preflight proves schema/source compatibility only. The
legacy Slurm file intentionally omits partition/GPU/CPU/memory/time directives and
cannot execute while the config is unresolved/unlocked. With a separately
approved resolved config it runs exactly one controller phase; formal remains
blocked until cache, smoke, and resume-smoke markers all pass. Cache creation,
GPU/FSDP execution, W&B identity selection, model-quality interpretation, and
every SFT2/WM/ValueHead action remain separately gated.

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

## Legacy

SFT1 scripts in `experiments/navigation_baseline/` are frozen. Do not add new files there.
