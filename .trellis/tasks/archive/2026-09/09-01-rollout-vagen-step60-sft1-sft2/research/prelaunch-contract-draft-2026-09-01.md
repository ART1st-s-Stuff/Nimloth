# Pre-launch contract draft — VAGEN step60 batch1

Date: 2026-09-01
Status: **exact candidate contract, not launch approval**; W-009 remote CPU preflight is complete. The docs-only evidence update still requires commit/push, then a separate exact `experiment_launch` approval before any merge, GPU, Slurm, service or rollout action.

## Purpose and validity boundary

Collect exactly batch1 (2,000 pinned source-train rows) with the frozen VAGEN step60 actor, then create linked K16 SFT1/SFT2 datasets. This is data collection, not training or model evaluation. Static success prevalence and the internal 200-row unseen-seed split cannot establish unseen-environment-distribution generalization.

Trainable modules/objectives: none. Actor, vision encoder, environment and all Nimloth modules are frozen/no-gradient. The terminal draft action is generated once but is never executed or supervised.

## Pinned inputs

- actor checkpoint: `/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z/global_step_60/actor`
- source train parquet: `/project/peilab/hligb/vagen-navigation/data/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z/train.parquet`
- parquet SHA256: `3c8161bd45adc4cde5d67157cf4db225753ed3925cb9a52e3a57d1dd11dbe9d6`
- unavailable source runtime commit (provenance only): `fee3ffac036a599b0ae979a6dd1ce2b21f7dec49`
- approved reconstruction base: `3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a`
- approved and pushed VAGEN patch ref: `origin/task/step60-runtime-reconstruction` at `170a673d1bf5855fc0ea6fbed0744b3d7168f8f0`; tree `58ef0eb66ad0bef7587c253c5c643af572c1d3a7`; canonical diff SHA256 `7f025476657de1289cf84b61d7702de26d248cd196412e9374a15e6de62730e9`
- approved and pushed post-W-012 Nimloth task ref: `origin/task/rollout-vagen-step60-sft1-sft2` at `187fe112038944a3ba7dd913fb4e87e15a33937e`
- remote-generated v3 runtime-contract payload SHA256: `cbb30382ffa5170daba37458f182d472e63b46c97f9fe588c6ce565214e6fcbf`; JSON file SHA256 `7b9184b8e33d76c0d410b141d4cff9ea993bef43708f5f9d16e7b2972718e9e8`
- model architecture/tokenizer lineage: `Qwen/Qwen2.5-VL-3B-Instruct`, source world size 8
- batch1 rule: category ordinals 0–999 from `base` and `common_sense`; internal held-out when `category_ordinal % 10 == 9`; expected 1,800 train + 200 held-out and zero bare-seed overlap

The source test parquet is excluded because all 128 `(eval_set, seed)` identities overlap source train.

## Frozen generation/environment contract

- source log field `actor_rollout_ref.rollout`: `do_sample=true`, temperature `0.7`, top-p `0.95`, top-k `-1`, `n=1`, `ignore_eos=false`; no separate `actor_rollout_ref.rollout.val_kwargs` exists in this run
- source W&B requirements evidence: vLLM `0.8.5.post1`, Transformers `4.49.0`, PyTorch `2.6.0`
- human-selected executable reconstruction: `/project/peilab/atst/nimloth/.venv/bin/python3`, vLLM `0.8.2`, Transformers `4.49.0`, PyTorch `2.6.0`; package drift is explicit and smoke-gated
- max response tokens `256`, max model length `6144`, max turns `20`
- history window `5`, at most six images per request
- strict source response: `<think>...</think><answer>one_action</answer>`
- archived prompt hashes:
  - system: `d691e077a5a4204386d3958a81d08f4322d6618dbee0f740b2c4848ddf2bc99a`
  - normalized initial user: `95d3469f8d076ab788b3d100407d0200541fcb33fe006af941f224f69a7757e2`
  - normalized post-step user: `c0d89b9a3949ef747676ba00d10b488a91b03fa80c2beb90d488d7de316824e7`
- pinned row config: `grounding_worldmodeling`, one action, format reward `0.02`, invalid-action penalty `-0.2`, success threshold `1.5`, no state reward
- source `step_length=0.5` metres: human-confirmed as the source VAGEN default, with no source-run override
- source `success_reward=10.0`: archived source W&B generation-table system prompt
- service timeout: 500 seconds

Reconstruction evidence status before launch approval:

- completed locally, pushed and remote-CPU rechecked: reviewed VAGEN patch `170a673...` on exact base `3003c2e...`, with clean worktree, ancestry/tree/diff/evidence hashes and golden prompt/parser/reward/API tests;
- completed locally, pushed and remote-CPU rechecked: Nimloth `187fe112...` implements v3 dual package provenance and rejects v1/v2, missing, single or overloaded package identities; affected remote suite is `108 passed`;
- exact legacy batch HTTP request/response contract under the patched mode;
- aligned finite `step_rewards` through the actual batch API, including golden `0.02`, `10.02`, `-0.2` and too-many-action `0.0` cases; any aggregate-reward fallback requires replanning;
- tokenizer EOS is the only generation stop: `ignore_eos=false`, empty custom stop strings/token IDs, and source-vLLM EOS tuple `(finish_reason="stop", stop_reason=null)` is required. Package/tokenizer/config hashes, EOS ID and generated token IDs are persisted. Length/custom/other finish or parser failure is audited but excludes the linked record from both SFT1 and SFT2; smoke uses `fail_shard`, formal collection uses `exclude_trajectory`.

## 2026-09-01 start-request preflight snapshot

Read-only rechecks after the human said to start established:

- checkpoint actor still contains exactly eight model and eight extra-state shards;
- pinned train parquet SHA256 still equals `3c8161bd45adc4cde5d67157cf4db225753ed3925cb9a52e3a57d1dd11dbe9d6`;
- stable output group `outputs/experiments/training/sft1-vagen-step60/` is still absent;
- approved Nimloth commit `696ee904e820636eb971e05ea09e43cffbe0b2a0` remains published at `origin/task/rollout-vagen-step60-sft1-sft2`, but the server canonical object store has not fetched it and no task worktree has been created;
- source commit `fee3ffac036a599b0ae979a6dd1ce2b21f7dec49` remains absent from both accessible VAGEN object stores and is not advertised by the configured VAGEN origin;
- the committed collector intentionally calls `source_runtime_commit()` and rejects any runtime whose clean Git HEAD is not exact `fee3ffac...`;
- current `normal` free-GPU snapshot has no healthy responsive node with four free GPUs: responsive mixed nodes expose only 1 or 2 each; nodes advertising eight free are `DOWN+NOT_RESPONDING` and are not launch candidates.

Therefore no merge, GPU allocation, Slurm submission or rollout was started. The human subsequently approved replanning to an evidence-backed reconstructed runtime. Launch remains blocked on fresh implementation approval, reviewed dual-repository commits, clean remote worktrees, complete literals and exact experiment launch approval.

## Staged commands

The exact first-stage merge + one-row smoke contract is now [`exact-merge-smoke-launch-contract-2026-09-02.md`](exact-merge-smoke-launch-contract-2026-09-02.md). It supersedes placeholders 1–4 below for that bounded launch request. The placeholder 100-row/formal batch/conversion commands below remain planning aids only and require new post-smoke launch approvals with fresh literal paths.

1. Partition (CPU, non-overwriting):

```bash
python3 experiments/training/sft1/vagen_step60_data.py \
  --source <PINNED_TRAIN_PARQUET> \
  --output <UNUSED_PARTITION_OUTPUT>
```

2. Actor audit/merge/load (CPU/RAM intensive; execution remains launch-gated):

```bash
python3 experiments/training/sft1/vagen_step60_checkpoint.py inspect-source \
  --actor-dir <PINNED_ACTOR_DIR> --hash-shards
python3 experiments/training/sft1/vagen_step60_checkpoint.py merge \
  --actor-dir <PINNED_ACTOR_DIR> \
  --target-dir <UNUSED_MERGED_HF_OUTPUT> \
  --python <APPROVED_REMOTE_PYTHON> \
  --merger-script external/VAGEN/verl/scripts/legacy_model_merger.py \
  --hash-shards --execute
```

3. Reconstruction runtime contract (CPU, clean approved VAGEN patch worktree; exact literals pending commit review):

```bash
python3 experiments/training/sft1/vagen_step60_runtime_contract.py \
  --runtime-root <CLEAN_APPROVED_RECONSTRUCTION_RUNTIME> \
  --expected-head <APPROVED_VAGEN_HEAD> \
  --expected-tree <APPROVED_VAGEN_TREE> \
  --expected-diff-sha256 <APPROVED_VAGEN_DIFF_SHA256> \
  --output <UNUSED_RUNTIME_CONTRACT_JSON>
python3 experiments/training/sft1/hash_vagen_step60_runtime_contract.py \
  --contract <UNUSED_RUNTIME_CONTRACT_JSON>
```

4. Collection entrypoint (GPU; first one-row smoke, then one 100-row concurrency shard, then only remaining batch1 shards after gates):

```bash
python3 experiments/training/sft1/vagen_step60_collect.py \
  --model-path <VERIFIED_MERGED_HF_OUTPUT> \
  --partition-manifest <PARTITION_MANIFEST> \
  --source-index <LITERAL_BATCH1_SMOKE_SOURCE_INDEX> \
  --shard-index 0 --shard-size 100 \
  --output-dir <UNUSED_UNIQUE_SHARD_OUTPUT> \
  --env-url <APPROVED_LEGACY_SERVICE_URL> \
  --run-id <UNIQUE_RUN_ID> \
  --source-runtime-root <CLEAN_APPROVED_RECONSTRUCTION_RUNTIME> \
  --source-runtime-contract <HASH_BOUND_RUNTIME_CONTRACT_JSON> \
  --expected-reconstruction-head <APPROVED_VAGEN_HEAD> \
  --expected-reconstruction-tree <APPROVED_VAGEN_TREE> \
  --expected-reconstruction-diff-sha256 <APPROVED_VAGEN_DIFF_SHA256> \
  --expected-runtime-contract-payload-sha256 <APPROVED_RUNTIME_CONTRACT_SHA256> \
  --format-failure-policy <APPROVED_LITERAL_POLICY> \
  --concurrency <APPROVED_LITERAL_CONCURRENCY> \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization <APPROVED_LITERAL_VALUE> \
  --engine-seed <APPROVED_LITERAL_ENGINE_SEED>
```

5. Conversion and independent validation (CPU, after all 2,000 identities are present in verified COMPLETE shards):

```bash
python3 experiments/training/sft1/vagen_step60_convert.py \
  --partition-manifest <PARTITION_MANIFEST> \
  <TWENTY_LITERAL_--shard-dir_ARGUMENTS> \
  --output-dir <UNUSED_UNIQUE_DATASET_OUTPUT> \
  --latent-token-count 16
python3 experiments/training/sft1/validate_vagen_step60_conversion.py \
  --output-dir <UNUSED_UNIQUE_DATASET_OUTPUT>
```

## Output, resume and monitoring

Stable group: `outputs/experiments/training/sft1-vagen-step60/`. Merge, smoke, concurrency gate, formal batch1 and conversion each require a different unused run directory. W&B is not required; if operational metrics are enabled later, its project/name/ID must be added before launch approval.

Shard resume accepts only a hash-valid `COMPLETE` marker whose raw records, source identities, image bytes, runtime/policy contracts, reward semantics and counts all match. Partial/failed directories remain evidence and are never consumed. Conversion has no partial resume; retry uses a fresh unique directory.

Monitor scheduler state, process/log errors, service health, GPU utilization, response-format failures, image creation, terminal non-step evidence and COMPLETE publication. Cancellation may target only the exact task-owned job ID recorded after submission and never deletes partial output.

## Resource direction and pending literals

Human-selected direction: `normal`, one node, four GPUs total: policy TP2 on two GPUs plus two environment GPUs. W-009 must re-query availability immediately before submission and fill exact CPUs, memory, walltime, device binding, hold-allocation/srun commands, job identity, expected duration and cancellation command.

## Current blockers

1. The 2026-09-02 remote preflight and exact merge/smoke contract task records need their normal docs-only commit and exact-ref push approvals.
2. Checkpoint merge/load, one-node four-GPU allocation, reconstructed service, AI2-THOR and the one-row smoke require a separate exact `experiment_launch` approval; none has run.
3. `normal` availability is transient. The latest snapshot has healthy nodes with four or more free GPUs, but availability and run-root nonexistence must be rechecked immediately before submission.
4. The 100-row concurrency gate and remaining batch1 are intentionally not included in the first launch request. Their literal outputs/resources/commands require actual smoke evidence and fresh approvals.
