# Pre-launch contract draft — VAGEN step60 batch1

Date: 2026-09-01
Status: **not launch approval**; W-007 must replace every pending field with read-only preflight evidence and obtain a separate `experiment_launch` approval.

## Purpose and validity boundary

Collect exactly batch1 (2,000 pinned source-train rows) with the frozen VAGEN step60 actor, then create linked K16 SFT1/SFT2 datasets. This is data collection, not training or model evaluation. Static success prevalence and the internal 200-row unseen-seed split cannot establish unseen-environment-distribution generalization.

Trainable modules/objectives: none. Actor, vision encoder, environment and all Nimloth modules are frozen/no-gradient. The terminal draft action is generated once but is never executed or supervised.

## Pinned inputs

- actor checkpoint: `/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z/global_step_60/actor`
- source train parquet: `/project/peilab/hligb/vagen-navigation/data/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z/train.parquet`
- parquet SHA256: `3c8161bd45adc4cde5d67157cf4db225753ed3925cb9a52e3a57d1dd11dbe9d6`
- source runtime commit: `fee3ffac036a599b0ae979a6dd1ce2b21f7dec49`
- model architecture/tokenizer lineage: `Qwen/Qwen2.5-VL-3B-Instruct`, source world size 8
- batch1 rule: category ordinals 0–999 from `base` and `common_sense`; internal held-out when `category_ordinal % 10 == 9`; expected 1,800 train + 200 held-out and zero bare-seed overlap

The source test parquet is excluded because all 128 `(eval_set, seed)` identities overlap source train.

## Frozen generation/environment contract

- `do_sample=true`, temperature `0.7`, top-p `0.95`, top-k `-1`, `n=1`
- max response tokens `256`, max model length `6144`, max turns `20`
- history window `5`, at most six images per request
- strict source response: `<think>...</think><answer>one_action</answer>`
- archived prompt hashes:
  - system: `d691e077a5a4204386d3958a81d08f4322d6618dbee0f740b2c4848ddf2bc99a`
  - normalized initial user: `95d3469f8d076ab788b3d100407d0200541fcb33fe006af941f224f69a7757e2`
  - normalized post-step user: `c0d89b9a3949ef747676ba00d10b488a91b03fa80c2beb90d488d7de316824e7`
- pinned row config: `grounding_worldmodeling`, one action, format reward `0.02`, invalid-action penalty `-0.2`, success threshold `1.5`, no state reward
- service timeout: 500 seconds

Pending exact-source evidence before approval:

- readable clean runtime at exact commit `fee3ffac...`;
- resolved source step length and success reward;
- exact legacy batch HTTP request/response contract;
- reward provenance: either aligned finite `step_rewards`, or `trajectory_terminal_reward` with one explicit terminal `info` key. No conversion flag may relabel this decision;
- terminal stop/finish boundary and format-failure behavior confirmed by one-row smoke.

## Staged commands

The exact approved commit, clean remote worktree and unique output identities are intentionally pending until commit approval. W-007 must render these commands verbatim with literal paths before launch approval.

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

3. Collection entrypoint (GPU; first one-row smoke, then one 100-row concurrency shard, then only remaining batch1 shards after gates):

```bash
python3 experiments/training/sft1/vagen_step60_collect.py \
  --model-path <VERIFIED_MERGED_HF_OUTPUT> \
  --partition-manifest <PARTITION_MANIFEST> \
  --shard-index <LITERAL_INDEX> --shard-size 100 \
  --output-dir <UNUSED_UNIQUE_SHARD_OUTPUT> \
  --env-url <APPROVED_LEGACY_SERVICE_URL> \
  --run-id <UNIQUE_RUN_ID> \
  --source-runtime-root <CLEAN_EXACT_SOURCE_RUNTIME> \
  --source-runtime-contract <HASH_BOUND_RUNTIME_CONTRACT_JSON> \
  --format-failure-policy <APPROVED_LITERAL_POLICY> \
  --concurrency <APPROVED_LITERAL_CONCURRENCY> \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization <APPROVED_LITERAL_VALUE> \
  --engine-seed <APPROVED_LITERAL_ENGINE_SEED>
```

4. Conversion (CPU, after all 2,000 identities are present in verified COMPLETE shards):

```bash
python3 experiments/training/sft1/vagen_step60_convert.py \
  --partition-manifest <PARTITION_MANIFEST> \
  <TWENTY_LITERAL_--shard-dir_ARGUMENTS> \
  --output-dir <UNUSED_UNIQUE_DATASET_OUTPUT> \
  --latent-token-count 16
```

## Output, resume and monitoring

Stable group: `outputs/experiments/training/sft1-vagen-step60/`. Merge, smoke, concurrency gate, formal batch1 and conversion each require a different unused run directory. W&B is not required; if operational metrics are enabled later, its project/name/ID must be added before launch approval.

Shard resume accepts only a hash-valid `COMPLETE` marker whose raw records, source identities, image bytes, runtime/policy contracts, reward semantics and counts all match. Partial/failed directories remain evidence and are never consumed. Conversion has no partial resume; retry uses a fresh unique directory.

Monitor scheduler state, process/log errors, service health, GPU utilization, response-format failures, image creation, terminal non-step evidence and COMPLETE publication. Cancellation may target only the exact task-owned job ID recorded after submission and never deletes partial output.

## Resource direction and pending literals

Human-selected direction: `normal`, one node, four GPUs total: policy TP2 on two GPUs plus two environment GPUs. W-007 must re-query availability immediately before submission and fill exact CPUs, memory, walltime, device binding, hold-allocation/srun commands, job identity, expected duration and cancellation command.

## Current blockers

1. Exact source commit object/worktree is still inaccessible; accessible VAGEN lineages are not substitutes.
2. Task code is uncommitted; no approved commit or clean remote worktree exists yet.
3. Merge/load and real service smoke are launch-gated and have not run.
4. Literal output paths, resource values and commands are incomplete, so this draft cannot authorize submission.
