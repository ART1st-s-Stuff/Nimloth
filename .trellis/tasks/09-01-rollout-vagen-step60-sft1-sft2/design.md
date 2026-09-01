# Design — VAGEN step60 rollout → SFT1/SFT2 datasets

## 1. Scope and ownership

This task prepares the reusable code and launches only **batch1**: 2,000 source train rows, composed of the first 1,000 `base` rows and first 1,000 `common_sense` rows in category-local parquet order.

Ownership boundaries:

- source checkpoint/parquet: read-only external inputs;
- `experiments/training/sft1/`: experiment entrypoints, partition/conversion/validation and Slurm orchestration;
- `src/nimloth/environment/navigation/`: only reusable environment/session behavior needed to preserve explicit source row identity and source protocol;
- `src/nimloth/rollout/`: versioned trajectory validation and optional audit-field contract;
- remote `outputs/experiments/training/sft1-vagen-step60/`: unique runtime outputs, datasets and manifests;
- `external/VAGEN`: dependency; do not edit its live server checkout or silently change its gitlink.

SFT1/SFT2 model training, cache building and batches2–10 are out of scope.

## 2. Data flow

```text
source train.parquet (SHA256 pinned)
  → deterministic 10-batch partition manifest
  → batch1 parquet + source-row manifest
  → deterministic batch1 split by shared seed ordinal
      ├─ train: 1,800 rows
      └─ internal held-out: 200 rows
  → source VAGEN step60 actor HF merge/export
  → source-protocol rollout (real CoT + executed action)
  → final observation
  → same VAGEN policy full terminal CoT + draft action (not executed)
  → raw shard: verbatim source prompt/chat + every RGB image + environment/result provenance
  → strict dual-view conversion
      ├─ source audit view
      │    verbatim prompt/messages/responses/terminal response + hashes
      ├─ SFT1 train_all / train_success / heldout_all
      │    K16-compatible prompt and only executed-action assistant turns
      └─ SFT2 nimloth_trajectory_v1
           K16-compatible training transcript
           real ordinary CoT + executed actions
           terminal CoT only as terminal state prefix
           terminal draft action only as audit evidence
```

## 3. Deterministic partition contract

A partition command reads the pinned source parquet and assigns category-local row ordinal `0..9999` to batch `ordinal // 1000 + 1`.

The manifest records:

- source path, size and SHA256;
- original global source row index;
- `eval_set`, seed and stable source key;
- category-local ordinal and batch number;
- per-batch parquet SHA256/counts;
- global checks: ten batches × 2,000 rows, no duplicate source index, union exactly `0..19999`, and 1,000 rows per category per batch.

No retry may repartition or resample.

Within batch1, `category_local_ordinal % 10 == 9` is internal held-out. Because the two categories share the same ordered seed sequence, both rows for a seed are assigned to the same side. The manifest must prove 1,800 train + 200 held-out rows and zero overlap by source index, `(eval_set, seed)` and bare seed. This is an internal unseen-seed split, not evidence for unseen environment-distribution generalization.

## 4. Checkpoint restoration

Use VERL's legacy FSDP merger against a read-only copy/view of `global_step_60/actor` and write to a new unique HF export directory. The final implementation must record the exact merger command/version and reject:

- missing/extra world-size ranks;
- incomplete tokenizer/config files;
- an existing target directory;
- non-finite or missing model tensors;
- model/tokenizer vocabulary or architecture mismatch.

A CPU merge/load preflight precedes rollout. If current VERL cannot faithfully load the source shards, stop; do not substitute a different checkpoint or approximate conversion.

## 5. Source-protocol policy adapter

The behavior adapter renders the actual source conversation and samples with the approved source contract:

- checkpoint: merged step60 actor;
- prompt/action: archived source `grounding_worldmodeling` golden transcript and runtime server prompt must agree;
- action format: strict `<think>...</think><answer>one_action</answer>`;
- action names: source legacy navigation names mapped bijectively to Nimloth action indices;
- sampling: `do_sample=true`, temperature `0.7`, top-p `0.95`, top-k `-1`, `n=1`;
- max turns `20`, max response tokens `256`.

The adapter persists the actual generated response and the exact source-rendered prompt/chat messages. It does not invent action probabilities when they are unavailable. Dataset conversion records that behavior token/action log-prob provenance is unavailable rather than inserting a fake one-hot distribution.

Raw collection and training conversion are separate views. The source-audit view is byte-faithful UTF-8 text plus canonical JSON/hash evidence. The training view deterministically rewrites only the source action-format instruction/envelope from `<answer>` to the current K16 Nimloth prompt/response/action contract; task content, observations and real CoT remain unchanged. Each converted record stores source and converted hashes plus the conversion contract version. Source text is never overwritten by the converted view.

## 6. Environment and terminal-state semantics

The source navigation profile uses the verified source values:

- source row `eval_set` and seed;
- `prompt_format=grounding_worldmodeling`;
- `max_actions_per_step=1`;
- `format_reward=0.02`;
- `invalid_action_penalty=-0.2`;
- `success_threshold=1.5`;
- no state reward;
- source/default step length and success reward must be verified in the smoke resolved config before formal launch.

Every executed turn stores `observation_t`, real response and executed action. After the final environment step, the collector stores `observation_T`, then calls the same policy one more time. It persists:

- raw terminal response;
- parsed draft action and format status;
- converted `terminal_assistant_prefix` ending at Nimloth `action_start`.

The terminal action is never sent to the environment, never appended to executed actions, and never receives reward/transition semantics. Neither SFT1 nor SFT2 LLM-backbone labels include the terminal response.

## 7. Raw shard and resume contract

Batch1 is split into bounded complete shards (final shard size selected in the exact launch contract; target 100 rows). Each shard owns a unique directory containing:

- source-row manifest subset;
- raw atomic JSONL containing verbatim source prompt/chat/ordinary responses/terminal full response;
- one image directory per trajectory;
- resolved runtime contract;
- validation result and `COMPLETE` marker written only after validation.

Resume consumes only shards with a valid complete marker whose manifest/hash/counts still match. A non-empty JSONL alone is never resumable. Failed/partial attempts remain isolated and traceable; retries use a new attempt directory unless the exact complete-shard resume boundary is proven.

## 8. Conversion contracts

### SFT1

Produce:

- `train_all`: every strict-valid trajectory in the 1,800-row train partition;
- `train_success`: strict-valid successful trajectories from `train_all`;
- `heldout_all`: every strict-valid trajectory in the 200-row internal held-out partition.

Only responses associated with executed actions become supervised assistant turns. The terminal response is excluded. Conversion preserves real CoT while replacing source prompt instructions and response `<answer>` action envelopes with the current K16 Nimloth format. The record links to the verbatim source prompt/chat and stores both source and converted hashes. Invalid records are excluded with an ID/reason sidecar, never silently repaired. Any later choice between `train_all` and `train_success` is outside this task.

### SFT2

Produce separate train and internal-held-out `nimloth_trajectory_v1` files using the same deterministic split, with:

- a K16-compatible training system prompt and observations in order;
- a separate verbatim source prompt/chat audit view and source/converted hashes;
- `T+1` RGB image paths;
- `T` converted assistant responses and executed action indices;
- source aggregate reward as `trajectory_terminal_reward` unless smoke proves exact step rewards are present;
- converted terminal CoT prefix;
- separate raw terminal draft-action audit evidence;
- source row/batch/checkpoint/conversion provenance.

Run trajectory and transition expansion validation. Exactly `T` transitions must result; the terminal draft action must not produce transition `T+1`.

## 9. Validation gates

1. Static/unit tests: partition coverage/disjointness, source answer parsing, terminal exclusion from SFT1, terminal prefix conversion, draft action non-execution, atomic shard completion and overlap checks.
2. CPU syntax/schema checks: Python compile, shell syntax, JSON/YAML parsing, task validation and `git diff --check`.
3. Remote CPU preflight: source hashes/paths, clean exact commit worktree, checkpoint shard/merge/load, output nonexistence and runtime imports.
4. One-trajectory GPU smoke: exact prompt hash/response format, `T+1` non-uniform RGB images, `T` executed actions, terminal response generated once and never stepped.
5. Production-concurrency gate: one bounded full-size shard before remaining batch1 shards.
6. Batch1 final: 2,000 exact source rows covered once, 1,800/200 split and zero seed overlap, complete-shard hashes, conversion counts, all image/action/message/transition alignment checks, and no claim beyond internal unseen-seed held-out.

## 10. Output and lifecycle

Stable group:

`outputs/experiments/training/sft1-vagen-step60/`

Each merge, smoke, concurrency gate and batch1 attempt uses a distinct date/run directory. No existing path is reused or removed. W&B, if used, records operational rollout metrics only; static dataset success prevalence is not model evaluation.

The launch contract is prepared for the human-selected `normal` partition and one four-GPU node: policy tensor parallelism 2 on two GPUs and two dedicated environment GPUs. CPU count, memory, walltime, exact device binding and command remain launch-gated and are revalidated against live Slurm availability immediately before submission.

Completion/failure/cancellation/pause triggers `on-experiment-end`. Cancellation targets only recorded task-owned job IDs and never deletes partial outputs.

## 11. Alternatives rejected

- Reuse prior step50 evaluation JSONL: rejected because it is partial and lacks persisted images/final observations.
- Treat source test as held-out: rejected because all 128 `(eval_set, seed)` keys overlap train.
- Generate terminal CoT with a later SFT1 checkpoint: rejected by current human decision; this dataset uses the same step60 policy for all observation-aligned CoT.
- Add terminal response as SFT1 supervision: rejected by human decision because terminal CoT exists only to obtain the final state.
- Infer unavailable behavior probabilities or step rewards: prohibited; missing provenance remains explicit.

## 12. Rollback

Before launch, rollback means reverting only task-owned code/docs. After launch, cancel only exact task-owned jobs, retain all logs/partial outputs, mark the attempt terminal, and start any retry in a new unique directory. Source checkpoint/parquet and prior datasets remain untouched.
