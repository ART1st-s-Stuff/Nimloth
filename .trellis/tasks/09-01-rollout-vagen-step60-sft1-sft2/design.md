# Design — VAGEN step60 rollout → SFT1/SFT2 datasets

## 1. Scope and ownership

This task prepares the reusable code and launches only **batch1**: 2,000 source train rows, composed of the first 1,000 `base` rows and first 1,000 `common_sense` rows in category-local parquet order.

Ownership boundaries:

- source checkpoint/parquet: read-only external inputs;
- `experiments/training/sft1/`: experiment entrypoints, partition/conversion/validation and Slurm orchestration;
- `src/nimloth/environment/navigation/`: only reusable environment/session behavior needed to preserve explicit source row identity and source protocol;
- `src/nimloth/rollout/`: versioned trajectory validation and optional audit-field contract;
- remote `outputs/experiments/training/sft1-vagen-step60/`: unique runtime outputs, datasets and manifests;
- `external/VAGEN`: dependency object store and reconstruction base; implement the isolated runtime patch only in a dedicated VAGEN worktree/branch, never in the live checkout and never through an unreviewed Nimloth gitlink change;

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

## 5. Evidence-backed source runtime and policy adapter

The exact source commit `fee3ffac...` is retained as checkpoint/source-run provenance but is unavailable for execution. The approved reconstruction uses VAGEN commit `3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a` as its explicit base. That commit descends from legacy `44be18c`, preserves the required Flask batch API and navigation dynamics, and already isolates compatibility prompt/parser modes.

A dedicated VAGEN branch `task/step60-runtime-reconstruction` adds one isolated `step60_source_reconstruction` mode and tests. The local VAGEN controller worktree is `/workspace/remote2/nimloth/external/VAGEN`, its measured common Git dir is `/workspace/remote2/nimloth/.git/modules/external/VAGEN`, and its task worktree is `/workspace/remote2/nimloth/.worktree/vagen-step60-runtime-reconstruction-vagen`. The remote controller worktree is `/project/peilab/atst/nimloth/external/VAGEN`, its measured common Git dir is `/project/peilab/atst/nimloth/.git/modules/external/VAGEN`, and its runtime worktree is `/project/peilab/atst/nimloth/.worktree/vagen-step60-runtime-reconstruction-vagen`. The exact push refspec is `HEAD:refs/heads/task/step60-runtime-reconstruction`. Creation verifies each worktree top-level, branch, `git rev-parse --path-format=absolute --git-common-dir`, clean status and base commit against its named controller; cleanup uses reviewed non-force exact-path worktree removal. Both remain separate from the live detached submodule checkout, and the Nimloth gitlink remains unchanged.

The patch must:

- consume a versioned canonical evidence artifact generated by Nimloth from W&B run `2q620nss`, hash-pinned table `generations_14_6bc61d7bb480498be805.table.json` row 0 / `output_1`, using the byte procedure in `research/reconstruction-evidence-2026-09-01.md`, and prove the three task-pinned hashes;
- use the strict compact `<think>...</think><answer>one_action</answer>` parser and action order;
- pin `step_length=0.5`, `success_threshold=1.5`, `success_reward=10.0`, `format_reward=0.02`, `invalid_action_penalty=-0.2`, one action and no state reward;
- reproduce the observed reward classes: strict valid non-success `0.02`, strict valid success `10.02`, invalid/forbidden/typo action or malformed strict envelope `-0.2`, and more than one otherwise valid action `0.0`; parser-invalid turns execute no action, while a valid THOR action that physically fails remains a valid action with `0.02` and records `last_action_success=false`;
- bind `base.json`/`common_sense.json` hashes and 60-row counts, with sampled source-evaluator instruction equality under `seed % 60`, then preserve the legacy batch endpoints and return aligned observation/reward/done/info identities;
- remain isolated from every existing prompt mode and include golden valid, invalid-format/action, too-many-action, physical-failure and success tests.

Nimloth owns `experiments/training/sft1/extract_vagen_step60_evidence.py`. It writes exactly one non-overwriting, canonical JSON artifact with schema `vagen_step60_reconstruction_evidence_v1` to the reviewed tracked fixture path `experiments/training/sft1/fixtures/vagen_step60_reconstruction_evidence_v1.json`. The artifact contains source path/table SHA/row/column, extraction version, literal system/initial/post-step prompt templates, their raw/normalized hashes, reward-class counts and normalization metadata, but no assistant response or CoT. It is written atomically only when the target does not exist. During implementation it is copied byte-for-byte, with destination nonexistence and SHA equality checks, to `vagen/env/navigation/evidence/vagen_step60_reconstruction_evidence_v1.json` in the isolated VAGEN worktree; VAGEN reads that committed artifact rather than a private W&B path or manually retyped prompt. Both repositories bind the same evidence-artifact SHA256 in tests and reconstruction manifests.

The reviewed VAGEN patch is exactly one non-merge commit whose sole parent is `3003c2e...`; `git rev-list --count 3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a..HEAD` must equal one and `git rev-list --parents -n 1 HEAD` must contain exactly HEAD plus that parent. Before launch Nimloth computes and verifies: clean status; `HEAD`; `HEAD^`; `HEAD^{tree}`; and SHA256 of `LC_ALL=C git --no-pager diff --binary --full-index --no-ext-diff 3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a..HEAD --`. The approval sequence is strict: human approves VAGEN diff/commit → commit is created → HEAD/tree/diff and canonical reconstruction-manifest payload hash are computed and reviewed → human separately approves exact refspec push → those literals are bound in Nimloth → Nimloth diff/commit and push receive their own approvals. A generated reconstruction manifest binds Git version, unavailable source commit, prompt evidence hashes, environment/reward fields, service API and test evidence; its canonical JSON uses UTF-8, sorted keys and separators `(',', ':')`, excludes only its own `manifest_sha256`, and hashes with SHA256. All identity values are computed from Git and compared with independently approved literals, never trusted from a self-declared manifest; relabeling another commit as `fee3ffac...` is prohibited.

The behavior adapter then renders the actual reconstructed source conversation and samples with the approved source contract:

- checkpoint: merged step60 actor;
- prompt/action: archived source `grounding_worldmodeling` golden transcript and runtime server prompt must agree;
- action format: strict `<think>...</think><answer>one_action</answer>`;
- action names: source legacy navigation names mapped bijectively to Nimloth action indices;
- sampling: source log field `actor_rollout_ref.rollout` has `do_sample=true`, temperature `0.7`, top-p `0.95`, top-k `-1`, `n=1`, `ignore_eos=false`; this run exposes no separate `actor_rollout_ref.rollout.val_kwargs` block;
- max turns `20`, max response tokens `256`; source W&B requirements record vLLM `0.8.5.post1`, Transformers `4.49.0` and PyTorch `2.6.0`, while the human-selected executable reconstruction runtime is the accessible `.venv` with vLLM `0.8.2`, Transformers `4.49.0` and PyTorch `2.6.0`; tokenizer EOS is the only stop boundary, with `ignore_eos=false`, empty custom stop strings and empty custom stop-token IDs. Runtime manifests preserve both source-evidence and executable package triplets and report the vLLM patch-version drift.

The adapter persists the actual generated response and the exact source-rendered prompt/chat messages. It does not invent action probabilities when they are unavailable. Dataset conversion records that behavior token/action log-prob provenance is unavailable rather than inserting a fake one-hot distribution.

Raw collection and training conversion are separate views. The source-audit view is byte-faithful UTF-8 text plus canonical JSON/hash evidence. The training view deterministically rewrites only the source action-format instruction/envelope from `<answer>` to the current K16 Nimloth prompt/response/action contract; task content, observations and real CoT remain unchanged. Each converted record stores source and converted hashes plus the conversion contract version. Source text is never overwritten by the converted view.

## 6. Environment and terminal-state semantics

The reconstructed navigation profile uses the following evidence-pinned source values:

- source row `eval_set` and seed;
- `prompt_format=grounding_worldmodeling`;
- `max_actions_per_step=1`;
- `format_reward=0.02`;
- `invalid_action_penalty=-0.2`;
- `success_threshold=1.5`;
- `step_length=0.5` metres (human-confirmed source VAGEN default, not inferred from the accessible `f7aefd...` runtime);
- `success_reward=10.0` (archived source W&B system-prompt evidence);
- no state reward.

The smoke resolved config and observed trajectory must still verify these pinned values before formal launch.

Every executed turn stores `observation_t`, real response and executed action. Ordinary and terminal generation audit persists `finish_reason`, `stop_reason`, generated token IDs, vLLM/Transformers/PyTorch versions, tokenizer/config hashes and EOS token ID. Both reviewed vLLM `0.8.5.post1` source and selected `0.8.2` executable contracts identify EOS completion by `finish_reason="stop"` plus `stop_reason=null` when custom stop strings/token IDs are empty; preflight/smoke must verify this behavior in the selected executable rather than inferring package parity. An ordinary-turn `finish_reason="length"`, non-null stop reason or missing/other finish reason fails the shard before `/batch/step`, preserving the partial attempt and never executing a guessed/truncated response. A terminal generation with those conditions, or any strict parser failure, retains raw audit evidence but makes the entire linked trajectory ineligible for both SFT1 and SFT2. Formal conversion uses `format_failure_policy=exclude_trajectory` for completed terminal/parser exclusions; the one-row smoke uses `fail_shard`. Ordinary boundary failures are always shard-fatal because a valid next observation cannot be fabricated. After the final environment step, the collector stores `observation_T`, then calls the same policy one more time. It persists:

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

Resume consumes only shards with a valid complete marker whose manifest/hash/counts still match. The human-selected backend drift introduces explicit v3 contracts before any real rollout was collected: runtime contract `vagen_step60_reconstruction_runtime_contract_v3`, raw row `vagen_step60_source_trajectory_v3`, shard manifest and COMPLETE marker `vagen_step60_complete_shard_v3`, dual-view conversion manifest `vagen_step60_dual_view_conversion_v3`, and rejection-sidecar envelope `vagen_step60_rejections_v3`. SFT2 remains structurally `nimloth_trajectory_v1` but its `source_audit.contract_version` is `vagen_step60_reconstruction_audit_v3`; SFT1 source-audit payload uses the same version.

The v3 runtime contract owns two non-overloaded fields: `source_generation_package_evidence={vllm:0.8.5.post1,transformers:4.49.0,torch:2.6.0,evidence:source_wandb_requirements}` and `executable_generation_packages={vllm:0.8.2,transformers:4.49.0,torch:2.6.0}`. Policy runtime records its actual `package_versions`, which must equal the executable field. Raw rows and shard manifests carry the full runtime contract and policy runtime; source audit preserves both through those bound payloads; conversion validators revalidate them from source shards. Existing v1/v2 runtime/raw/shard/COMPLETE/conversion/rejection/source-audit records and any single/overloaded package triplet are rejected without migration. Each v3 record also carries unavailable source commit, reconstruction base/HEAD/tree/diff/manifest hashes and `step_rewards`. The deterministic partition manifest and HF merge manifest remain v1 because their semantics do not change.

A non-empty JSONL alone is never resumable. Failed/partial attempts remain isolated and traceable; retries use a new attempt directory unless the exact complete-shard resume boundary is proven.

Remote `/project` is NFSv3 and returned `EINVAL` for `renameat2(RENAME_NOREPLACE)`. Publication therefore uses the human-selected NFS-compatible reservation protocol rather than weakening no-overwrite safety. The helper accepts exactly one marker from `{partition_manifest.json, COMPLETE, conversion_manifest.json}`. It treats the final target lexically (never `resolve()` on the final path), resolves only both parents to prove one filesystem, rejects any existing file/directory/symlink, and atomically reserves with `mkdir`. The selected marker must be one non-symlink regular file at the staging root; missing, directory, symlink, nested duplicate, or presence of another known marker fails before reservation.

The staging path must be an existing real directory, not a symlink, distinct from the lexical final target, and a direct sibling under the verified same resolved parent. After reservation, the helper creates root sentinel `.NIMLOTH_PUBLISHING.json`, moves all non-marker root payload entries into the reserved target, and fsyncs. It then renames the selected marker to the reserved internal name `.NIMLOTH_READINESS.tmp`, removes the now-empty staging directory, fsyncs target/parent, removes the sentinel, and fsyncs again. The final atomic rename from the internal name to the selected readiness marker is the commit point and intentionally the last fallible publication operation; no fsync or cleanup follows a visible readiness marker. Existing/concurrent publishers fail at `mkdir` and never touch the winner. An interruption before sentinel removal leaves sentinel present and marker absent; interruption after sentinel removal but before the commit rename leaves both sentinel and readiness marker absent, with the internal readiness file retained. Consumers reject both. Neither final target nor remaining staging content is automatically deleted or reused. Call-site handlers may append failure evidence but may not clean either exposed target or remaining staging directory.

Consumers use a path-aware published-artifact gate before parsing: final path must be a real directory, not a symlink; sentinel must be absent; exactly the expected root marker must be a regular non-symlink file; no other known or nested marker is allowed. The partition loader additionally hashes every sibling batch parquet against the manifest before collector/converter use it. Shard and conversion validators apply the same marker/sentinel gate before their existing deep hash/schema validation. Documentation calls this atomic target reservation plus atomic readiness publication, not atomic whole-directory visibility.

Local tests cover lexical dangling symlinks, existing file/symlink/directory preservation, marker type/location, two concurrent reservations, successful marker-last order, and injected mid-publication failure. A fresh remote CPU NFSv3 probe must exercise all three marker types with existing-target, concurrent, marker-last and interrupted-publication rejection before any new GPU request.

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
- aligned finite source `step_rewards` returned by each `/batch/step`; any proposal to fall back to `trajectory_terminal_reward` requires replanning;
- converted terminal CoT prefix;
- separate raw terminal draft-action audit evidence;
- source row/batch/checkpoint/conversion provenance.

Run trajectory and transition expansion validation. Exactly `T` transitions must result; the terminal draft action must not produce transition `T+1`.

## 9. Validation gates

1. Static/unit tests: partition coverage/disjointness, source answer parsing, terminal exclusion from SFT1, terminal prefix conversion, draft action non-execution, atomic shard completion and overlap checks.
2. CPU syntax/schema checks: Python compile, shell syntax, JSON/YAML parsing, task validation and `git diff --check`.
3. Remote CPU preflight: source hashes/paths, clean approved Nimloth commit worktree, clean approved VAGEN reconstruction patch worktree with verified base/diff manifest, checkpoint shard/merge/load, output nonexistence and runtime imports.
4. One-trajectory GPU smoke: exact prompt hash/response format, `T+1` non-uniform RGB images, `T` executed actions, terminal response generated once and never stepped.
5. Production-concurrency gate: one bounded full-size shard before remaining batch1 shards.
6. Batch1 final: 2,000 exact source rows covered once, 1,800/200 split and zero seed overlap, complete-shard hashes, conversion counts, all image/action/message/transition alignment checks, and no claim beyond internal unseen-seed held-out.

## 10. Output and lifecycle

Stable group:

`outputs/experiments/training/sft1-vagen-step60/`

Each merge, smoke, concurrency gate and batch1 attempt uses a distinct date/run directory. No existing path is reused or removed. Every run records both the approved Nimloth commit and the approved VAGEN reconstruction base/patch commits plus reconstruction-manifest hash. W&B, if used, records operational rollout metrics only; static dataset success prevalence is not model evaluation.

The launch contract is prepared for the human-selected `normal` partition and one four-GPU node: policy tensor parallelism 2 on two GPUs and two dedicated environment GPUs. CPU count, memory, walltime, exact device binding and command remain launch-gated and are revalidated against live Slurm availability immediately before submission.

Completion/failure/cancellation/pause triggers `on-experiment-end`. Cancellation targets only recorded task-owned job IDs and never deletes partial outputs.

## 11. Alternatives rejected

- Reuse prior step50 evaluation JSONL: rejected because it is partial and lacks persisted images/final observations.
- Treat source test as held-out: rejected because all 128 `(eval_set, seed)` keys overlap train.
- Generate terminal CoT with a later SFT1 checkpoint: rejected by current human decision; this dataset uses the same step60 policy for all observation-aligned CoT.
- Add terminal response as SFT1 supervision: rejected by human decision because terminal CoT exists only to obtain the final state.
- Infer unavailable behavior probabilities or step rewards: prohibited; missing provenance remains explicit.
- Silently execute `f7aefd...`, `44be18c`, `3003c2e` or another accessible commit as the exact source runtime: rejected. The selected route adds a separately named, evidence-bound patch commit on `3003c2e` and reports it as reconstruction, not source-code parity.
- Install a task-local vLLM `0.8.5.post1` overlay: declined by the human in favor of directly using accessible vLLM `0.8.2`; package drift remains explicit and smoke-gated.

## 12. Rollback

Before launch, rollback means reverting only task-owned Nimloth code/docs and the isolated VAGEN reconstruction branch/worktree; the live VAGEN checkout and Nimloth gitlink remain untouched. After launch, cancel only exact task-owned jobs, retain all logs/partial outputs, mark the attempt terminal, and start any retry in a new unique directory. Source checkpoint/parquet and prior datasets remain untouched.
