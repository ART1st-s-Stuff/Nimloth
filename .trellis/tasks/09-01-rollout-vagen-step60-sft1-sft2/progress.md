# Progress — VAGEN step60 batch1 rollout datasets

## 2026-09-01 — W-001 RED contract tests

Completed:

- Recorded the human-selected dual-view prompt/chat contract: K16-compatible SFT1/SFT2 training views plus verbatim source prompt/full chat/terminal response audit evidence and hashes.
- Added `tests/training/sft1/test_vagen_step60_data.py` covering:
  - ten balanced/disjoint 2,000-row batches and exact 20,000-row union;
  - batch1 1,800/200 shared-seed split with zero bare-seed overlap;
  - source `<answer>` to K16 conversion while preserving verbatim source text;
  - terminal response exclusion from SFT1/backbone supervision;
  - terminal draft-action audit without execution or extra transition;
  - unavailable behavior-probability provenance and incomplete-shard rejection.

Evidence:

- `python3 -m py_compile tests/training/sft1/test_vagen_step60_data.py` passed.
- Expected RED import failure: `No module named 'experiments.training.sft1.vagen_step60_data'`.
- The local system Python lacks `pytest`, `torch` and `PIL`; a real pytest run remains pending in the approved project runtime. This limitation is not reported as a test pass.

Next:

- W-002 implements the deterministic partition manifest and strict source-row/split validation required by the first two RED tests.

Memory/spec review:

- No new curated memory proposed. The durable dual-view rule is task-specific and is now explicit in the reviewed PRD/design rather than duplicated in memory.

## 2026-09-01 — W-002 deterministic partition manifest

Completed:

- Added `experiments/training/sft1/vagen_step60_data.py` with a non-overwriting parquet partition entrypoint and pure manifest builder.
- Validates the pinned train SHA256, exact 20,000-row count, exact remote parquet schema/config, category ordering, unique `(eval_set, seed)` keys and shared ordered seed sequence.
- Produces ten balanced 2,000-row batches with per-batch source indices, row-manifest hash and parquet hash.
- Labels only approved batch1 as 1,800 train + 200 internal held-out; future batches remain `unassigned`.
- Added explicit overlap measurement and rejection of an overlapping candidate mislabeled as held-out.
- Indexed the entrypoint in `experiments/training/sft1/README.md`.

Evidence:

- Read-only remote schema audit confirmed `extra_info.seed` and `extra_info.env_config.eval_set` plus the pinned source env fields.
- Python compile passed for the new module and RED test file.
- Direct pure-function checks passed for 20,000-row coverage, ten balanced batches, 1,800/200 counts, zero batch1 overlap, source-test 128-key overlap, and count/order/shared-seed drift rejection.
- `git diff --check` passed for the W-002 source/test/README files.
- Full pytest and an actual pyarrow parquet write remain pending because the local system runtime lacks project dependencies; no remote code was edited or experiment launched.

Next:

- W-003 adds source checkpoint shard/export preflight tooling. The actual merge/load remains launch-gated.

Memory/spec review:

- No curated memory proposed. Exact source parquet schema evidence was added to this task's research record.

## 2026-09-01 — W-003 checkpoint merge/preflight tooling

Completed:

- Added `experiments/training/sft1/vagen_step60_checkpoint.py`.
- Source inspection requires exactly eight model shards and eight extra-state shards, the verified config/tokenizer sidecar, Qwen2.5-VL architecture/model type/vocabulary, and no pre-existing HF weights.
- Merge planning rejects an existing target and records the exact Python, legacy VERL FSDP merger command, script hash, component mapping and source VAGEN commit.
- Execution is inert without `--execute`; an executed merge requires source-shard hashes, keeps a failed unique partial target for evidence, and never loads critic/optimizer/PPO state.
- Post-merge validation performs a real local HF model/processor load, rejects loading-info mismatches and non-finite tensors, validates embedding/tokenizer bounds, and hashes every exported artifact before writing the success manifest.
- Added focused checkpoint contract tests and indexed the entrypoint in the SFT1 README.

Evidence:

- Read-only remote listing reconfirmed eight model shards (rank 0–7), eight extra-state shards, and the exact nine-file config/tokenizer sidecar with no model weights.
- Source config is `Qwen2_5_VLForConditionalGeneration`, `qwen2_5_vl`, vocabulary 151,936.
- Current candidate merger provenance: VAGEN dependency `9f1e89eb8c9839a406b6e62aa75703494a79e5b5`; legacy merger SHA256 `3e2794e1e9e566a4aeb0d709dad7d2b8864c8b91e4f72cf0d265ecb62c311044`.
- Python compile, fake eight-shard inspection/plan gates and `git diff --check` passed.
- Actual 19 GB source hashing, merge and HF load were not run; they remain explicitly gated by experiment launch approval.

Next:

- W-004 implements source-protocol rollout collection, terminal generation, verbatim source chat persistence and atomic complete-shard resume.

Memory/spec review:

- No curated memory proposed; checkpoint provenance and remaining runtime uncertainty stay in the task record.

## 2026-09-01 — W-004 source collector and atomic shard contract

Completed:

- Added `experiments/training/sft1/vagen_step60_collect.py` with the exact legacy batch-service HTTP contract, source config, strict response parser and stable source-row identities.
- Validates archived source system/initial prompt hashes before any action and stores verbatim source messages plus every full processor-rendered policy prompt.
- Uses the source five-turn history window, at most six images, explicit step60 sampling fields and a required launch-time engine seed.
- Collects bounded microbatches inside deterministic balanced shards with globally unique environment IDs.
- Saves every observation including the final observation, then generates one full terminal CoT+draft action with the same policy and never calls environment step afterward.
- Persists ordinary responses, environment-extracted actions, raw per-step reward/info, terminal audit, exact images and eligibility/rejection reasons.
- Publishes `raw.jsonl`, image hashes, `shard_manifest.json` and `COMPLETE` only in a partial attempt directory; validates the complete payload before atomically renaming to the final unique shard path. Failed attempts lose `COMPLETE` and remain traceable.
- Added complete-shard consumption validation to `vagen_step60_data.py`, focused collector tests and README indexing.

Evidence:

- Read-only server audit verified the archived prompt hashes across both source categories and confirmed source sampling/window fields from the resolved training log.
- Focused lightweight suite: 13 passed, 2 W-005 conversion tests deselected.
- Ruff passed for all new step60 scripts/tests; Python compile and `git diff --check` passed.
- Fake service/policy test proves two ordinary environment actions produce exactly two terminal generations and zero terminal environment steps, with `T+1` images and atomic complete-shard validation.

Incomplete/runtime blocker:

- The exact source checkout commit `fee3ffac036a599b0ae979a6dd1ce2b21f7dec49` is permission-denied and absent from accessible VAGEN object databases. The collector rejects other commits; `44be18c`/`f7aefd` are not substituted because their prompt/config differ. Exact runtime access must be resolved before launch approval.
- No checkpoint merge, environment server, GPU model load or rollout was started.

Next:

- W-005 implements the dual-view SFT1/K16 `nimloth_trajectory_v1` conversion and makes the remaining two RED tests pass.

Memory/spec review:

- No curated memory proposed. The exact-source access blocker and prompt hashes are task evidence, not a general project memory.

## 2026-09-01 — W-005 dual-view SFT1/SFT2 conversion

Completed:

- Added strict source response parsing and deterministic action-format-only K16 prompt/response conversion to `vagen_step60_data.py`.
- Converted SFT1 records contain only ordinary executed-action turns and `T` images; the terminal observation/response is absent from supervision.
- Converted SFT2 records use `nimloth_trajectory_v1`, K16 prompts, `T+1` observations/images, `T` responses/actions, a terminal CoT prefix ending at action start, and a separate unexecuted draft-action audit.
- Both views embed the byte-faithful source prompt/chat/policy-request audit and bind source/converted hashes plus conversion version.
- Source behavior action/token probabilities remain empty with `policy_credit_assignment=none`; rollout validation now explicitly permits this only for offline non-planner records rather than inventing one-hot probabilities.
- Added optional rollout source/terminal/conversion audit fields with backward-compatible decoding and fail-closed terminal non-execution validation.
- Added `vagen_step60_convert.py` to consume only exact COMPLETE shards, enforce 2,000-row coverage and 1,800/200 split identity, emit SFT1 train_all/train_success/heldout and SFT2 train/heldout, validate every trajectory/transition, write rejection sidecars and atomically publish a hash manifest satisfying input = valid + excluded.
- Updated SFT1 and rollout ownership READMEs.

Evidence:

- Focused full local dependency suite: 45 passed across new step60 tests, rollout package tests and `tests/test_wm_transition_dataset.py`.
- The W-001 conversion RED tests are now GREEN, including verbatim source preservation, K16 response checks, terminal exclusion, empty behavior-probability provenance and exactly `T` transitions.
- Ruff, Python compile and `git diff --check` pass for the changed implementation/test paths.

Not yet runtime-verified:

- No real raw shard exists yet, so the full 2,000-row conversion orchestrator has not run against production data; its per-record/schema/hash paths are locally tested, while batch-level evidence remains a post-rollout gate.
- Reward provenance remains an explicit required conversion argument. The launch smoke must decide `step_rewards` only if returned per-step semantics are verified; otherwise use `trajectory_terminal_reward`.

Next:

- W-006 runs full-scope checks, complete-diff review and prepares the exact committed-source launch contract without launching.

Memory/spec review:

- No curated memory proposed. The offline `policy_credit_assignment=none` contract is now documented in the rollout README and code validation rather than duplicated in memory.

## 2026-09-01 — W-006 full-scope review reopened implementation items

The read-only `trellis-check` review found blockers that invalidate the prior completion state of W-002 through W-005. Their checkboxes were reopened; W-006 is blocked until remediation and re-review.

Required fixes:

- bind source identity/reward/actions/images/runtime/checkpoint provenance into canonical hashes;
- stop appending K16 instructions to observation content that has no source format block;
- restrict unavailable behavior probabilities/messages to a strict versioned offline conversion contract;
- validate the full pinned partition manifest in every consumer;
- preserve source identity through `RolloutTrajectory` roundtrip;
- bind/revalidate merged artifact hashes before collection;
- make reward provenance a smoke-verified contract rather than an arbitrary relabel;
- add conversion orchestrator/tamper/atomic no-replace tests;
- keep exact `fee3ffac...` source runtime access as a launch blocker.

This review is not a regression pass and no completion/commit/launch claim is made.

### W-002 remediation

- Added one shared partition consumer validator that recomputes pinned source identity, exact union/order, category ordinals, all ten batch summaries/hashes, 1,800/200 split and overlap evidence.
- Collector and converter now require a sealed published manifest rather than trusting labels/counts.
- Directory publication now uses Linux `renameat2(RENAME_NOREPLACE)` and fails closed instead of replacing an existing target.
- Added partition-tamper and concurrent-target preservation tests; focused result: 7 passed.

### W-003 remediation

- Merge success manifests now have a recomputable payload hash and require all 16 source shard hashes.
- Collection startup revalidates every merged artifact file size/hash and the artifact-manifest hash instead of trusting a stale manifest.
- Raw records and shard manifests now bind merge-manifest file/payload hashes, merged artifact hash and source actor directory.
- Added a post-merge artifact tamper test; checkpoint+collector focused result: 9 passed.

### W-004 remediation

- Added the archived normalized post-step prompt hash, established from 71 sampled turns across both categories after removing only the evaluator splitter's optional trailing blank line.
- Collector now gates initial and every post-step prompt template; parsed policy action must still match runtime-extracted action.
- Each raw record has a canonical hash covering identity, full chat, actions, rewards/success, policy/runtime contracts and per-image byte hashes. Complete-shard validation recomputes it and rejects semantic tampering even if outer JSONL/manifest hashes were rewritten.
- Added a required hash-bound exact-source runtime contract for commit/root, legacy batch API, 500-second timeout, action vocabulary, prompt hashes, resolved step length and success reward.
- The exact `fee3ffac...` runtime remains inaccessible and is explicitly still a launch-preflight blocker; no accessible lineage is substituted.
- Focused partition/collector/shard tests pass (7 passed in the selected remediation subset); Ruff passes.

### W-005 remediation

- Format conversion now leaves observations without a source action-format block byte-identical; it never appends synthetic K16 instructions to arbitrary observation content.
- Source audit now validates and binds the collector's full raw-record hash, source identity, runtime/checkpoint contracts, actions, rewards/success, turns, image artifacts and complete chat.
- Added versioned `source_identity` to standard `RolloutTrajectory` serialization so identity survives load/save roundtrip.
- Empty behavior probabilities/messages are accepted only for the exact hash-verified step60 offline conversion contract; the public validator no longer permits an arbitrary `policy_credit_assignment=none` bypass.
- Reward provenance is now fixed in the smoke-verified raw source-runtime contract and cannot be changed by a conversion CLI flag.
- Added orchestrator tests for exact 2,000-row accounting, 1,800/200 outputs, rejection sidecar, paired SFT1/SFT2 counts, no-replace publication, duplicate shard indices and missing coverage. The new test exposed and fixed a tuple/source reward-provenance counting bug.
- Current focused full result: Ruff passed and 52 tests passed across all step60 tests, `tests/rollout` and `tests/test_wm_transition_dataset.py`.

### Second W-006 review remediation

A second read-only check found three residual blockers. They were fixed before resuming W-006:

- complete-shard validation now binds every raw record to shard source keys, batch, exact runtime commit/contract, policy artifact/runtime, format-failure policy and recomputed counts;
- the converter rejects cross-shard checkpoint/runtime contract mixtures and validates source key and batch against the pinned partition, with no default identity values;
- step-reward conversion rejects any raw aggregate reward that differs from the sum of raw step rewards rather than silently replacing it.

Tests now cover outer-hash-resealed provenance tampering, mixed-shard provenance, missing source key and aggregate reward drift. The residual focused result is 48 passed plus Ruff.

### W-006 final quality and pre-launch draft

- The final complete-shard reward gate validates row/runtime provenance, finite turn/event/claimed rewards, turn-event agreement, finite step aggregate equality, and explicit terminal aggregate equality before a shard can resume.
- Multiple read-only `trellis-check` passes were repeated until no blocker/high code finding remained; launch-only blockers are kept separate.
- Full planned affected test command passed: 80 tests across all `tests/training/sft1`, `tests/rollout` and `tests/test_wm_transition_dataset.py`.
- Ruff, compileall, shell syntax, Trellis task validation and `git diff --check` pass.
- Added `research/prelaunch-contract-draft-2026-09-01.md` with pinned inputs, generation/split/freeze semantics, staged entrypoints, output/resume/monitoring contract and every still-pending literal. It explicitly is not launch authorization.
- No checkpoint merge, GPU smoke, rollout, commit or launch occurred.

Commit gate:

- The complete task-owned diff is ready for human commit review. Unrelated dirty changes, protected memory, other tasks, `.pi/task-tree/`, external dependencies and runtime outputs must remain excluded.
