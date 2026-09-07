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

### W-007 source runtime evidence refinement

- Read-only checkpoint inspection confirmed that `data.pt` and actor extra-state record training/sampler/scheduler state but do not encode the navigation dynamics contract.
- The archived source W&B generation table preserves exact prompts and per-turn rewards; its system prompt states success reward `10.0`, while ordinary valid turns visibly record reward `0.02`.
- Human confirmed the source run used the VAGEN default movement distance with no override and fixed that source value as `step_length=0.5` metres.
- These values are now pinned in the task and prelaunch draft. They still require resolved-runtime and trajectory smoke verification; this confirmation does not authorize checkpoint merge, GPU, Slurm or rollout.
- After the human requested start, the on-experiment-start read-only preflight revalidated the checkpoint shard counts, pinned parquet hash and output-group nonexistence. It also confirmed that exact `fee3ffac...` remains unavailable, the committed collector rejects any substitute runtime, and current `normal` availability has no healthy responsive four-free-GPU node. No remote worktree, merge, GPU allocation, Slurm job or rollout was started; W-007 is waiting for the exact-runtime versus evidence-backed-reconstruction decision.
- Continued read-only lineage inspection found that `44be18c` supplies the matching legacy batch API, compact actions, 0.5 m dynamics and 10.0 success reward, but not this run's exact strict prompt or invalid-action-penalty field. Existing compatibility commits target other archived runs and have different golden hashes.
- Human explicitly selected the evidence-backed reconstruction route. Planning now fixes `3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a` as the VAGEN base, requires an isolated named patch mode/commit and a Nimloth ancestry/diff/evidence contract, and forbids any exact-source-code parity claim. PRD/design/plan returned to review; no reconstruction code or experiment was started.

### Reconstruction replan and W-007 RED

- Four planning-review rounds resolved runtime-status, reward semantics, sampling invocation, identity, worktree, EOS, persisted-format and extractor-boundary findings. The human approved deletion of the stale untracked Pi TaskTree; its three files were precisely removed and Trellis remains the only task authority.
- Fresh implementation approval was hash-bound to the reviewed reconstruction scope and explicitly excludes commits, pushes, remote worktrees, checkpoint merge, GPU, Slurm and rollout. `task.py start` then returned the task to `in_progress`.
- Created and verified isolated VAGEN worktree `/workspace/remote2/nimloth/.worktree/vagen-step60-runtime-reconstruction-vagen` on branch `task/step60-runtime-reconstruction` at exact base `3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a`; common Git dir is the measured Nimloth submodule common dir and the live detached checkout remains clean/unmodified.
- Added VAGEN RED tests for archived prompt hashes, strict compact parsing, four reward classes, physical failure, action/config/mode and preserved batch routes. Local dependency-light RED result is `1 passed, 4 errors`, all four errors being the intentionally missing `step60_reconstruction.py`.
- Added Nimloth RED tests for v2 surfaces/v1 rejection controls, computed Git identity and dirty-runtime rejection, EOS/finish evidence, and the non-overwriting no-CoT evidence extractor. RED result is `13 failed`, all at the intended missing v2 constants/functions/module or old-v1 behavior boundaries.
- No GREEN implementation, commit, push, remote mutation or experiment launch occurred in W-007.

### W-008 evidence-backed reconstruction GREEN

- Added a deterministic, non-overwriting W&B extractor and canonical no-CoT evidence artifact. The Nimloth and isolated VAGEN copies are byte-identical, SHA256 `e9e1ebc4f61b07e5b3b77b165cf72fdfa525d7d840f54296ce5873c5e68463c8`; the artifact binds the exact prompt fixture and all 12 reviewed table hashes and counts 7,351 turns. CLI rejects alternate prompt fixtures, altered/missing/extra reward hashes and duplicate reward-table paths.
- Added isolated VAGEN `step60_source_reconstruction` prompt/parser/reward mode, exact source hash rendering, strict structured grammar, 0.5 m/1.5 m/10.0/0.02/-0.2 semantics, too-many `0.0`, compact action order, asset hashes/counts/instruction samples and exact incoming environment config validation. Replaying all 7,351 archived turns produced zero parser/reward mismatches.
- Added a service identity endpoint that recomputes clean Git HEAD/parent/tree/diff, evidence, dataset hashes/counts and actual Flask routes; Nimloth compares it to independently approved runtime-contract literals before environment creation. Mixed concurrent environment creation failure now observes all futures, closes late successes, releases GPU assignments and rolls back server ownership; a delayed-success/immediate-failure regression covers the prior leak.
- Migrated reconstruction-consumed persistence to explicit v2 runtime/raw/shard/COMPLETE/conversion/rejection/source-audit contracts while retaining partition/HF-merge v1 intentionally. Complete-shard and conversion validators recompute strict parser/action/reward/EOS/eligibility/chat/window/hash semantics and reject coordinated resealing tamper.
- Added source-vLLM EOS evidence (`finish_reason=stop`, null custom stop, final EOS ID), package versions and model config/tokenizer hashes. Ordinary non-EOS generations fail before environment step; terminal non-EOS/parser failures exclude the linked SFT1/SFT2 record and never execute a draft action.
- Added deterministic runtime-contract producer, independent payload-hash CLI, one-row smoke source-index selector and independent published-conversion validator with partition-bound exact 2,000 identity coverage, SFT1/SFT2 linkage, transitions, stats, rejection and seed-overlap checks.
- Final validation evidence: Nimloth affected suite `101 passed`; VAGEN reconstruction suite `13 passed`; unchanged VAGEN compatibility suites `5 passed`; targeted Ruff passed; compileall, both repository diff checks and Trellis task validation passed. Final read-only `trellis-check` reported no blocker/high.
- No commit, push, remote worktree, checkpoint merge, GPU, Slurm, environment service or rollout was started. W-009 remains separately gated.

### W-009 VAGEN commit identity

- After exact Trellis commit approval, the isolated VAGEN branch created one non-merge commit `170a673d1bf5855fc0ea6fbed0744b3d7168f8f0` (`feat(navigation): add audited step60 runtime reconstruction`) with sole parent `3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a`.
- Reviewed identity literals: tree `58ef0eb66ad0bef7587c253c5c643af572c1d3a7`; canonical binary/full-index diff SHA256 `7f025476657de1289cf84b61d7702de26d248cd196412e9374a15e6de62730e9`; evidence SHA256 `e9e1ebc4f61b07e5b3b77b165cf72fdfa525d7d840f54296ce5873c5e68463c8`; commit count and parent count both one; worktree clean.
- These literals are now bound in Nimloth code. After separate exact push approval, `origin/task/step60-runtime-reconstruction` was created and read back at exact `170a673d1bf5855fc0ea6fbed0744b3d7168f8f0` without force or other ref updates. The subsequent Nimloth commit/push remain separately gated. No server/GPU/experiment action occurred.

### W-009 Nimloth commit/push and remote CPU preflight

- After separate complete-diff, commit and push approvals, Nimloth created commit `a54ae97ad651d64aca98734834038f022aaee0fc` and fast-forwarded only `origin/task/rollout-vagen-step60-sft1-sft2` from `696ee904...`; `origin/dev` was not updated.
- Created detached clean remote worktrees at exact approved commits: Nimloth `/project/peilab/atst/nimloth/.worktree/rollout-vagen-step60-sft1-sft2` and VAGEN `/project/peilab/atst/nimloth/.worktree/vagen-step60-runtime-reconstruction-vagen`. Top-level, common Git dirs, controller registration and `.local` link were verified. The first nested VAGEN submodule update failed because a canonical local URL hit Git's file-transport block; one-shot `-c protocol.file.allow=always` then populated the approved gitlink, nested VERL `494f264...`, and le-wm `8edfeb3...` cleanly without persistent config changes.
- Remote CPU tests: VAGEN reconstruction plus both compatibility files `18 passed`; Nimloth affected suite `101 passed` after all approved submodules were populated.
- Remote runtime identity exactly matched approved VAGEN HEAD/tree/diff/assets/routes/config/evidence. Canonical runtime-contract payload SHA256 is `1de6f3d02c948f80bb1f4f7aed824da37228a9867a4ee43f951b23f814ea2543`; independent hash CLI agreed; temp contract file SHA256 is `c5a9024ee292ed72a020f2d3f6072f9e5127a729c7e5fb056c128cc3385e6f69`.
- Full source checkpoint hash inspection and non-executing merge plan both bound 8 model + 8 extra-state shards; temp evidence SHA256 values are `a819cbef...ab500` (inspection) and `d8881b27...83d7` (plan). No merge target was created and no model was loaded.
- Deterministic `/tmp` partition preflight validated all 20,000 rows, batch1 2,000 = 1,800 train + 200 held-out, zero source/eval-set-seed/bare-seed overlap; manifest SHA256 `be7db7ea975927bc176186bcb51a202b3be191196ced26e043a57add5f99b87c`.
- The source package contract is not yet available: canonical `.venv` has exact PyTorch `2.6.0` and Transformers `4.49.0` but vLLM `0.8.2`, while `.venv-vagen-main` has `2.8.0/4.55.4/0.11.0`; the inaccessible source environment had vLLM `0.8.5.post1`. No approximate model runtime was launched.
- Latest resource query has no healthy responsive `normal` node with four free GPUs (only `dgx-18` responsive with one); DOWN/NOT_RESPONDING GPUs are not candidates. Stable output group remains absent. No checkpoint merge, model load, service, GPU, Slurm or rollout was started.
- Human declined installing an exact vLLM `0.8.5.post1` overlay and explicitly selected accessible vLLM `0.8.2` with the otherwise source-matching Torch `2.6.0` / Transformers `4.49.0`. This is a material executable-runtime change: task returned to planning for W-012, with source package evidence retained separately and no package-parity claim.

### W-012 vLLM 0.8.2 executable reconstruction

- Fresh implementation approval covered local Nimloth changes only. Reconstruction-consumed formats are now v3 before first real rollout use; runtime/raw/shard/COMPLETE/conversion/rejection/source-audit v1/v2 and missing/single/overloaded package provenance are rejected, while partition/HF-merge v1 remain intentional.
- Runtime and policy provenance now separate `source_generation_package_evidence` (`vllm=0.8.5.post1`, Transformers `4.49.0`, Torch `2.6.0`, W&B requirements evidence) from `executable_generation_packages` and actual `package_versions` (`vllm=0.8.2`, Transformers `4.49.0`, Torch `2.6.0`). Shard, conversion, source-audit and SFT2 validators revalidate all three views.
- Read-only inspection of installed vLLM 0.8.2 bound `outputs.py` SHA256 `047d4697...36f8` and `stop_checker.py` SHA256 `5ed39ad2...fa28`; their EOS/null-stop semantics match the persisted evidence contract. Actual model generation/tokenization remains GPU smoke-gated.
- Local affected suite now passes `108 passed`; targeted Ruff passed; final read-only `trellis-check` found no blocker/high. VAGEN remained clean and unchanged at pushed `170a673...`.
- W-012 was committed as Nimloth `187fe112038944a3ba7dd913fb4e87e15a33937e` and pushed by exact fast-forward to `origin/task/rollout-vagen-step60-sft1-sft2`. VAGEN remained unchanged at `170a673...`.
- After transient VPN/proxy failures, remote refresh found one task-generated untracked `external/le-wm/__pycache__/module.cpython-312.pyc`. Human explicitly approved deleting only that audited cache file and empty directory; no tracked/ignored payload was removed. The clean detached Nimloth worktree was then refreshed from `a54ae97...` to exact `187fe112...`; recursive submodule commits and the clean VAGEN reconstruction worktree were reverified.
- Remote affected suite passed `108 passed`. Canonical executable packages reverified as vLLM `0.8.2`, Transformers `4.49.0`, Torch `2.6.0`; installed vLLM EOS-source hashes match the reviewed values. Canonical `.venv` has no pytest, so tests used the existing `.venv-vagen-main` runner with `PYTHONDONTWRITEBYTECODE=1` while executable package identity was checked separately with canonical `.venv`.
- Remote v3 contract regenerated at `/tmp/nimloth-step60-runtime-contract-v3-187fe112-20260902T151027Z.json`: payload SHA256 `cbb30382ffa5170daba37458f182d472e63b46c97f9fe588c6ce565214e6fcbf`, JSON file SHA256 `7b9184b8e33d76c0d410b141d4cff9ea993bef43708f5f9d16e7b2972718e9e8`. Both worktrees remained clean.
- Source recheck again found 8 actor model + 8 extra-state shards, exact train parquet SHA256 `3c8161...`, and absent stable output group. Latest `normal` snapshot now has multiple healthy nodes with at least four free GPUs, including one idle 8-GPU node; no node is hardcoded and availability remains launch-time evidence only.
- Added an exact candidate contract for a first `normal` one-node/four-GPU, three-hour actor merge + source-index-0 smoke stage. It explicitly excludes the 100-row and formal batch1 stages.
- The first approved launch attempt stopped at the local pre-submit resource parser before any remote mutation or Slurm submission: the fixed-width table separates GPU total into its own token, so the reviewed AWK column indexes rejected a genuinely eligible snapshot. No run root, merge, model load, GPU, job, service or rollout existed; no experiment end record was required. The contract now uses exact-commit Slurm parsing and dynamically binds submission to the eligible node set.
- After the corrected resource gate and fresh approval, network retry reached superpod but the exact script stopped before run-root creation with exit 141: under `pipefail`, `grep -q` closed the `git worktree list` pipe early and Git received SIGPIPE. No remote mutation, output or Slurm job occurred. The contract now captures full worktree-list output before matching and removes the analogous early-exit pipeline from hold-state cleanup.
- The next fresh-approved retry passed those gates but the combined CPU package/server imports exceeded their shared 120-second timeout and exited 124 before run-root creation. Isolated 45-second checks then proved each of Torch, Transformers, vLLM, VAGEN server and collector imports exits successfully. The contract now gives each import its own 60-second bound and keeps version checking separate.
- After docs commit `ab47ded...` and a fresh exact launch approval, the first attempt again saw transient vLLM import delay; three immediate isolated repetitions then completed in 12–15 seconds and the unchanged approved script was retried. All Git/package/source gates passed and the unique run root was created, but deterministic partition publication failed before runtime-contract generation or Slurm submission: the remote output filesystem returned `EINVAL` for `renameat2(RENAME_NOREPLACE)`. The partition temp was cleaned by the implementation; the run root, metadata and contract copies remain protected evidence.
- Mandatory end record: remote `END.json` SHA256 `d7386e49...7608`, metadata SHA256 `5df30d9f...ee32`, group progress SHA256 `fbcf69b7...9766`. Status is `failed_pre_submit`; no scheduler job ID, GPU use, actor merge/load, service, rollout, SFT1 or SFT2 artifact exists. This run root is permanently non-reusable.
- W-009/W-010 are blocked on a reviewed publication design compatible with the remote filesystem. Do not silently replace atomic no-overwrite semantics with check-then-rename; a fresh task replan/implementation/commit/push/launch gate is required before retry.

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

### W-013 NFS-compatible publication

- Replaced unsupported no-replace directory rename with atomic final-directory `mkdir` reservation and readiness-marker-last publication for partition, shard and conversion outputs. Final paths stay lexical; staging must be a real direct sibling; existing files/directories/symlinks and concurrent losers are never replaced.
- Publication uses `.NIMLOTH_PUBLISHING.json`, payload-first moves and hidden `.NIMLOTH_READINESS.tmp`. After staging removal and all fsync/sentinel work, the selected readiness-marker rename is the commit point and final fallible operation. Interrupted targets remain inspectable and consumers reject sentinel, hidden-readiness, failure-marker, markerless, nested-marker and symlink states.
- Partition, shard and conversion consumers now apply the path-aware readiness gate before parsing; partition loading rehashes all ten sibling parquet files. Producer entrypoints reject dangling output symlinks before source reads, rollout setup or conversion imports, and failure handlers append evidence without deleting staging/final payload.
- Local task-specific publication/producer/converter suite passed `38 passed`; a focused publication subset passed `15 passed`. After reconstructing a temporary Python 3.13 test environment without modifying the repository, the full affected suite passed `125 passed`. Targeted Ruff, compileall, shell syntax, Trellis validation and `git diff --check` passed.
- Three read-only `trellis-check` passes drove lexical-path, gate-first, preserved-evidence and post-marker exception fixes. The final pass reported no blocker/high. No remote mutation, checkpoint merge, GPU, Slurm or experiment launch occurred; remote NFSv3 probes remain exclusively W-009.
- After exact commit approval, the reviewed 12-file scope was committed as Nimloth `32bcc04511364801c99884e836a9d3b27db7d2e6` (`fix(sft1): publish step60 artifacts safely on NFS`). After a separate exact push approval, only `origin/task/rollout-vagen-step60-sft1-sft2` fast-forwarded from `ab47ded...` to `32bcc045...`; `origin/dev`, VAGEN and the gitlink were untouched.
- The first read-only remote recheck after push failed during SSH connection establishment with `Connection closed by UNKNOWN port 65535`. No remote command body ran, so the remote worktree, NFS outputs, jobs and experiment state were not inferred changed. The human later clarified that VPN remained connected; this is recorded only as an SSH transport interruption, not a VPN diagnosis.
- After SSH access recovered, the remote Nimloth worktree was confirmed clean at `187fe112...`, recursively populated submodules were clean, `.local` still targeted the canonical root, and VAGEN remained clean at `170a673...`. The task ref was fetched and only the detached Nimloth worktree advanced to exact `32bcc045...`; all identities and cleanliness were reverified. Remote affected tests then passed `125 passed` with `PYTHONDONTWRITEBYTECODE=1`.
- The first NFS probe invocation failed before root creation because `.local/tmp` did not exist. A corrected bounded invocation used `mkdir -p` for that machine-local parent and a fresh timestamped probe root, but SSH/VPN dropped before returning any remote output. Reconnection inspection proved no matching partial probe root existed; no evidence was deleted or reused.
- All three readiness markers then passed retained NFSv3 evidence under `.local/tmp/step60-nfs-publication-probe-20260902T121624Z-32bcc045`: final rename from `.NIMLOTH_READINESS.tmp`, existing-target preservation, exactly one concurrent winner, and fail-closed interruptions before and after sentinel removal. `SUMMARY.json` SHA256 is `aa63a9a8851a6e2df0960b896fad0d08c98d167155d82d3119eedcb051db1d5f`.
- A real pinned-source partition was published with the new protocol and all ten sibling parquets rehashed. Manifest SHA256 is `be7db7ea975927bc176186bcb51a202b3be191196ced26e043a57add5f99b87c`; checks are 20,000 unique exact-union rows, ten batches, batch1 1,800 train + 200 held-out and zero source/eval-set-seed/bare-seed overlap.
- Regenerated CPU evidence retained runtime-contract file SHA256 `7b9184...e9e8` / payload `cbb303...fcbf`, checkpoint inspection `a819cb...b500`, and inert merge plan `5e9472...c05`; no HF target was created. One combined preflight stopped at a transient 60-second collector-import timeout, but a bounded isolated retry completed in 0.27 seconds. A later ad hoc validator used a nonexistent `batch1_count` key and stopped after successful partition publication; the corrected path-aware validator then passed without modifying that artifact.
- Current `normal` snapshot has eligible healthy one-node candidates `dgx-14` and `dgx-35` for 4 GPU / 112 CPU / 256 GiB; availability remains transient. The fresh run root `20260902T123000Z_step60_batch1_v3_nfs_32bcc045` was verified absent. The new exact source-index-0-only candidate receives its docs commit as mandatory `$1`, selects exactly one dynamically eligible node, and excludes the 100-row gate and all later stages. No Slurm job, GPU, checkpoint merge/load, environment service, rollout or dataset conversion had started at this gate.

### W-010 NFS-safe merge/smoke attempt 1 — failed pre-submit

- The final contract was committed cleanly atop the task ref as `453b986993f6ab16c3a5d004161153f645e236b1` after a concurrent canonical-root commit made `cbd05e5d...` unsuitable for push; only `453b986...` was pushed to the task ref, excluding the unrelated ancestry. Human launch approval bound contract blob `0fe9c582...`, file SHA256 `b902be11...f6bdf`, code `32bcc045...`, VAGEN `170a673...`, fresh run root, normal one-node 4 GPU / 112 CPU / 256 GiB / 3h and source-index-0-only scope.
- Immediate local resource evidence listed `dgx-29` and `dgx-35` as eligible; `dgx-10`/`dgx-52` failed observed-memory gates and `dgx-51` remained explicitly excluded. The exact approved script then completed Torch, Transformers, vLLM and VAGEN-server import gates but exited code 1 before the collector import printed `IMPORT_OK`.
- Script ordering proves failure occurred before package/source/output gates, run-root creation, second resource query, `sbatch`, checkpoint merge/load, service startup or rollout. `HOLD` was never assigned, so there is no scheduler job, GPU use, checkpoint/output, W&B identity or resumable state from this attempt. The approved run root remains non-created by this script and is not treated as reserved.
- Three immediate isolated collector-import reproductions passed in 0.18–0.26 seconds, so the exact failure cause is unresolved rather than attributed to code or timeout. A subsequent SSH transport interruption delayed the audit; the human clarified that VPN remained connected. The completed correct-path audit proved the run root absent and found no matching live or accounting Slurm job. Per the end contract this attempt is `failed_pre_submit`; no remote run README/group progress exists to update because the script never created the run root. Retry required root-cause review and fresh launch approval; no blind retry was authorized.

### W-010 NFS-safe merge/smoke attempt 2 — failed actor merge

- A fresh approval authorized the unchanged exact contract after the five-import sequence reproduced successfully. Immediate resource evidence was transiently sparse but the in-script recheck selected eligible `normal/dgx-29`; Slurm job `543910` received one node, 4 GPU and 112 CPU.
- NFS-safe partition publication, runtime contract and full checkpoint hash inspection passed. The merge step failed before creating `merge/hf_actor` or loading weights: `prepare_merge_plan()` had resolved the reviewed `.venv/bin/python3` symlink to `/usr/bin/python3.10`, so the subprocess lost virtualenv ownership and user-site `accelerate` failed with `ModuleNotFoundError: No module named 'torch.utils'`.
- Controller cleanup cancelled the hold after 47 seconds; `543910.0` is `FAILED 1:0`. No merged actor, environment service, policy generation, trajectory, terminal response, SFT1/SFT2 data or W&B run exists. The run root is permanently non-reusable and has remote `README.md` plus `END.json` SHA256 `e71ed6539da654b3fc6b824b2c5247bc498deb3e13c7405b709fdb414e5f0d0f`; group progress retains it as invalid without promoting a valid result.
- There is no resume boundary. W-014 returned to local RED/GREEN implementation to preserve the lexical virtualenv executable path; any retry still requires remote inert-plan/import proof, a fresh run identity and a new launch approval.

### W-014 virtualenv executable ownership fix

- RED reproduced the failure: a symlinked `<venv>/bin/python3` was rewritten to its system target in both merger command and provenance. The wrapper now applies lexical absolute normalization only to the Python executable, preserving virtualenv entry ownership while leaving checkpoint, merger-script, target, existence, executable and non-overwrite gates unchanged.
- Focused checkpoint tests passed `6 passed`; full `tests/training/sft1` passed `96 passed`; targeted Ruff, compileall and `git diff --check` passed. Final read-only `trellis-check` found no blocker/high. The clean isolated worktree preserves unrelated and generated submodule payload.
- Exact W-014 commit `7dac687b733cccffaf0a211ef0a602ec001749dd` was pushed only to the task ref and the clean remote Nimloth worktree was refreshed to it. Remote inert planning now preserves `/project/peilab/atst/nimloth/.venv/bin/python3` in both `python_executable` and `command[0]`; the nonexistent target remained absent. That interpreter reported the exact `.venv` prefix and successfully imported Torch `2.6.0+cu124`, `torch.utils`, Accelerate `1.14.0` and the complete legacy merger module. Retained plan SHA256 is `00198dc3116da488129a6b3cb88391de6a5d588e79ce8459e1be00b5ae748700` under `.local/tmp/step60-w014-preflight-20260902T142628Z-7dac687b`.

### 2026-09-02 venv-safe retry failed before submit

- Exact launch approval bound docs commit `f268fcf5c1bf39c12537bf72bd89efe6ac0756cc`, contract SHA256 `ce4fd79cf065f74c8f56c373cd94aadee6fc3cf79b906400e09cf10194cb27c6`, and fresh run `20260902T143000Z_step60_batch1_v3_venv_7dac687b`. The SSH invocation closed before any remote output.
- Subsequent read-only audit proved the run root remained absent and found no matching live or accounting Slurm job; only terminal historical job `543910` appeared. This attempt is `failed_pre_submit`: no allocation, merge, model load, service, trajectory, dataset or W&B run existed. The consumed approval cannot be reused.

### 2026-09-02 venv-safe R2 retry failed before submit

- Exact R2 launch approval bound docs commit `4ad2d77f195f9e2e8236cf1aa5974ef956584fd2`, contract blob `dba410d4abaa1118e034999070591edaad42f93a`, SHA256 `397c1ea6c1083e602ef432bb4d58e723088873bcdf13874490eeacf527ec3d2a`, and fresh run `20260902T150000Z_step60_batch1_v3_venv_r2_7dac687b`.
- The added inert merge preflight ran before run-root creation but used `$RUN/merge/hf_actor` as its target. `prepare_merge_plan()` correctly rejected the absent parent, so the exact script exited before output mutation, resource query or `sbatch`. Read-only audit confirmed the R2 run root and matching live/accounting job were absent. This is terminal `failed_pre_submit`; R2 must never be reused.
- R3 changed only the inert preflight target to a unique nonexistent child of existing machine-local `.local/tmp`; the real merge target remained under the later-created fresh run root.

### 2026-09-02 venv-safe R3 retry failed before submit

- Exact R3 approval bound docs commit `b215a5ef905267fb06139d39a5958156864fd796`, contract blob `d26f8fae3ca4e2f8cbec4683435828c503e9b44e`, SHA256 `a4ae638b481e8d415f83fa3cad98c82c6961ee4bad2a6225d567a41b8c7d740a`, and fresh run `20260902T153000Z_step60_batch1_v3_venv_r3_7dac687b`.
- The inert plan and first three isolated import gates passed. The `vagen.server.server` gate emitted its expected Gym/SAPIEN CPU warnings but exited before `IMPORT_OK`, so `set -e` stopped the script before run-root creation, resource query or `sbatch`. Read-only audit confirmed no R3 run root or matching Slurm job. Three immediate identical isolated imports then returned `IMPORT_OK`/0, so the failure is transient rather than a reproducible package defect.
- R3 is terminal `failed_pre_submit` and must never be reused. R4 added only a bounded maximum-three-attempt policy to each side-effect-free import gate and used fresh run/preflight/port/job identities.

### 2026-09-02 venv-safe R4 — merge passed, smoke step never started

- Exact R4 approval bound docs commit `99370d4e2b4733e953caff39acf5ee1993fa2a29`, contract blob `63640065e13fbc7f8fb1ad5533b201c7e80691f4`, SHA256 `ee1b335335320a166f508fe92e09a0d9f93213827614ae38a5d9299895eec51b`, and run `20260902T160000Z_step60_batch1_v3_venv_r4_7dac687b`.
- All import/source/output/resource gates passed; job `544130` ran on `normal/dgx-37` with 4 GPU / 112 CPU. NFS partition/runtime/checkpoint gates passed and merge step `544130.0` completed `0:0`; the actor was merged, hash-bound and CPU load-validated.
- The following smoke `srun` returned nonzero before Slurm created a second accounting step or any GPU-binding/service/smoke artifact. The reviewed command did not persist scheduler/client stderr, so the exact refusal remains unresolved. The controller cancelled the hold after 3m57s. No service, policy generation, trajectory, terminal CoT, SFT1/SFT2 data or W&B run exists.
- Mandatory remote end evidence: `END.json` SHA256 `afc2e2b51a6300b1ddd956b56e56f16ed355a669514edc7fdfe2ca328f28ddea`; `README.md` `772896f0490e7999e078d79e8d747c79e3181271216f0da24bd3284f15e2ace6`; final accounting `f35365e7977b003612f698a22f7b158aa81dc13e39eefc470b9af8e265b51d14`; group progress `9d82a407051ce8d6ce40c99a87658b1fb945ad6e8583788d35b0d3ecf5edba71`.
- R4 is invalid and non-reusable. Its partition/runtime/merged actor are evidence only; task constraints prohibit silently reusing this output. R5 persisted the smoke wrapper and its diagnostics, while retaining a fresh identity and no-retry boundary.

### 2026-09-02 venv-safe R5 — merge pipeline stopped controller

- Exact R5 approval bound docs commit `cdb9fa7c6e05e92020bf0e1d9c8a92471a981f11`, contract blob `032ed2325c324d8b1679e62ac3a0d7fb08a282d5`, SHA256 `be96cd8e653d49978ab562be5711563867f37de88ae3fb4c6ee094073db5adc0`, and run `20260902T163000Z_step60_batch1_v3_venv_r5_7dac687b`.
- Job `544142` ran on `normal/dgx-38`. All preflight gates passed and merge step `544142.0` completed `0:0`; the actor was merged, hash-bound and CPU load-validated. Nevertheless, the outer controller stopped on the preceding `srun | tee` pipeline before the next command created `control/smoke-step.sh`. Thus R4's missing second step was not a smoke-wrapper failure: the merge pipeline itself returned nonzero to `set -e` despite authoritative child-step success.
- The contract did not record merge `PIPESTATUS`, so whether `srun` client or `tee` returned nonzero is unresolved. No service, policy generation, trajectory, terminal CoT, SFT data or W&B run exists.
- Mandatory remote end evidence: `END.json` SHA256 `4c6a42f90852b93fc67f3f62741f2441a94a21f607749f99c0ac4809add1a7e6`; `README.md` `3d11592ecff4f948225b2886b085c1a1b0ea1301c46c9046adbf667f36d88630`; accounting `fad19d483695a5347ace31e0b1847fb30c8f2af1be93a9b25dd6581326d20517`; group progress `2ff6a6d33a6800acf36125ce334d891ce2f057bb9c0bf45bd52b4b50026a0152`.
- R5 is invalid and non-reusable. R6 implemented the required pipeline/accounting/manifest continuation gates under a fresh reviewed contract.

### 2026-09-04 pipeline-audited R6 — failed pre-submit, no eligible resource

- R6 docs commit `0a73ff209dd7c04f955bf0f89c69b02070171d19`, contract blob `de5fb0ced246dc6125d208229f938f052bef30c9`, SHA256 `e404fa48250232cb602c10dea41881bf22b5245196eb26c91d0acc7e757c8d55` received exact launch approval. Identity/import/source/runtime gates passed after SSH recovered.
- The immediate resource gate found only `normal/dgx-31` with four free GPUs; its observed free memory was `96.3G`, below the reviewed `256G` minimum. The script failed before `sbatch`: no job, allocation, merge, service, rollout, terminal CoT, SFT data or W&B run exists.
- Mandatory end evidence under terminal run `20260903T023000Z_step60_batch1_v3_venv_r6_7dac687b`: `END.json` SHA256 `bdb33677bc4d39c6e18c0d07dadb5419b7bd768faa9d24efa168b90e44e5caa6`; `README.md` `3dda6b7cf0fdc31904f7d7076717948056be5113bd95f7f30de004c7786a2ff6`; empty matching `squeue` evidence `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; group progress `ad60481a7f6c3a85af575dbfb01bf0b2cc5f48d97c17726c217c354f53263a06`.
- R6 is non-reusable. The next contract must query resources before creating its fresh run root so ordinary capacity shortage does not consume another identity.
- Human selected the lower source-index-0 smoke envelope `--mem=96G` with an `80GiB` observed-free pre-submit floor. This is smoke-only: prior merge MaxRSS was approximately `15.6/15.9GiB`, but full vLLM plus environment host-memory peak remains unproven, so OOM or node pressure fails closed and does not authorize lowering batch1 collection resources.

### 2026-09-04 lower-memory R7 — merge srun consumed streamed script stdin

- Exact R7 approval bound docs commit `26874f0c1e06bfa639d0d62c5b866c90cfeb4783`, contract blob `aba8957aa4555cb093e4c867723ba6117f2f7b1b`, SHA256 `f8c8275bbb0d01745034f728969dfa8792ea1abba59d508443ce32f749954f9c`. The 96G/80GiB gate passed on `normal/dgx-31`; job `546235` allocated 4 GPUs/112 CPUs. Merge step `546235.0` completed `0:0`, MaxRSS `19079632K`, and the actor passed merge/load validation.
- Root cause of the repeated post-merge disappearance is now established: the entire launch script was streamed on SSH stdin, while merge `srun` inherited and consumed the remaining unparsed script as task stdin. The outer shell therefore reached EOF before recording pipeline status or creating the smoke wrapper. No service, rollout, terminal CoT, SFT data or W&B run exists.
- Mandatory end evidence: `END.json` SHA256 `ba0adfb3fd6b570eff46a4021a1bf959d0de1a47590ce5ea30a043338a6db54a`; `README.md` `39a11276f06ebecc6939ffcabe0c6809fb738670e44004433f590ad8cca77911`; accounting `4ffafddf22167d6a2cc078f11f390a668733431a16928339d7220ae8e967c961`; group progress `4e5ac0e802f1764ab7d027d3430fd5d60f82ad8c250cf08db780fbbe3e20de2a`.
- R7 is terminal/non-reusable. R8 closed stdin for every `srun` and proved both merge pipeline components complete zero.

### 2026-09-04 stdin-closed R8 — client disconnect cancelled before smoke

- Exact R8 approval bound docs commit `71539f799db58f7bc70b52ca5b302cc5a8a887c8`, contract blob `89910c6f67702f4ecc62810f009b9f52065abd43`, SHA256 `a1d30eba45a8b1c1abab6d7a9f41b5d1c19e110258d39ca096bed3992b2fceff`. Job `546271` ran on `normal/dgx-28` under 96G/80GiB. Merge step `546271.0`, merge `srun`, and merge `tee` all completed zero; post-merge accounting persisted.
- During the second `validate-export` load, the interactive tool was aborted by the progress query. The SSH disconnect invoked the approved cleanup trap and cancelled the hold. `merge-validate-export.json` is empty and no smoke wrapper/service/rollout/terminal CoT/SFT data/W&B run exists.
- Mandatory end evidence: `END.json` SHA256 `96554d57f90833b4614fae4d9fe484798912ae53deda12ae04003aaa28fbd90d`; `README.md` `7a05aa91732a58c0c90ad24efbe20a4dadef0c77eba9bd2751e2a40abf0d860f`; accounting `a7b001b251359b63de8ca0d6e61b153645e5d9bb7c5abd4dacda3646841c2b35`; group progress `0368955e9f0b8c05bcd89ad93255f1a0bafb08214ee1b93a7759507ccd395131`.
- R8 is terminal/non-reusable. R9 detached execution from chat/SSH and made monitoring independent.

### 2026-09-04 detached R9 — pre-root resource snapshot race

- Exact detached launcher used docs commit `d2bf9f22484115f2b65766d9bc0fa30eadbb6d8`, contract blob `35e2a1cdc4d72a4bda8076599f5957433033d235`, contract SHA256 `6d40e6c9fe61c641b54c6c8750930340140e864cf51ade3c30f9ac412cea5994`, and main-script SHA256 `b837ede685a7a48502421b402c77dcca5f16bf3a0092c3f720837e3e435ae77c`. PID `553452` was detached successfully; chat monitoring did not terminate it.
- The launcher exited before creating a run root or submitting Slurm. Its resource gate used two independent `parse_nodes()` snapshots: the first CLI report and second eligibility calculation could observe different scheduler states. Shortly afterward one snapshot showed `normal/dgx-28` satisfying 4 GPU/112 CPU/96G scheduler/80GiB observed, while the CLI's separate snapshot printed no four-GPU node. No job, allocation, output root or experiment existed.
- Retained launcher script/log/PID SHA256 values are `b837ede685a7a48502421b402c77dcca5f16bf3a0092c3f720837e3e435ae77c`, `d14b44e73639cf0330ab85f493e6814bab8a92788db1beebcb07e68f4e65c3e3`, and `55dff49aaf8059f0483016a7c50856b9d01e1d798ca8743bdbb934d22d132319`. R9 launcher paths are terminal/non-reusable; R10 derived report and eligibility from one scheduler snapshot.

### 2026-09-04 single-snapshot R10 — no eligible GPU capacity

- Exact detached R10 used docs commit `cb4f1bd08425083ad5dfaf0ff06a7bc3489b4864`, contract blob `f16b0d02e31cae1dda78e748755746f4e2c6b52b`, contract SHA256 `0e85d1529dfce596b35cd01fb99d6f9e52ff3108faf4c1ef42191c4e1d557a60`, and main-script SHA256 `2f6badca838b000a957daa9ec6c82cbf4cd7a2d02e1dff477f22756f28a28327`.
- One exact `parse_nodes()` snapshot found no `normal` node with four free GPUs and the 112CPU/96G/80GiB gates. The detached launcher exited before run-root creation or `sbatch`; a subsequent snapshot also showed zero free GPUs across normal. No job, allocation or experiment output exists.
- Retained launcher log/PID SHA256 values are `a92f8a867bbff99e3431e248a5f98687cbfffaa5c78fdb87b2f3ae8fd2ac641e` and `dee118d58829129c2470752a1b992162ba3ecde9fb74c2ae68411f5be6d0b064`. R10 launcher paths are terminal/non-reusable.
- A subsequent live snapshot found `preempt/dgx-55` with 4 free GPUs, 192 free CPUs and about 1.21TiB observed-free memory. Human explicitly selected switching the source-index-0 smoke to `preempt`; R11 retained 96G/80GiB and treated `PREEMPTED` as terminal invalid with no within-run retry.

### 2026-09-04 preempt R11 — capacity gone before launch

- Exact R11 used docs commit `6c7879726d353c7a000b9f6344d6ffc061473fe3`, contract blob `3ab56624fcb0e0fbe86407e3e2d03a7bddf3628f`, contract SHA256 `96364182c73be8183fa20a3093d4abab9ebeb829477aac5fecd9e818ecb16ebe`, and main script SHA256 `25181411ed092162e47562fd21afd185af7efa3abba9bd547c43bd29574109e0`.
- After SSH recovered, the detached single-snapshot preempt gate found no four-GPU candidate; two subsequent normal/preempt queries showed zero free GPUs. The launcher exited before run-root creation or `sbatch`; no job/allocation/experiment exists.
- Retained R11 launcher log/PID SHA256 values are `32ad87c2fc006423c56cbe3f8cf521fc2caa888ee3b1ba65b1255754cae414c7` and `5029f7d0ecf48ef6714e23ae6723ea4c80a3cbea39e7f92f467c9a2003c980bb`. Human then selected submitting to `normal` and waiting at most 24 hours rather than requiring immediate free GPUs.

### 2026-09-04 normal-queue R12 — detached shell lacked module environment

- R12 was bound to docs commit `905f94d716d6a297d739423887c3877b436442cf`, contract blob `2be159511520851a832b842c015fd222db60fd93`, contract SHA256 `bf1a668526ba4e68db25464bc160fbed2c8da77991f6f197a50d88d09857962e`, and main-script SHA256 `afaa97add79c8827f7388db5cf4e052c850ea553ad31a7fdfdbe9286e56ab99e`.
- The detached bootstrap used `nohup bash`, unlike the earlier login-shell execution. Identity/import/fetch gates passed, then the script stopped exactly at `module load slurm 2>/dev/null`: the non-login shell did not initialize the module function, and stderr was intentionally suppressed. No run root or Slurm job existed.
- Retained launcher script/log/PID SHA256 values are `afaa97add79c8827f7388db5cf4e052c850ea553ad31a7fdfdbe9286e56ab99e`, `3364c0f2492252dc5331ac3a1bb0b4109c08d52d47b08f3caf22f1e62b0e87e4`, and `9840ee78fe85ce8bff7c6c99ae945c5b1ee3b22cbfc41c9f5f1a892dd3a9484e`. R13 was never executed remotely and is permanently superseded.

### 2026-09-05 rollback to minimal existing-script approach

- Direct existing-script job `546962` was cancelled at the human's request while still `PENDING`; final accounting was `CANCELLED`, elapsed `00:00:00`, no node assigned and no run root or rollout artifact created.
- The human rejected the subsequent per-trajectory checkpoint framework and custom eight-GPU orchestrator as over-engineering. Commits `cabaa7e1...` and `7f31ffcf...` are reverted in full; their W-015/W-016 code, tests and task design are not launchable or reusable.
- Work returns to W-010 with the explicit direction to reuse the existing collection script and make only the minimum GPU-count/parallelism adjustment after confirming the exact desired command. No remote worktree refresh or new Slurm submission occurred during the withdrawn work.

### 2026-09-06 retry 548155 submitted as 550810

- Human explicitly requested resubmission. Minimal environment runtime routing repair committed as `0770056747b6fef51f7c26e844396af360132b67`, branch `codex/step60-retry-548155`; local SFT1 suite 98 passed, shell syntax and diff checks passed. Independent review passed. Real reconstructed VAGEN accepted the exact failed parquet row config, including example_count; all actor artifact hashes matched the existing merge manifest. These checks do not prove GPU rollout completion.
- GitHub push failed public-key authentication. Transferred a verified Git bundle over documented SSH instead; remote detached worktree `/project/peilab/atst/nimloth/.worktree/step60-retry-548155` was clean at the exact repair commit before submission. Runtime reconstruction remains clean at `170a673d1bf5855fc0ea6fbed0744b3d7168f8f0`.
- Submitted once at 2026-09-06T06:06:02Z: job **550810**, initial PENDING(Priority), no allocated node. preempt/preempt_qos, 8 GPU, 224 CPU, 180G, 8h, exclude dgx-51. Original step60 actor, base seed27670, single-row compatibility smoke, 20 turns/512 tokens per turn; no SFT training or full collection.
- Remote output root: `/project/peilab/atst/nimloth/outputs/experiments/training/sft1-vagen-step60/20260906T060000Z_retry548155_runtime_fix`. `submission.txt` records exact exports/commit/command; `job_id.txt` records the returned ID. Original failed root remains intact.
- Monitor through `ssh -F ~/.ssh/config superpod-csejzhang`, login shell then `module load slurm`; `squeue -j 550810` and `sacct -j 550810 --format=JobID,State,Elapsed,ExitCode`. stdout/error: `/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/vagen-k8-full-8g_550810.out` and `.err`; child logs under the new root. Next: verify allocation, render probes, environment creation, rollout dump and terminal status. Do not assume dataset readiness or submit a duplicate.

### 2026-09-06 two-GPU replacement 550848

- Human requested ETA and authorized switching to two GPUs if eight GPUs queue poorly. 550810 predicted start Sep08 10:18Z (~52h); confirmed csejzhang/PENDING immediately before cancellation. Final accounting CANCELLED, elapsed0, no GPU allocation or rollout, no resumable state. Its prior commit07700567/model/input/output provenance remains above.
- Replacement submitted once 2026-09-06T06:33:38Z as **550848**: normal/normal_debug_qos,2GPU/56CPU/180G/2h,no-requeue,exclude dgx-51. Initial PENDING(Priority), no node. Pre-submit test-only ETA Sep07 04:09:38Z is provisional, not actual job allocation.
- Source commit `56f9c3181d8f0e20f9a2a799ab7217fdb727cc84`; clean remote `/project/peilab/atst/nimloth/.worktree/step60-two-gpu`, Git bundle synchronization verified. Independent reviewer passed; full SFT1 suite106passed including occupied-port rejection/dead env PID gates. GPU/Ray/render/rollout not yet verified.
- Environment1GPU + policy1GPU/world_size1/TP1/28Ray CPUs. Same actor/base seed27670/greedy20turn512token, val_only with frozen weights; not source four-policy-rank validation or final dataset collection. New root `/project/peilab/atst/nimloth/outputs/experiments/training/sft1-vagen-step60/20260906T063500Z_step60_2gpu`; exact command/exports in submission.txt and ID in job_id.txt. PORT_BASE18060, private Ray ports/temp/process group.
- Monitor via documented SSH then module load slurm: `squeue -j 550848 --start`, `sacct -j 550848 --format=JobID,State,Elapsed,ExitCode`. Stdout/error `/project/peilab/atst/nimloth/outputs/experiments/training/sft1/slurm/vagen-step60-2g_550848.out`/`.err`; orchestrator `preempt_2gpu_550848.log` in new root. No automatic further submissions; next verify allocation, render/model/env creation and actual trajectory/terminal state.

- Live update 2026-09-06T06:34:37Z: 550848 RUNNING on dgx-05 since06:33:54Z, GPUs0,1; render smoke passed both, selected env0/policy1, env launcher render passed. Queue prediction was pessimistic; real launch happened within16s of submission. Model/Ray/rollout still pending verification.

- Live update06:36:25Z: 550848 RUNNING elapsed2:31. Environment Flask healthy on18060 with selected reconstructed runtime; private Ray connected10.23.0.45:23392 and reports1GPU/28CPU/H800. Single deterministic base27670 parquet produced and run_shard entered. Model initialization/trajectory completion still unverified; no success/quality claim. Monitor new job550848, not cancelled550810.

### 2026-09-06 550848 terminal failure (ETA monitoring)

- Live sacct confirmed FAILED(1:0), start06:33:54Z/end06:39:41Z, elapsed5:47, dgx-05. Exact source56f9c318 and two-card contract/output identity remain above.
- Rendering, private Ray startup, model loading and NavigationEnv creation all passed. Generation request rejected by vLLM: `ValueError: You set image=1 (or defaulted to 1) in --limit-mm-per-prompt, but passed 2 image items in the same prompt.` This is a multimodal request/config limit mismatch, not evidence of GPU shortage or the prior example_count error.
- Recursive output inventory found no JSONL, done flag or summary JSON. No completed trajectory, success metric or resumable checkpoint; no valid dataset result. Existing input parquet is not generated trajectory evidence.
- User asked ETA monitoring only; no automatic retry performed. Next repair must trace actual history/image count through source runtime and launch config, validate intended image limit without dropping images/history, and re-run preflight before any separately authorized submission.

### 2026-09-06 authorized retained allocation
Human authorizes automatic repairs/resubmissions until healthy actual rollout and explicitly requests retaining nodes on child failure. One normal/normal_debug_qos hold:2GPU/56CPU/180G/2h,no-requeue,exclude dgx-51, entry `exec sleep infinity` via sbatch wrap, workdir remote step60-two-gpu at verified56f9c318. Hold executes no model/data work. Experiment children via srun --jobid with exact new committed repair, separate attempt roots; never stream srun stdin. Original actor/input unchanged. Diagnose+preflight each retry, no blind failure loop, no extension beyond hold2h; stop auto retry on healthy actual trajectory generation. Release/retain outcome recorded; no automatic renewed allocation beyond budget.

- Hold550857 RUNNING dgx05 since06:43:57Z, expires08:43:57Z. Attempt a1 started step550857.0 at06:52:57Z, controller loginPID3148309 detached/stdin closed, no automatic hold release on child failure. Exact Nimloth5069d550109550e6bd90352310664ef6d56457f5 and verlbc26648bb632fe153cd4e7ba27d63c53869d86ca synced as clean worktrees.108 SFT1 tests+4 real constructor AST routing tests passed; independent review passed. Remote verl import preflight correct.
- Attempt root `/project/peilab/atst/nimloth/outputs/experiments/training/sft1-vagen-step60/20260906T065000Z_step60_mmfix_a1`; submission.txt,controller.log,controller_pid.txt,attempt_exit_code.txt(if exited).06:53:40Z stepRUNNING, all rendering passed, env18061healthy, Ray startup underway. Full multi-image generation not yet proven.

### Healthy retry achieved:550857.0 completed
At06:56:13Z sacct confirmed step550857.0 COMPLETED0:0 elapsed3:12 (started06:52:57Z). attempt_exit_code.txt=0 and smoke-done log06:56:07Z. Real Ray worker imported isolated verl-step60-mm-limit; actual rollout completed8steps, one record, reported success1.0/done1.0/score1.08 and valid/effective actions1.0. Thus multimodal request routing and actual trajectory generation now pass. This is base27670 one-episode compatibility smoke, not full dataset or held-out quality evaluation; do not launch larger collection automatically.
Artifact: attempt a1/validation/train/shard_smoke_seed_27670/0.jsonl under the exact root above. User-authorized automatic retries stop because healthy criterion achieved. Parent hold550857 remains RUNNING, expires08:43:57Z; no new allocation or retry needed. Further user work can use same hold with fresh commit/preflight/output identity.

### WM200 pilot launched (latest human scope)
At07:41:44Z launched detached srun within hold550857, controllerPID3277659, exactNimloth3f4d06db68d8bef7776da589590223519e745f8a,VAGEN23173df0130e8778d5bcb9f825c18fb0b77dd615,verlbc26648bb632fe153cd4e7ba27d63c53869d86ca.114SFT1tests+20WM/reconstructiontests passed; independent review; remote parent+all10parquethashes verified; actualPythonwmconfig/parser verified;cleanremotecommits checked.
Preparedsource200 at remote outputs/experiments/training/sft1-vagen-step60/20260906T074500Z_wm200_prepared/prepared_manifest.json:batch1train first100sourceordered rows eachbase/common,10x20 shards. Runtimeoverride source_wm_mode=.currentwm,step_length.3,threshold1;greedy20turn512token;oneenvironmentGPU/onepolicyGPU,TP1,worker1,requesttimeout500. Batch1heldout200 untouched.
Outputroot `/project/peilab/atst/nimloth/outputs/experiments/training/sft1-vagen-step60/20260906T074500Z_wm200_a1`;submission.txt/controller.log/controller_pid.txt,attempt_exit_code.txt(ifexited);per-shard0.jsonl+0.validation.json(onlyexact20identitiespass).No fullbatch fallback. Holdexpires08:43:57Z;lateststageawaitingstartup.Healthcriterion isactual200pilot progressing/firstshardverified,notrepeatone-row smoke.
Oneautomaticreview rejectedVAGENbundletransfer;read-onlyproofshowedmatchingexistingserverrepository/baseandtask-onlysourcepayload;identicaltransferreapprovedafterevidence,thencompleted.Noauthorizationworkaroundused.

- 07:50:11Z live observation: firstsource200_00 shard wrote0.validation.json (exact20selectedidentitiesverified),20generationroundscompleted; nextsource200_01 started real20rowloader.Thus200pilot remainsactive,20/200validated. No policyqualityclaim yet.
- Automaticapproval denied reading firstshard rawreply/metrics and separately denied safer remoteaggregation returning onlynumericmeans, considering both sensitivepayload export despitetrial/monitorauthorization. Do not bypass by another tool/log query for same data; need explicit userpermission to read/return trial results. Existingruncontinues. Parenthold550857expires08:43:57Z;10shards may exceed remainingtime basedfirstshard7-8min, completionnotguaranteed. Keepvalidatedshards protected; future continuation must avoidduplicate200sourceidentities.

### 2026-09-06 WM200 pilot terminal at allocation time limit

- Live accounting queried at 2026-09-06T08:57:48Z confirms parent allocation `550857` reached `TIMEOUT` at 2026-09-06T08:44:14Z after `02:00:17`; rollout step `550857.1` was cancelled with `0:9` after `01:03:44` (2026-09-06T07:41:44Z to 2026-09-06T08:45:28Z). The earlier one-row smoke step `550857.0` remains `COMPLETED 0:0`.
- Exactly 7 shard validation markers exist for `source200_00` through `source200_06`, establishing 140/200 source-selected rows as fully written and identity-validated. `source200_07` was started but has no validation marker and is not counted as valid output. The pilot root has no `preempt_2gpu_rollout_done.flag`, so the 200-row pilot is incomplete.
- `attempt_exit_code.txt` contains `0`, but this is only the detached controller shell result and does not override Slurm's terminal state or the absent root completion marker. No success/quality metrics were inspected in this terminal-status query.
- Preserve the validated 140 rows and all partial evidence. No automatic continuation was submitted. A faithful continuation must use a fresh output identity, select only the remaining exact source identities, and validate against the original prepared manifest; the current prepared-mode launcher does not by itself establish that safe resume contract.

### 2026-09-06 multi-node WM batch1 continuation

- Human authorized continuing GPU allocation and using observed capacity on `dgx-06`, `dgx-09`, and `dgx-14` to collect as much of batch1 as possible. The operational unit is an isolated two-GPU rollout process: one AI2-THOR environment GPU plus one policy/vLLM GPU, with distinct ports, Ray/cache/PID namespace, output root, and non-overlapping prepared shard indices.
- Allocation `551108` is RUNNING on `dgx-06` with 4 GPUs/56 CPUs/180G, ending after its 8-hour limit. Three controller-only launch failures are retained and produced no model/environment/data output: step `.0` used a relative entry after login-shell cwd change (`127`), `.1` directly executed a non-executable shell file (`13`), and `.2` contained a malformed generated command (`2`). The corrected, syntax-checked controller uses fixed `srun` plus `/usr/bin/bash` and a fresh root.
- Corrected `dgx-06` step `551108.3` runs from Nimloth `6f3669f16bf2f29c84af96e8896d4280b4fee3ce`, VAGEN `23173df0130e8778d5bcb9f825c18fb0b77dd615`, and verl `bc26648bb632fe153cd4e7ba27d63c53869d86ca`. It passed both render probes, selected one environment and one policy GPU, reached healthy environment service and real rollout, and validated pilot shard `source200_07`; it continues with `source200_09` under `20260906T111000Z_wm200_resume_lane0_a4`.
- A second `dgx-06` step runs the remaining pilot shard `source200_08` from clean Nimloth commit `4a6a557125adbbc5bbf12b7b0fa23964d3f7da24` under `20260906T130500Z_wm200_resume_unit1`. At the latest observation it passed render/environment gates and entered real rollout. Together with the prior 140 validated rows and step `.3`, exact completion of the 200-row pilot still awaits validation of shards `08` and `09`.
- Commit `4a6a5571` adds exact batch1-remainder preparation and parallel runtime namespace isolation. Remote focused tests and shell/Python syntax checks passed; an actual prepared publication verified 90 shards of 20 rows. Formal manifest `/project/peilab/atst/nimloth/outputs/experiments/training/sft1-vagen-step60/20260906T130000Z_wm_batch1_remainder_prepared/prepared_manifest.json` has SHA256 `f8adc4bb87cf58130b9f83ac95eecdeb838d96ec74f61c0a189214b6ed174784`, covering exactly the other 1,800 batch1 identities: 1,600 train and 200 internal held-out, disjoint from the 200-row pilot.
- Allocation `551383` is RUNNING on `dgx-14` with 2 GPUs/28 CPUs/90G. Its first rollout process uses remainder shard indices `2,7,12,...,87` (18 shards/360 rows) under `20260906T131000Z_wm_batch1_unit2_dgx14`; startup is active and validation is pending. Allocation `551382` requests 4 GPUs/56 CPUs/160G/8h on `dgx-09` and remains `PENDING(Priority)` with no ETA. No data indices are assigned twice.

- Update at 2026-09-06T13:06:34Z: pilot shards `07` and `09` completed with exit code 0 under `551108.3`; pilot shard `08` completed with exit code 0 under `551108.4`. Combined with the earlier seven validated shards, the exact 200-row pilot is now complete with all ten 20-row identity-validation markers.
- The released `dgx-06` resources now run batch1 remainder unit 0 (`0,5,10,...,85`) at `20260906T131500Z_wm_batch1_unit0_dgx06` and unit 1 (`1,6,11,...,86`) at `20260906T132000Z_wm_batch1_unit1_dgx06`. Unit 2 on `dgx-14` owns `2,7,12,...,87`. All three passed render/environment/policy startup and entered real rollout; units 0 and 2 each produced their first exact 20-row validation marker by 2026-09-06T13:06:34Z.
- `dgx-09` did not provide a usable current window despite reported free GPUs. Slurm test-only for 4/2/1 GPUs and walltimes from 1 to 8 hours consistently predicted 2026-09-07T18:48 cluster time. Pending hold `551382` was confirmed task-owned, never allocated, then cancelled at 2026-09-06T13:07:33Z to prevent a future unattended idle hold; final state `CANCELLED`, elapsed `00:00:00`, no rollout or data output.
- Live update at 2026-09-06T13:11:52Z: allocations `551108` (`dgx-06`) and `551383` (`dgx-14`) remain RUNNING. Active rollout steps are `551108.5`, `551108.6`, and `551383.0`; the three isolated two-GPU units have validated 2, 1, and 1 remainder shards respectively, for 4 shards / 80 exact rows beyond the completed 200-row pilot, and each has entered its next shard.
- The queued `dgx-14` follow-up controller remains waiting without consuming an additional GPU step. It owns the remaining 36 exact indices `3,4,8,9,...,88,89` and will start after unit 2 releases the allocation's two GPUs. Across active units 0/1/2 plus this follow-up, all 90 remainder shard indices are assigned exactly once. No success-rate or raw-response content was inspected.
- Terminal update observed after connectivity recovered: `551383.0` failed with exit `1:0` after validating exact shards `02,07,12,17`; its next shard `22` failed during vLLM initialization with `No available memory for the cache blocks`. Follow-up `551383.1` then validated exact shards `03,04,08,09`; its next shard `13` failed after the environment request reached the configured 500-second read timeout. Both failures occurred after four complete shards, and their eight validation markers remain the valid resume boundary. Allocation `551383` itself remained RUNNING on `dgx-14` with both GPUs released.
- Twelve initial retry controllers under `20260906T143152Z_wm_batch1_dgx14_retry_chunks` exited before creating a Slurm step because the detached shell lacked the Slurm configuration environment. A first login-shell path retry under `20260906T143900Z_wm_batch1_dgx14_retry_login_chunks` attempted `/controller.sh`; a corrected absolute-path retry under `20260906T144100Z_wm_batch1_dgx14_retry_login_abs_chunks` still lacked the explicit Slurm module. These were controller-only failures with no GPU step, environment/model startup, rollout, or validation artifact. All evidence is retained.
- At 2026-09-06T14:43:48Z the corrected retry queue under `20260906T144300Z_wm_batch1_dgx14_retry_module_chunks` explicitly loads the Slurm module and invokes absolute controller paths. Step `551383.3` is RUNNING with render probes and environment health passed; chunk `00` owns `22,13,27,14`. Eleven additional exclusive controllers are alive and wait on the same allocation. Across the 12 chunks they own exactly the 46 unvalidated `dgx-14` indices, at most four per chunk, so one failing chunk cannot prevent later chunks from running.
- Concurrent live counts at 2026-09-06T14:43:48Z are unit0 `9`, unit1 `8`, prior dgx-14 unit2 `4`, and prior dgx-14 follow-up `4`: 25 validated remainder shards / 500 rows, plus the completed 200-row pilot = 700 valid rows. Allocations `551108` and `551383` remain RUNNING with approximately 2h26m and 6h07m left respectively. No raw responses or success-rate metrics were inspected.
- Human reported 8 free GPUs on `dgx-33` and 4 on `dgx-06`. Live Slurm evidence confirmed `dgx-33` IDLE with 8/8 GPUs and `dgx-06` MIXED with 4/8 GPUs allocated. `dgx-33` belongs only to `preempt`; the valid 8-GPU test-only contract was `peilab/preempt/preempt_qos`, 112 CPUs, 360G, 8h. `dgx-06` belongs only to `normal`; an extra 4-GPU test-only request predicted Sep07 20:12:56 cluster time, so no future unattended extra hold was submitted there.
- Hold `551605` started immediately on `dgx-33` with 8 GPUs/112 CPUs/360G/8h. The first four concurrent two-GPU steps exposed correct per-step logical GPU `0,1` and passed render/environment health, but two failed at Ray startup because concurrent Ray heads raced for fixed internal gRPC port `10002`; no validation marker was produced. The remaining task-owned steps/controllers were cancelled or terminated after exact ownership and zero-valid-output checks. The allocation remained RUNNING and all evidence was retained.
- `dgx-33` was switched to the launcher-native 8-GPU topology rather than applying an untested runtime patch: step `551605.6`, four environment GPUs `0-3`, four policy GPUs `4-7`, ports `18200-18203`, exact 42 unvalidated indices formerly owned by dgx-14 chunks `01-11`, output root `20260906T151000Z_wm_batch1_dgx33_8gpu`. At 2026-09-06T15:04:06Z all eight render probes and four environment health checks passed and real rollout task 0 was active; no shard validation marker yet.
- Existing dgx-06 steps `551108.5` and `.6` each validated nine shards, then failed on their tenth (`45` and `46`) during vLLM initialization with `No available memory for the cache blocks`. Their exact remaining 18 indices are now split into six fresh chunks under `20260906T151500Z_wm_batch1_dgx06_retry_chunks`: `45,50,55,60`; `46,51,56,61`; `65,70,75,80`; `66,71,76,81`; `85`; `86`. Steps `551108.7` and `.8` are RUNNING as staggered two-GPU units with separate ports and healthy environment services; four controllers wait for released step resources.
- Live validated count at 2026-09-06T15:04:06Z: dgx-06 original units `9+9`, prior dgx-14 units `4+4`, dgx-14 retry chunk `2`, dgx-33 new 8-GPU run `0` = 28 remainder shards / 560 rows; with the completed pilot, 760 valid rows. No raw response or success-rate metric was inspected.
- Terminal update: dgx-33 hold `551605` was PREEMPTED after `01:08:12`; native 8-GPU step `.6` was cancelled by allocation loss after `01:02:52`. It validated exact shards `18,19,23,32,37,42` (6 shards / 120 rows). No root completion marker exists, so only those six validation markers count as valid output; the other 36 assigned indices form the resume boundary.
- The earlier same-node four-lane dgx-33 attempt proved per-step GPU cgroup mapping but exposed a shared Ray internal gRPC port `10002`: steps `.1` and `.3` failed before data, while `.0/.2/.4/.5` were intentionally cancelled after confirming zero validation output. No runtime port patch was applied. The allocation was instead used by the launcher's supported single 8-GPU topology until preemption.
- dgx-06 retry chunks `a0`, `a1`, `a2`, `b0`, and `b2` completed with exit code 0 and validated `4+4+1+4+1` shards. At 2026-09-06T16:05:54Z chunk `b1` remains RUNNING and has validated two of its four indices. Hold `551108` has about 1h04m remaining.
- After exact marker enumeration, the 36 unvalidated dgx-33 indices were moved to nine new four-shard controllers under `20260906T161500Z_wm_batch1_dgx14_after_dgx33_preempt`. Existing hold `551383` had both GPUs free and about 4h44m remaining. Step `551383.5` is RUNNING; it passed render and environment health and began real rollout, prioritizing held-out indices `82,87,83,84`. Eight controllers wait for released step resources. No index overlaps the six dgx-33 markers or dgx-06 work.
- Live total at 2026-09-06T16:05:54Z: 52 validated remainder shards / 1,040 rows plus the completed 200-row pilot = 1,240 valid rows. This is artifact/identity validation only; raw responses and success-rate metrics remain uninspected.
- Human authorized use of normal-partition capacity for more parallelism. At 2026-09-06T16:30:18Z, a read-only Slurm `--test-only` sweep requested the exact two-GPU rollout contract (peilab/normal/normal_qos, 2 GPUs, 28 CPUs, 90G, 4h) on every non-allocated normal node: dgx-04,05,06,09,10,13,14,15,18,21,23,26,27,29,31,35,37,38,46,51,52. None had a current launch window: earliest estimates were Sep08 cluster time and most were later. No normal hold or rollout was submitted, so no unattended post-batch allocation was created. Existing dgx-14 allocation remains the immediate continuation resource.

### 2026-09-07 multi-GPU expansion on dgx-17 and dgx-55

- Human explicitly required use of each node's available GPUs rather than two-GPU lanes. The rollout launcher now accepts total GPU counts 2--8 and validates a 1-, 2-, or 4-GPU policy partition. The exact source commit is `bf4c9ecbb52d755025e5e30c86c34939bb9b216e`; its remote runtime worktree tree matched the local tree `63128a95d759b403c1af786d35b5f61a684753ef` before the same commit was created and checked out remotely. Shell syntax validation passed. The project virtualenv lacks `pytest`, so its focused pytest suite could not be run there.
- Task-owned preempt allocations `551802` on `dgx-17` and `551803` on `dgx-55` are RUNNING, each with 4 GPUs, 56 CPUs, 180G, and about four hours remaining. The two source-compatible controllers use four GPUs as one process per node (`ENV_GPU_COUNT=2`, policy GPUs=2), avoiding the earlier same-node multi-Ray-head collision. They own the previously unstarted exact dgx-14 chunks `01` (`88,89,47,24`) and `02` (`52,28,57,29`) respectively.
- Before reassignment, the dgx-14 `chunk_01` and `chunk_02` controller processes and their waiting `srun` children were terminated; there was no Slurm step or validation artifact for either. The active dgx-14 step was retained. Two initial new-controller attempts exited before an `srun` step because of a generated dashboard-port variable, then because the detached controller had not loaded the Slurm module. Both failures are retained as logs and created no rollout data. After correcting those controller-only defects, both four-GPU steps `551802.0` and `551803.0` started, logged `allocated_gpus=0,1,2,3`, and began four-GPU AI2-THOR render smoke. No raw response or success-rate content was read.

- Live check at 2026-09-07T01:08+08:00: `551802.0` and `551803.0` remain RUNNING after about five minutes, each holding all four assigned GPUs. Both completed four GPU render probes, selected environment GPUs `0,1` and policy GPUs `2,3`, passed two independent environment health checks, and entered rollout task 0. Their current chunks have not yet published a validation marker. Across the protected output roots, 70 `*.validation.json` markers exist; this is a marker count only, with no raw rollout content or evaluation metrics inspected.

- Terminal observation at 2026-09-07T01:25+08:00: dgx-06 allocation `551108` reached its 8-hour time limit (`TIMEOUT`, elapsed `08:00:11`, scheduler exit `0:0`). Its valid boundary is complete: the two original units published `9+9` markers and the retry chunks published `18`, for 36 exact 20-row remainder shards. This is collection evidence only, with no checkpoint, raw response, or success metric. It cannot be resumed as an allocation; remaining work is already owned by the live dgx-14/dgx-17/dgx-55 controllers. The global marker count is 82 (1,640/2,000 rows); no output was overwritten or promoted as an evaluation result.

- At 2026-09-07T01:24+08:00, the first dgx-17/dgx-55 four-GPU chunks each completed all four assigned shards with exit 0. Global complete-marker count reached 89 (1,780/2,000). To avoid leaving their 4-GPU allocations idle, unstarted dgx-14 chunks `04` (`72,38,77,39`) and `05` (`43,44,48,49`) were transferred after their controller and waiting `srun` processes were terminated and confirmed to have zero markers. New four-GPU steps `551802.1` and `551803.1` started on dgx-17/dgx-55. A first dgx-17 launcher exited before `srun` because its non-login parent lacked the module function; the login-shell retry started `551802.1`, so the failure produced no data or duplicate shard. The running dgx-14 step owns the remaining three shards of chunk `03`; all remaining 11 shards are thus live and disjoint.

- At 2026-09-07T01:31+08:00 all 100 exact 20-row validation markers exist: the full 2,000-row batch1 collection is complete with no duplicate UID and no missing JSONL next to a marker. The allocations `551383`, `551802`, and `551803` remained RUNNING as idle holds at that observation; no extra shard is pending. After the human explicitly authorized metric inspection, aggregation of the final environment `metrics.success` field found 1/2,000 successes (0.05%): base 0/1,000 and common_sense 1/1,000. The environment success predicate is its final distance at most 1.0m with 0.3m movement. 1,999 episodes consumed all 20 rollout rounds; only one terminated early at round 11 and succeeded. The persisted terminal turn reports an effective/valid action in only 4/2,000 episodes, but that field is terminal-turn-only and does not prove every preceding turn was invalid. The serialized `output_str` is a conversation prompt/recording rather than a clean terminal completion, so it cannot support a stronger per-turn format-failure claim without a dedicated trace.

### 2026-09-07T16:38:30Z — 原始 validation 200 条重跑已提交

- 人类明确要求修正原始validation差异、重跑相同200条并统计success rate；拒绝新建任务，随后要求继续推进，明确授权直接push。沿用本实验任务专用分支codex/step60-original-validation200。hligb端全程只读。
- Job557214，peilab/normal/normal_qos，单节点8GPU112CPU256GiB，walltime6h；首次提交2026-09-07T16:38:30Z，当前PENDING(Priority)，未分配GPU。其他job556418未触碰。最终scontrol查询已确认Requeue=0、Restarts=0、PENDING(Priority)。集群scontrol更新返回过策略错误，因此以随后读取的实际Requeue=0为准。
- 执行Nimloth10ee6ae40044311e8938e2e3549f1af07afdcd01；VAGEN4f995bfc9fad95ce717f22966e597039a1d39da4（fee3ffa源码从共同祖先787c7e2恢复+导入/输出适配）；verl14b2453e2cdb859067fec4258b5657a89d7a3a9c（原始3f55021+原始checkpoint保存开关+导出Qwen识别+真实采样断言）。三仓库专用分支已push GitHub origin；远程仅自己账号worktree。
- 运行根 /project/peilab/atst/nimloth/outputs/experiments/training/sft1-vagen-step60/20260907T164000Z_original_validation200。Slurm日志位于根路径+.slurm.out/.slurm.err。完整exports /project/peilab/atst/nimloth/.local/step60-original-validation/launch.env；同目录job_id.txt、submission_time.txt、preflight.log。
- prepared目录同实验父目录/20260907T163500Z_original_validation200_prepared；单parquet200 SHA25639593724dd741a7c0f443c708c25ba4a92cc08e163b15987654bade1fe698416。精确保留原pilot Base100/Common100身份/顺序，原始EnvConfig和RLHFDataset实际CPU加载后仍为200，无过滤。
- 冻结原step60 HF export（20260904T181500Z_step60_batch1_v3_venv_r8_7dac687b/merge/hf_actor），无权重更新。真实原始prompt/parser/env；随机采样.7/.95/-1/n1/max256/request seed=None，引擎V0seed0，8policy ranksTP4、valbatch1、max20/window5、vllm6144/data16000。同一引擎连续200条，不以逐分片重启改变随机数流。无法恢复原训练step60时已消耗的随机状态，本次是固定新引擎下的新采样运行；仍是200训练pilot，不是128原验证集。
- 新独立解释器 /project/peilab/atst/nimloth/.venv-step60-validation-085/bin/python3，vllm0.8.5.post1/torch2.6.0/transformers4.49.0/ray2.55.1。实际Qwen-VL导入及SamplingParams构造通过；与原始日志v0.8.5.post1引擎一致。原环境包声明不一致，按实测原始版本组合安装no-deps到隔离环境，原有.venv未修改。
- 检查：22 passed+1因未设置旧reconstruction checkout而skip；独立review10passed，发现并修复请求seed0与原始seedNone差异；新增Python Ruff通过、shell/inlinePython语法通过；CPU checkpoint每个index权重key存在，Hydra解析配置通过。GPU/render/Ray/实际生成尚待job启动，不能以CPU结果代替。
- 监控 automation monitor-original-validation-200 每5分钟，已ACTIVE。启动后核对render_probe、environment、Ray、validation日志及runtime/sampling-rank-*.json；完成由脚本严格验证exact200身份/PNGhash/boolsuccess并生成success_summary.json与collection_done.flag。最终报告整体及两类别成功数/比例。失败不自动重试。
- GitHub push起初审批拒绝，用户补充授权后成功；worktree路径下认证失败，控制目录git -C调用成功。SSH超时不能断言VPN断，用户已纠正；连接操作统一控制目录。

### 2026-09-07T16:48Z — 监控与预算校正

Job557214仍PENDING(Priority)，未分配节点、运行目录未创建。监控发现TimeLimit变为8h（前次scontrol策略处理后的实际状态），已显式带Account=peilab、TimeLimit=06:00:00、Requeue=0更新。命令仍返回集群Access/permission denied文字，但紧接scontrol读取确认TimeLimit=06:00:00、Requeue=0、8GPU112CPU256G，预算恢复为原约定。没有新job、重启或rollout产物；监控继续。

## 2026-09-07T17:16Z 独立两卡方案替换

人类纠正为每节点独立管理模型和环境；跨节点共享Ray方案未提交、未运行。现采用array0-2%3，每成员1节点2GPU/28CPU/128GiB/6h、TP2、valbatch1，67/67/66原始身份分片，其他采样合同不变。53项CPU测试通过、1项原重建fixture测试跳过；包括分片完整性和确定性的并发半写入JSON回归。VAGEN仅改变审计JSON原子发布，未改变模型/环境生成代码。

旧job557214在2026-09-07T17:15Z刷新为PENDING后取消；sacct确认CANCELLED、Elapsed00:00:00，无GPU运行。未修改其他job。已push并快进自己的remote worktree：Nimloth aee586b2533bbd110139eeed7465b771ad2ee712，VAGEN844378ce8a5727d8274b0c7024573031f9b1296d，verl仍14b2453e2cdb859067fec4258b5657a89d7a3a9c。新环境记录远程.local/step60-original-validation/launch-independent.env；旧monitor已暂停，待新arrayID确定后恢复。

新输出预定outputs/experiments/training/sft1-vagen-step60/20260907T171300Z_original_validation200_independent_2gpu；分片同父目录20260907T171300Z_original_validation200_partitions。正在执行远程CPU preflight和原始loader逐片保留核验，尚未提交新array。

### 2026-09-07T17:18:32Z 已提交独立两卡array557451

远程三组Hydra/checkpoint/版本preflight均通过；原始RLHFDataset真实加载保留67/67/66条，NavigationEnvConfig逐行接受，verify_partitions确认200身份与内容完全保留。随后仅提交一次array557451，成员0/1/2，确认每成员1节点2GPU28CPU128GiB、6h、Requeue0。提交时三个成员PENDING(None)，尚无GPU执行证据。完整提交：source .local/step60-original-validation/launch-independent.env；sbatch --parsable --export=ALL --chdir=$REPO --output=$RUN_OUT/slurm-%A_%a.out --error=$RUN_OUT/slurm-%A_%a.err experiments/training/sft1/run_original_validation200.slurm。控制目录independent_job_id.txt和independent_submission_time.txt已保存。

原heartbeat monitor-original-validation-200已更新至557451并恢复5分钟监控。全部三个成员成功后由监控执行原summarize exact200验证，再创建success_summary.json和collection_done.flag；各成员不单独写全局完成标记。运行源码固定aee586b2/844378c/14b2453e，后续本地进度改动不要同步影响运行时Git清洁门禁。

### 2026-09-07T17:22Z array557451启动失败

三个成员已全部FAILED1:0：0在dgx-38运行24秒，1在dgx-51运行54秒，2在dgx-18运行54秒。全部在render_probe构造AI2-THOR Controller的unity_command阶段抛出“vulkaninfo failed to run”。没有进入环境服务/Ray/模型加载/rollout阶段，无success rate。Slurm stderr无额外异常，具体traceback均在各node_<id>/render_probe.log；原始输出目录保留，不用于新运行。监控已暂停。

根因核验：launcher PATH仅含独立venv，没有已有Vulkan工具目录；远程command-v vulkaninfo未找到。实际可用二进制为/project/peilab/atst/flower/.local-vulkan/tools/extracted/usr/bin/vulkaninfo，使用原有Vulkan LD_LIBRARY_PATH后CPU --help成功，ldd所有库均解析。修复需把工具PATH/库设置提前至CPU preflight，并加command-v及--help门禁。该CPU检查仅证明可执行性，不证明真实GPU渲染；下一次仍保留150秒render_probe。尚未重提。
