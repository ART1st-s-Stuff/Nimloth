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
- R5 is invalid and non-reusable. Before any retry, the merge invocation must capture both pipeline component statuses and use completed scheduler accounting plus full merge-manifest revalidation as its continuation gate; a fresh run and approval remain mandatory.
