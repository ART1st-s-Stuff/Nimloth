# Implementation plan — VAGEN step60 batch1 rollout datasets

## Ordered work items

- [x] [W-001] **Add RED tests for deterministic source partitioning and source-protocol conversion**
  - Assert ten 2,000-row batches, category balance, disjoint source indices and exact union.
  - Assert batch1's shared-seed rule yields 1,800 train + 200 internal held-out rows with zero bare-seed overlap.
  - Assert `<answer>` parsing preserves real CoT and maps only the executed source action.
  - Assert converted SFT1/SFT2 prompts and responses are K16-compatible while verbatim source prompt/chat text and hashes remain available in the audit view.
  - Assert terminal response is excluded from SFT1 supervision and LLM-backbone labels.
  - Assert terminal draft action is audited but creates no executed action or transition.
  - Assert incomplete/non-empty shards without a valid completion manifest are rejected.

- [x] [W-002] **Implement deterministic batch manifests and strict source-row validation**
  - Add a canonical SFT1 experiment entrypoint that partitions the pinned parquet by category-local ordinal.
  - Persist source/global indices, keys, hashes, deterministic 1,800/200 split labels and all coverage/overlap checks.
  - Fail on source SHA/count/category/order drift, train/held-out seed overlap or source test overlap misclassification.

- [x] [W-003] **Implement source VAGEN step60 checkpoint merge and preflight tooling**
  - Validate all world-size-8 actor shards and tokenizer/config files.
  - Wrap the exact compatible VERL legacy merger in a non-overwriting entrypoint.
  - Validate merged HF architecture/tokenizer/weights and write a provenance manifest.
  - Do not load critic/optimizer or modify the source checkpoint.

- [x] [W-004] **Implement source-protocol rollout collection with terminal generation**
  - Add explicit source-row episode specs instead of synthetic sequential seeds.
  - Add source `grounding_worldmodeling` environment/profile validation against the archived golden transcript.
  - Generate real source CoT/action responses with step60 and execute only ordinary-turn actions.
  - Save every observation image, including the terminal image, plus the verbatim source-rendered prompt and complete real chat transcript.
  - Generate one terminal CoT+draft action, persist audit evidence, and prove no environment step follows it.
  - Persist bounded atomic shards and completion manifests; support only verified complete-shard resume.

- [x] [W-005] **Implement strict SFT1 and SFT2 dataset conversion**
  - Emit SFT1 `train_all`, `train_success` and `heldout_all` with K16-compatible prompts and only executed-action assistant turns.
  - Emit separate train/held-out current K16 `nimloth_trajectory_v1` records with `T+1` observations/images, `T` actions/responses and terminal prefix.
  - Preserve an immutable source-audit view containing the verbatim prompt/full chat/terminal response; bind source and converted views with hashes and a conversion version.
  - Record aggregate reward provenance honestly; do not invent step rewards/log-probs.
  - Write hashes, before/after counts, rejection sidecars and source/checkpoint/batch lineage.

- [x] [W-006] **Run local full-scope quality checks and prepare the exact launch contract**
  - Run targeted tests, affected rollout/navigation tests, compile/shell/config checks and `git diff --check`.
  - Review every changed file against PRD/design and selected known errors.
  - Record exact code commit/worktree, commands, output paths, W&B identity if used, resume and cancellation procedures.
  - Stop for implementation/commit approvals required by workflow; do not launch from dirty or uncommitted code.

- [x] [W-007] **Add RED tests for the evidence-backed reconstructed runtime contract**
  - In an isolated VAGEN worktree based on exact `3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a`, add failing golden tests for the three archived prompt hashes, compact strict parser/action order, legacy batch API, 0.5 m dynamics, 1.5 m threshold, 10.0 success reward, 0.02 format reward and -0.2 invalid-action penalty.
  - Add a deterministic Nimloth evidence extractor test/fixture contract for hash-pinned W&B JSON: exact UTF-8 JSON decoding, `columns`→row mapping, Qwen role-boundary regex, first-field normalization patterns, newline/`rstrip()` rules and prompt/reward output hashes. The extractor persists no assistant CoT.
  - Add failing Nimloth tests requiring reconstruction base/patch/diff/tree/manifest identity, one non-merge commit, clean actual runtime HEAD and independently approved literals; prove that `fee3ffac...`, `44be18c` or arbitrary metadata relabeling cannot bypass validation.
  - VAGEN tests cover valid, invalid-format/action, too-many-action, physical-action-failure and success behavior. Nimloth tests own ordinary/terminal finish reasons, EOS/length/parser failure, whole-linked-trajectory exclusion and proof that no terminal environment step occurs; no test generates fixed CoT.
  - Add separate RED cases rejecting every v1 reconstruction-consumed surface: runtime contract, raw row, shard manifest, COMPLETE marker, conversion manifest, rejection envelope and SFT1/SFT2 `source_audit.contract_version`. Add positive controls proving partition manifest v1 and HF merge manifest v1 remain intentionally supported.

- [x] [W-008] **Implement and verify the isolated VAGEN step60 reconstruction**
  - Add one isolated `step60_source_reconstruction` prompt/parser/environment mode on the VAGEN reconstruction branch; preserve every existing mode and the legacy Flask batch API.
  - Implement `experiments/training/sft1/extract_vagen_step60_evidence.py` and schema `vagen_step60_reconstruction_evidence_v1`; atomically fail on an existing output, extract only source prompt/config/reward-class evidence from the hash-pinned archived W&B/log assets, persist golden input/output hashes and provenance, and implement no behavior absent from reviewed evidence. Write the reviewed Nimloth fixture path, then copy it byte-identically with no-overwrite/SHA checks into the isolated VAGEN evidence path; no runtime depends on the private W&B path.
  - Update the Nimloth checkpoint constants, collector, raw/shard format versions, conversion/source-audit fields, validators, tests and README to identify the unavailable source commit separately from reconstruction base, approved patch HEAD/tree/diff and manifest hash. Remove the overloaded `source_runtime_commit` meaning through an explicit versioned migration; no legacy record is silently reinterpreted.
  - Validate clean actual Git state, exact single-parent patch (`HEAD^=3003c2e...`), approved HEAD/tree literals and SHA256 of the canonical binary/full-index diff. Manifest values are recomputed and compared against independently approved literals.
  - Fix reconstruction reward provenance to `step_rewards`; persist and validate generated token IDs plus `finish_reason`/`stop_reason`, package/tokenizer/config identities and the source-vLLM EOS contract. Any terminal-aggregate fallback or generation-boundary change returns to planning.
  - Run VAGEN and Nimloth focused tests, then full affected tests; stop and replan if a runtime semantic remains unverified.

- [x] [W-012] **Adapt and revalidate the human-selected vLLM 0.8.2 reconstruction runtime**
  - Add RED tests for explicit v3 contracts and two separate runtime fields: `source_generation_package_evidence` (`0.8.5.post1/4.49.0/2.6.0` plus W&B requirements provenance) and `executable_generation_packages` (`0.8.2/4.49.0/2.6.0`). Require policy runtime actual versions to equal the executable field and propagate/revalidate both through raw/shard/source-audit/conversion.
  - Bump every reconstruction-consumed v2 format to v3 before first real use; reject v1/v2 and any missing, single or overloaded package-triplet record while retaining partition/HF-merge v1 intentionally.
  - Update the collector package gate to require executable vLLM `0.8.2` while retaining source `0.8.5.post1` as non-executable evidence; do not weaken EOS/custom-stop/tokenizer/config checks.
  - Run local affected suites and record the already inspected canonical `.venv` vLLM `CompletionOutput`/`StopChecker` 0.8.2 source hashes plus package-drift limitation. Remote CPU execution waits until W-009 has committed, pushed and refreshed the detached worktree.
  - Obtain fresh implementation approval before code changes; W-009 commit/push gates remain separate.

- [x] [W-013] **Implement NFS-compatible reserved-directory marker-last publication**
  - Add RED tests for a real direct-sibling staging directory, lexical dangling symlink plus existing file/symlink/directory preservation, two concurrent publishers, root regular marker ownership, marker-last ordering and injected failures both before and after sentinel removal; both interruption windows retain target/staging evidence and lack readiness.
  - Implement atomic final-directory `mkdir` reservation without resolving the final path; require same resolved parent and one explicit root non-symlink regular marker from `partition_manifest.json`, `COMPLETE`, or `conversion_manifest.json`; move payload first and publish only that marker last.
  - Persist `.NIMLOTH_PUBLISHING.json` during publication. Reject missing/other/nested/symlink/directory markers, residual sentinel and any pre-existing target; do not fallback to check-then-rename, replace an empty target, delete partial target/staging evidence, or reinterpret a markerless directory as complete.
  - Add path-aware partition/shard/conversion publication gates. Partition consumers must rehash all ten sibling parquet files; shard/conversion validators must require marker-last completion before existing deep validation.
  - Update partition, collector and converter call sites plus ownership docs; call the guarantee atomic reservation/readiness publication rather than atomic whole-directory visibility; run full local affected tests and read-only `trellis-check`.
  - Stop after local affected tests and read-only review. W-009 exclusively owns the subsequent Nimloth commit approval, push approval, clean remote refresh, all-three-marker NFSv3 probes and fresh partition publication; no remote mutation, checkpoint merge, GPU or Slurm occurs in W-013.

- [ ] [W-009] **Complete post-W-013 review, commit gates and remote preflight**
  - Preserve approved/pushed VAGEN `170a673...` unchanged. Treat Nimloth launch code `187fe112...` and all later docs-only commits as pre-W-013; none contains the NFS publication fix and none is launchable for a retry.
  - After W-013 local tests/review, obtain a separate approval for one task-owned Nimloth code/docs commit, then a separate exact push approval for `HEAD:refs/heads/task/rollout-vagen-step60-sft1-sft2`. Do not change `origin/dev`, live server checkouts or the Nimloth gitlink.
  - Refresh the existing clean remote Nimloth worktree to that exact post-W-013 commit only after push, while retaining VAGEN worktree `170a673...`. Reassert top-level/HEAD/common-dir/controller registration and recursive approved submodules; run remote CPU affected suites and all-three-marker NFSv3 publication probes, then verify actual 0.8.2 package/import/EOS provenance, source hashes, reconstruction manifest, checkpoint merge/load plan and a fresh output identity. Cleanup, if later requested, uses reviewed non-force exact-path removal only.
  - Recheck `normal` one-node/four-GPU availability and present exact TP2 policy + two-environment GPU binding, CPU/memory/walltime, paths, commands, resume/cancel and monitoring contract.
  - Obtain a separate explicit experiment launch approval for the exact merge/smoke/concurrency/batch1 commands.

- [x] [W-014] **Preserve virtualenv executable ownership in the checkpoint merger**
  - Add RED coverage proving a symlinked `<venv>/bin/python3` remains the merger command executable rather than resolving to the system interpreter target.
  - Replace only executable-path canonicalization with lexical absolute normalization while retaining existence/executable checks; keep checkpoint, merger-script and target path validation unchanged.
  - Run focused checkpoint tests and the affected SFT1 suite, then review and commit/push only the exact task ref from the clean isolated worktree.
  - Re-run remote inert merge planning and require both `command[0]` and `python_executable` to equal `/project/peilab/atst/nimloth/.venv/bin/python3`; verify `torch`, `torch.utils`, `accelerate` and merger imports under that interpreter before any new launch approval.
  - Preserve failed run `20260902T123000Z_step60_batch1_v3_nfs_32bcc045`; retry only with a fresh run root, refreshed exact code commit and fresh launch approval.

- [x] [W-015] **Add per-trajectory durable checkpoints and explicit interrupted-shard resume**
  - Add RED tests proving each completed trajectory is atomically checkpointed and fsynced before the next trajectory completes, an interruption preserves completed rows, and explicit resume invokes policy/environment only for unfinished source rows.
  - Add RED cases for fresh-vs-resume mode, stable staging identity, changed ordered specs/runtime/policy/max-step/format contract, corrupt/truncated checkpoint JSON, record-hash drift, missing/tampered image evidence, duplicate/unknown source rows and an existing final output; all must fail before rollout calls.
  - Implement a stable direct-sibling in-progress directory with immutable hash-bound collection metadata, per-trajectory checkpoint files and attempt-unique image namespaces. Preserve incomplete attempt evidence; never infer or resume half an environment trajectory.
  - On complete coverage, regenerate `raw.jsonl` deterministically in requested source-spec order, retain resume audit evidence, produce the existing strict manifest/COMPLETE marker and publish marker-last. Keep final v3 raw/shard/conversion semantics unchanged.
  - Add CLI `--resume` fail-closed behavior, document the exact fresh/resume commands, run focused collector/data/converter tests and then one final affected SFT1 suite plus Ruff/compile/shell/diff checks. Obtain implementation approval before code edits and separate commit/push approvals; do not launch GPU work in this item.

- [x] [W-016] **Implement the eight-GPU preempt collection orchestrator**
  - Add RED static/integration tests for a one-node eight-GPU batch script that assigns four isolated environment GPUs and four independent TP1 policy collectors, uses the existing environment-service entrypoint, and never shares one allocated GPU between roles.
  - Run source-index-0 smoke on one pair, validate the complete v3 shard, then and only then launch four distinct 100-row shard collectors in parallel on the four pairs. Any smoke/shard failure must fail the job and block later work.
  - Detect each output as exactly one of absent (fresh), valid complete (skip), or matching in-progress (explicit `--resume`); reject every other state. Preserve per-shard logs, PIDs/status, cancellation cleanup and attempt evidence.
  - Require exact code/runtime/model/partition/contract paths through environment variables; do not merge actors, repartition data, convert datasets or launch beyond four gate shards.
  - Run focused RED/GREEN tests, shell syntax, affected SFT1 tests and read-only review. Obtain separate commit/push and launch approvals; no GPU/remote launch occurs in this item.

- [ ] [W-010] **Launch and monitor checkpoint merge, smoke and production-concurrency gate**
  - Run the approved merge/load preflight.
  - Run one-trajectory TP1 smoke and validate prompt hashes, parser/reward/runtime contract, transcript/image alignment and terminal non-execution.
  - If smoke passes, run four production-size shards in parallel using four environment/policy pairs; inspect scheduler/log/GPU/resource/output evidence until healthy or terminal.
  - On any terminal event, run `on-experiment-end`; retry only after root-cause review and fresh approval if the contract changes.

- [ ] [W-011] **Launch, monitor and finalize batch1**
  - Launch only remaining approved batch1 shards.
  - Monitor to terminal completion; validate exact 2,000-row coverage and every complete-shard manifest.
  - Convert and validate SFT1/SFT2 datasets, prove the 1,800/200 split and zero seed overlap, and record hashes/counts/rejections/reconstruction limitations and exact resume state.
  - Do not launch batches2–10.

## Planned validation commands

Exact filenames may be refined during implementation without changing semantics; material scope changes require replanning.

```bash
pytest -q tests/training/sft1 tests/rollout tests/test_wm_transition_dataset.py
python3 -m compileall -q experiments/training/sft1 src/nimloth/environment/navigation src/nimloth/rollout
bash -n experiments/training/sft1/*.sh experiments/training/sft1/*.slurm
PYTHONPATH=<VAGEN_RECONSTRUCTION_WORKTREE> pytest -q \
  <VAGEN_RECONSTRUCTION_WORKTREE>/tests/test_navigation_step60_reconstruction.py \
  <VAGEN_RECONSTRUCTION_WORKTREE>/tests/test_navigation_hligb_single_action_compat.py \
  <VAGEN_RECONSTRUCTION_WORKTREE>/tests/test_navigation_source_eval_compat.py
python3 -m compileall -q \
  <VAGEN_RECONSTRUCTION_WORKTREE>/vagen/env/navigation \
  <VAGEN_RECONSTRUCTION_WORKTREE>/vagen/env/utils \
  <VAGEN_RECONSTRUCTION_WORKTREE>/tests/test_navigation_step60_reconstruction.py
python3 ./.trellis/scripts/task.py validate .trellis/tasks/09-01-rollout-vagen-step60-sft1-sft2
test "$(git -C <VAGEN_RECONSTRUCTION_WORKTREE> rev-parse --show-toplevel)" = <VAGEN_RECONSTRUCTION_WORKTREE>
test "$(git -C <VAGEN_RECONSTRUCTION_WORKTREE> branch --show-current)" = task/step60-runtime-reconstruction
test "$(git -C <VAGEN_RECONSTRUCTION_WORKTREE> rev-parse --path-format=absolute --git-common-dir)" = /workspace/remote2/nimloth/.git/modules/external/VAGEN
git -C /workspace/remote2/nimloth/external/VAGEN worktree list --porcelain | grep -Fqx "worktree <VAGEN_RECONSTRUCTION_WORKTREE>"
test "$(git -C <VAGEN_RECONSTRUCTION_WORKTREE> rev-list --count 3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a..HEAD)" = 1
test "$(git -C <VAGEN_RECONSTRUCTION_WORKTREE> rev-list --parents -n 1 HEAD | awk '{print NF}')" = 2
test "$(git -C <VAGEN_RECONSTRUCTION_WORKTREE> rev-parse HEAD^)" = 3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a
git -C <VAGEN_RECONSTRUCTION_WORKTREE> rev-parse HEAD
git -C <VAGEN_RECONSTRUCTION_WORKTREE> rev-parse 'HEAD^{tree}'
test -z "$(git -C <VAGEN_RECONSTRUCTION_WORKTREE> status --porcelain=v1 --untracked-files=all)"
LC_ALL=C git -C <VAGEN_RECONSTRUCTION_WORKTREE> --no-pager diff --binary --full-index --no-ext-diff 3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a..HEAD -- | sha256sum
git -C <VAGEN_RECONSTRUCTION_WORKTREE> diff --check 3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a..HEAD --
git diff --check
```

Remote preflight/launch commands will be written verbatim into the task research/run contract after implementation and before launch approval; placeholders are not launch authorization.

## Risk and rollback points

- Checkpoint merger incompatibility → stop before GPU rollout; do not substitute checkpoint.
- Reconstruction prompt/parser/reward/API evidence mismatch → stop before commit or GPU; do not weaken golden hashes or relabel another runtime.
- Prompt/runtime/image mismatch → stop after one-row smoke; preserve evidence and replan.
- Missing terminal image/response or accidental terminal step → invalidate attempt; no conversion.
- Partial shard → retain but exclude; only valid completion manifest is resumable.
- Source parquet/hash drift → stop; do not regenerate partition silently.
- Scheduler/preemption failure → record end event; resume only complete shards under the approved contract.
- Unknown dirty/local changes → preserve and exclude from task commits.

## Approval gates

1. Fresh planning review and **implementation approval** for the material reconstruction scope before W-007 code changes.
2. Complete-diff review and exact **commit/push approvals** for both the Nimloth task ref and isolated VAGEN reconstruction ref.
3. Exact **experiment launch approval** after both committed refs, clean remote worktrees, reconstruction evidence, partition and total GPU allocation are presented.
4. Separate approvals for any batch after batch1.
