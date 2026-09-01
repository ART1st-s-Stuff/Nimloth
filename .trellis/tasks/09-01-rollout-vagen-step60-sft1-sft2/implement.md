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

- [ ] [W-007] **Execute remote preflights and request exact experiment launch approval**
  - Recheck source paths/hashes, remote clean worktree/commit, Python/runtime, checkpoint merge/load and output nonexistence.
  - Recheck the selected `normal`, one-node/four-GPU topology and present exact TP2 policy + two-environment GPU binding, CPU/memory/walltime and final commands.
  - Obtain a separate explicit approval for the exact merge/smoke/batch1 launch contract.

- [ ] [W-008] **Launch and monitor checkpoint merge, smoke and production-concurrency gate**
  - Run the approved merge/load preflight.
  - Run one-trajectory smoke and validate prompt/transcript/image/terminal semantics.
  - Run one production-size shard; inspect scheduler/log/GPU/resource/output evidence until healthy or terminal.
  - On any terminal event, run `on-experiment-end`; retry only after root-cause review and fresh approval if the contract changes.

- [ ] [W-009] **Launch, monitor and finalize batch1**
  - Launch only remaining approved batch1 shards.
  - Monitor to terminal completion; validate exact 2,000-row coverage and every complete-shard manifest.
  - Convert and validate SFT1/SFT2 datasets, prove the 1,800/200 split and zero seed overlap, and record hashes/counts/rejections/limitations and exact resume state.
  - Do not launch batches2–10.

## Planned validation commands

Exact filenames may be refined during implementation without changing semantics; material scope changes require replanning.

```bash
pytest -q tests/training/sft1 tests/rollout tests/test_wm_transition_dataset.py
python3 -m compileall -q experiments/training/sft1 src/nimloth/environment/navigation src/nimloth/rollout
bash -n experiments/training/sft1/*.sh experiments/training/sft1/*.slurm
python3 ./.trellis/scripts/task.py validate .trellis/tasks/09-01-rollout-vagen-step60-sft1-sft2
git diff --check
```

Remote preflight/launch commands will be written verbatim into the task research/run contract after implementation and before launch approval; placeholders are not launch authorization.

## Risk and rollback points

- Checkpoint merger incompatibility → stop before GPU rollout; do not substitute checkpoint.
- Prompt/runtime mismatch → stop after one-row smoke; preserve evidence and replan.
- Missing terminal image/response or accidental terminal step → invalidate attempt; no conversion.
- Partial shard → retain but exclude; only valid completion manifest is resumable.
- Source parquet/hash drift → stop; do not regenerate partition silently.
- Scheduler/preemption failure → record end event; resume only complete shards under the approved contract.
- Unknown dirty/local changes → preserve and exclude from task commits.

## Approval gates

1. Final planning review and **implementation approval** before `task.py start`.
2. Complete-diff review and **commit approval** before committing task code.
3. Exact **experiment launch approval** after committed code, remote preflights, partition and total GPU allocation are presented.
4. Separate approvals for any batch after batch1.
