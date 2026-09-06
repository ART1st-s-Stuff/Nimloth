# Research: Source batch1 200-row minimal reuse

- Query: Reuse existing partition assets and rollout runner for exact 200 source rows, wm prompt, 0.3 m movement, 1 m success threshold.
- Scope: internal, read-only code inspection; no remote asset verification.
- Date: 2026-09-06

## Findings

### Files and contracts
- `experiments/training/sft1/vagen_step60_data.py:320` builds the pinned source partition. Original SHA256 is `3c8161bd45adc4cde5d67157cf4db225753ed3925cb9a52e3a57d1dd11dbe9d6`; source is exactly 20,000 rows, base first 10,000 then common_sense 10,000.
- `vagen_step60_data.py:369-395` defines category ordinal from actual file order, batches per 1,000 category rows, heldout when ordinal modulo 10 equals 9 within batch1. It verifies both categories have the SAME ORDERED seeds, ensuring shared seeds cannot be split across train/heldout.
- `vagen_step60_data.py:375-386` row manifest includes source_index, eval_set, seed, source_key (`eval_set:seed`), category_ordinal, batch, dataset_split. Batch1 source indices are 0..999 and 10000..10999. They are row positions, NOT seed ranges.
- `vagen_step60_data.py:425-446` requires 1,800 train / 200 heldout and zero intersection by source_index, eval_set/seed, and bare seed.
- `vagen_step60_data.py:608` load_published_partition_manifest validates publication readiness and manifest payload, then rehashes every sibling batch parquet. Batch1 parquet retains original source schema, with source_index ordering provided by batch entry source_indices; do not index its 2,000 rows with global indices such as 10000.
- `vagen_step60_data.py:237` source identity reader enforces ORIGINAL environment fields (including success_threshold=1.5 and grounding_worldmodeling). Validate source BEFORE creating runtime overrides; do not mutate/reseal original manifest to pretend new 0.3/1/wm settings were source defaults.

### Minimal runner integration
- `experiments/training/sft1/rollouts_greedy_parallel.slurm:363-432` currently invents rows from contiguous `range(seed_start,seed_end+1)` and uses source_eval_mode; it cannot implement the source-row request as-is.
- Add one opt-in prepared parquet input to `run_shard` around lines449-456, bypassing write_split_parquet entirely for this launch, and dispatch exactly that one prepared shard instead of the legacy loops at618-675. Avoid creating another collector/runtime framework.
- Prepare input from validated published batch1 parquet + manifest: select explicitly approved row manifest entries; map source_index to batch-local ordinal via batch source_indices; copy the actual row, preserve source seed/eval_set and source prompt; apply only approved env overrides (prompt mode runtime mapping must be verified, step_length=0.3, success_threshold=1.0). Add source_index/source_key/dataset_split provenance in extra_info and a sidecar selection manifest with source and parent partition hashes, ordered selected entries and runtime overrides.
- Require exactly200 unique selected source indices, expected category/split counts, membership in batch1, no altered seed/key, and SHA256 of prepared parquet. Keep original 2,000-row split manifest separate from runtime val_files naming: val_only collection through trainer does NOT make all selected rows heldout.
- `rollouts_greedy_parallel.slurm:492-499` hardcodes greedy `do_sample=False,temperature=0` including val_kwargs. If sampling requested, both execution val_kwargs and declarations must change; merely source parquet metadata cannot override it. At line506 max_response_per_turn512, line524 max_turns20 (verify exact current line), and trainer val_only must be considered when aligning requested generation parameters.
- Existing runner provides ordinary trajectory dumping, not automatically terminal-CoT or the old strict conversion COMPLETE format. This research does not establish SFT consumption readiness.

### Related specs / external references
- `.trellis/workflow.md`: evidence persisted under existing task; no new task needed for this subtask.
- `.trellis/spec/experiments/task-contract.md`: exact data lineage, parameters, resource/approval, health and validity boundaries before launch.
- No external web references used. Runtime reconstruction prompt-mode mapping has not been inspected by this researcher.

## Caveats / Not Found
- Parent session is resolving WHICH200 and sampling with user. Do not silently equate first200 with the200 heldout. In file order, first200 batch rows are all base; balanced pilot needs explicit category selection.
- Existing batch parquet/partition remote paths and live integrity not checked here; parent must verify assets before reuse.
- User requested wm; legacy reconstructed runtime may expose grounding_worldmodeling rather than literal wm. Confirm actual prompt text/parser dispatch; do not assume these names are interchangeable.

## Follow-up implementation audit

- User resolved selection: first100 batch1 TRAIN rows per category in source order, greedy20turns512tokens. Local prepared selector implements that; excludes ordinal9,19,... rather than selecting heldout or first100 unfiltered.
- Reviewed `prepare_source200.py`: source_index to batch-local row mapping, exact seed/eval_set checks, immutable source copying, 10x20 output shards, source published hash gates and recomputed exact prepared rows are correct. Suggested additional assertions for redundant declared source/batch1 hash/selection metadata; sent to implementer.
- Reviewed prepared runner branch: task0 only, smoke rejected, TRAIN_BATCH_SIZE=20, VAL_BATCH_SIZE=20, AGENT_NUM_WORKERS=1; this avoids the empty drop-last train loader at default24. Existing greedy20turn512token overrides remain relevant. Parent full launcher limits prepared work to task0.
- Reviewed separate VAGEN wm200 worktree bridge: source_wm_mode uses SOURCE_ACTION_LOOKUP, dedicated parser, current wm section order, matching base/format/initial/subsequent prompt text and no examples (config default0). It does not reuse source_eval_mode parser or alter legacy grounding_worldmodeling. Runtime reward continues existing source config format_reward (0.02 in source) rather than silently substituting eval per_turn reward. Current eval reward parity is not claimed.
- No real environment/GPU execution performed by this reviewer. Remote original publication hashes, live health and exact count200 remain parent launch checks.
