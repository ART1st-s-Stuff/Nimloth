# Research: Stage2 empty-CoT failure

- Query: Diagnose `query alignment requires the recorded nonempty CoT for this observation` and identify the smallest correct repair without inventing CoT, silently skipping data, or weakening validation.
- Scope: internal
- Date: 2026-09-08

## Findings

### Conclusion

The failed run exposed an input-conversion contract gap, not an answer-index or observation-pairing bug. The current converted JSONL contains at least one assistant answer rendered as an empty CoT. The historical converter deliberately turns both “source has no `<think>` block” and “source has an empty/whitespace `<think>` block” into the same `<think></think>` output, while its validation accepts that output. Stage2 correctly rejects it because query alignment requires a real, nonempty CoT from the same observation.

The original source payload is needed to distinguish a missing tag from an empty body for each affected turn. That distinction is no longer recoverable from the converted row alone because conversion erased it.

### Runtime evidence

- Job step `561674.0` reached four-rank model initialization, then rank 3 failed in the first training iterator at `QueryAlignmentCollator -> answer_examples`; no optimizer update or checkpoint was produced. The recorded traceback points to `src/nimloth/training/sft/stage2/data.py:85-87`. The task record preserves the run identity and artifact boundary at `.trellis/tasks/09-05-unify-sft-training-evaluation/progress.md:263-284`.
- The launch used `train_success.jsonl` and `val_all.jsonl` from the old `converted_strict_k8_b6c811c` artifact (`experiments/training/sft/evaluation/run_stage2_from_corrected_sft1.sh:28-32,132-139`). It did not use the newer trajectory schema or a Stage2-specific eligibility manifest.
- The current experiment preflight counts answers/images and verifies paths, but never checks assistant CoT content (`experiments/training/sft/evaluation/pipeline_contract.py:88-145`). This is why the expensive model/GPU path found the defect instead of preflight.

### Why indexing and observation pairing are not the defect

- `AnswerPrefixDataset` indexes every assistant message as `(trajectory row, message index)` and returns the exact prefix ending at that answer (`src/nimloth/training/sft/stage2/data.py:22-48`).
- `answer_examples` selects the sole image from the immediately preceding user message, consumes it once, and rejects missing or ambiguous observations (`src/nimloth/training/sft/stage2/data.py:51-90`). The observed exception occurs only after this image check, at the assistant-content check.
- Tests exercise two observations/two different CoTs and the per-answer prefix path (`tests/training/sft/stage2/test_query_alignment.py:119-132,293-329`). These tests do not prove the remote dataset is valid, but they do cover the suspected index/pairing mechanism.

### Conversion defect and source-data boundary

- `convert_assistant` extracts `<think>(.*?)</think>`, then sets `think = ""` both when the tag is absent and when its body becomes empty after `strip()` (`experiments/navigation_baseline/convert_sft1_rollouts_to_nimloth.py:52-56,168-177`). It then emits `<think>{think}</think>` regardless.
- `validate_record` checks action counts and Nimloth action/query tokens but does not require a present, nonempty CoT (`experiments/navigation_baseline/convert_sft1_rollouts_to_nimloth.py:254-282`). Consequently, a successful trajectory with an empty converted CoT can enter `train_success`; `val_all` includes records regardless of validation issues in any case (`experiments/navigation_baseline/convert_sft1_rollouts_to_nimloth.py:350-364,383-404`).
- The old VAGEN navigation parser recognizes `<think>...</think>` structurally but does not require its body to be nonempty (`external/VAGEN/vagen/envs/navigation/utils/parse.py:23-31,62-67`). Thus an action-valid rollout is not automatically query-alignment-valid.
- Current SFT2 explicitly requires an observation-aligned nonempty CoT and rejects missing CoT rather than fabricating text (`src/nimloth/training/sft/stage2/data.py:80-87`; `src/nimloth/training/sft/stage2/README.md:20-26`; `.trellis/spec/governance/cot-and-state.md:5-17`; `.trellis/spec/domains/sft-stages.md:17-25`). The pseudocode also places query state after the recorded CoT (`src/nimloth/training/sft/spec.md:19-27,44-53`). Weakening this gate would violate the agreed algorithm.

### Smallest correct repair

1. Add a CPU Stage2 eligibility audit that scans every assistant turn in both exact JSONLs before model loading. It must report split, record ID, assistant-turn index, source path/line, and distinguish `missing_think_tag` from `empty_think_body` when original source is available. The launch must fail if accounting is incomplete.
2. Preserve the old JSONLs. Materialize new, versioned Stage2 input JSONLs plus a manifest containing source hashes, before/after trajectory and answer counts, every exclusion ID/reason, output hashes, and conversion version. Exclude an entire trajectory if any of its supervised assistant turns lacks real nonempty CoT; this avoids silently changing a trajectory's history or relabeling an invalid turn. Use these explicit artifacts for both train and validation.
3. Update the historical converter for future conversions so absent and empty CoT become validation issues rather than being normalized into an apparently valid answer. Keep the runtime nonempty-CoT gate.
4. Extend the experiment preflight to validate all Stage2 CoT/query/action boundaries and record the Stage2 input manifest before requesting/using GPU resources. Do not merely add a collator-time skip.
5. If retaining every old turn is required, data exclusion is insufficient: regenerate real observation-aligned responses from an explicitly selected checkpoint and sampling contract, preserve provenance, and create a new dataset identity. A code-only repair cannot reconstruct missing CoT.

The existing DINO cache may be reusable as a superset for an explicitly filtered JSONL because targets are loaded by observation path, but this must be verified against its manifest/loader identity before retry; the cache must contain every retained image and no launch contract may claim its original counts equal the filtered dataset.

## Files Found

- `src/nimloth/training/sft/stage2/data.py` — answer indexing, observation pairing, nonempty-CoT gate, and DINO target loading.
- `src/nimloth/training/sft/stage1/data.py` — converted JSONL loader and image placeholder binding.
- `experiments/navigation_baseline/convert_sft1_rollouts_to_nimloth.py` — old VAGEN-to-SFT converter that collapses absent/empty CoT and omits validation.
- `experiments/training/sft/evaluation/pipeline_contract.py` — current preflight, which checks counts and image alignment but not CoT content.
- `experiments/training/sft/evaluation/run_stage2_from_corrected_sft1.sh` — failed run's exact old-data and Stage2 launch wiring.
- `tests/training/sft/stage2/test_query_alignment.py` — coverage for prefix indexing, observation pairing, and empty-CoT rejection.
- `src/nimloth/training/sft/spec.md` — SFT2 pseudocode and teacher-forced query-alignment semantics.
- `.trellis/spec/governance/cot-and-state.md` — hard ban on invented/filler CoT and requirement for observation alignment.
- `.trellis/spec/experiments/data-and-splits.md` — conversion provenance, exclusion accounting, and schema-validation contract.

## Related Specs

- `.trellis/spec/domains/sft-stages.md`
- `.trellis/spec/governance/cot-and-state.md`
- `.trellis/spec/experiments/data-and-splits.md`
- `src/nimloth/training/sft/spec.md`

## Caveats / Not Found

- Live SSH inspection failed with `Permission denied (publickey)` during this research pass. Exact affected record/turn counts and original raw-response forms were therefore not re-counted. Current evidence proves at least one invalid converted assistant turn, but not whether there are one or many.
- The traceback does not print the record ID or turn index. Adding identity-rich audit/error output is necessary before retry.
- No evidence supports borrowing a later observation, filling default reasoning, or relaxing the nonempty-CoT check.
