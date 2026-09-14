# Research: B experiment prompt conversion audit

- Query: Does current SFT1 preserve the rollout prompt while converting semantic action output requirements into action tokens?
- Scope: internal; current source, local real generation artifact, collection source audit.
- Date: 2026-09-10

## Findings

### Actual caller and preserved source protocol

`experiments/training/sft1/vagen_step60_convert.py:15` imports `convert_source_record` from `vagen_step60_data.py`; its actual prompt conversion is `vagen_step60_data.convert_source_prompt:648`, invoked for system and observation strings at lines 1106–1112. The generic `convert_rollouts.rewrite_prompt_instruction:108` is not this dataset's active converter.

Collection evidence `.trellis/tasks/09-01-rollout-vagen-step60-sft1-sft2/research/runtime-path-audit-2026-09-01.md:22` establishes grounding_worldmodeling, strict think/answer, legacy action names and exactly one action per response. Current conversion preserves observation/reasoning/prediction text and substitutes answer envelopes. Source records retain the original system prompt/messages in source_audit (`vagen_step60_data.py:954`). No original data should be edited.

### Confirmed contradictory instructions in a real current prompt

The local actual generation artifact `.trellis/tasks/09-10-sft1-rollout2000/research/20260910T165800Z_weight8_epoch2_outputs/responses.jsonl:1` contains the rendered prompt for `vagen-step60/000009`. Within its prompt_text (display line numbers below, not JSONL line numbers):

- line 13: `Format correct only when the answer is exactly one valid action name.`
- line 15: `Invalid action names receive negative feedback.`
- line 22: `The answer must be exactly one of these lowercase action names: moveahead, moveback, moveright, moveleft, rotateright, rotateleft, lookup, lookdown.`
- lines 31 and 48 (system and user): `The action must be exactly one of these lowercase action names: moveahead, moveback, moveright, moveleft, rotateright, rotateleft, lookup, lookdown.`

Yet lines 20/39/56 request `<|action_start|><|action_(idx)|><|action_end|>`, concrete examples already use action_(0), and line 25 explicitly forbids natural-language action names inside the new envelope. Therefore the actual training/evaluation prompt contradicts itself, beyond merely retaining useful semantic descriptions.

This is a credible contributing defect, not proof that it alone caused all format failures. The weak initial action head is separately established and should not be dismissed.

### Why current checks missed it

`convert_source_prompt:654–682` detects/replaces answer envelope strings and checks their removal; it does not rewrite the prose action-name requirements. Lines 683–688 append only `Nimloth action indices: 0=moveahead, ...`, not explicit token-to-meaning definitions. `_validate_sft1_record` (`vagen_step60_convert.py:93`) checks answer tags but not obsolete prose requirements.

`tests/training/sft1/test_vagen_step60_data.py:49` uses a simplified source prompt containing just navigation and a format example. Checks around 599 verify tag removal/target preservation, so they do not reproduce the real contradictory source instructions. `test_convert_rollouts_prompt.py` covers the other converter.

### Mapping is semantically consistent

`vagen_step60_data.py:64–73` order is moveahead, moveback, moveright, moveleft, rotateright, rotateleft, lookup, lookdown. `_action_envelope:634–645` uses that exact position for the action token. VAGEN canonical ACTION_NAMES in `/workspace/remote2/nimloth/external/VAGEN/vagen/envs/navigation/utils/nimloth_format.py:5` correspond respectively to move_forward, move_backward, move_right, move_left, turn_right, turn_left, look_up, look_down. No direction swap was found.

### Minimal correction before B

1. Create a new derived SFT1 view; preserve raw rollout/source_audit and existing dataset/cache immutably.
2. Rewrite only output-protocol prose in both system and user messages: valid action name -> valid action token; exact lowercase-name requirement -> exact full eight-token enumeration. Keep semantic action descriptions, actual goals, feedback, observation text and CoT unchanged.
3. Replace numeric-only legend with explicit `<|action_(0)|> = moveahead (move forward)` etc., distinguishing movement from turning and camera tilt.
4. Keep correct concrete examples and envelope; pure stage1 removes latent markers through its existing render path.
5. Add a regression using the real prompt clauses above, checking no old output-name requirement survives in system or user and all eight mappings are correct. Validate the full derived 1709/193 dataset before rebuilding caches; old cache cannot be reused with changed prompt.
6. Preserve count/split/action target/CoT/image lineage and publish a source-to-derived prompt-only diff and hashes.

## Related specs

`.trellis/spec/domains/sft-stages.md`: stage1 format supervision; preserve genuine CoT and reject truncation/misalignment. `.trellis/workflow.md`: persist research and keep task-scoped evidence.

## Caveats / Not Found

No remote query performed. Local real artifact proves the defect in sampled prompts; main session is checking full current dataset coverage. Current external/VAGEN source is used for canonical mapping only, not asserted identical to the historical collection checkout. This audit changed no code, original data, cache, configuration, or training process.

## Main-session live full-dataset audit
2026-09-10 a100-1 read-only: source paths from conversion_manifest.json and every record source_audit checked. All1709train+193heldout source_record_sha256 match original files; all1902 source_prompt contain <answer>; all1902 converted records retain lowercase action names output requirements; 22,364 action name/index pairs match canonical order. Raw row_000000 recording.history[0].obs_str directly confirms original collection uses <answer>...</answer> with legacy action names. Data correction remains pending user approval; source outputs untouched.
