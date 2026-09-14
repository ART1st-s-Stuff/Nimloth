# Reconstruction evidence — VAGEN step60 source behavior

Date: 2026-09-01

This is read-only planning evidence for the approved evidence-backed runtime reconstruction. It is not exact-source-code evidence and does not authorize implementation or launch.

## Immutable archived prompt fixture

Source W&B run directory:

`/project/peilab/hligb/vagen-navigation/wandb/wandb/run-20260813_092455-2q620nss`

Selected table fixture:

- file: `files/media/table/val/generations_14_6bc61d7bb480498be805.table.json`
- file SHA256: `6bc61d7bb480498be80547a45ff8932415c50e3de5943a1f159d0a5f47580c27`
- table row index: `0`
- transcript column: `output_1`
- extracted message byte lengths: system `3487`, initial user `1187`, first post-step user `1350`

Extraction parses Qwen chat boundaries exactly as `<|im_start|>{role}\n...<|im_end|>`. The initial prompt replaces only the first full `Human Instruction:` line with `<INSTRUCTION>`. The post-step prompt replaces only the first extracted-actions, feedback, reward, done and instruction lines with `<ACTIONS>`, `<FEEDBACK>`, `<REWARD>`, `<DONE>` and `<INSTRUCTION>`, then applies `rstrip()`. The resulting SHA256 values are:

- system: `d691e077a5a4204386d3958a81d08f4322d6618dbee0f740b2c4848ddf2bc99a`
- normalized initial: `95d3469f8d076ab788b3d100407d0200541fcb33fe006af941f224f69a7757e2`
- normalized post-step: `c0d89b9a3949ef747676ba00d10b488a91b03fa80c2beb90d488d7de316824e7`

Implementation must extract the literal fixture from this hash-pinned table, not retype or paraphrase it.

## Reward/parser evidence

A read-only scan of all 12 archived W&B generation tables parsed 7,351 assistant→environment turns. Returned step rewards were:

| reward | turns | observed class |
|---:|---:|---|
| `-0.2` | 596 | no extracted action; invalid/forbidden/typo answer or malformed strict envelope |
| `0.0` | 35 | no extracted action; more than one otherwise valid compact action |
| `0.02` | 6,361 | one strict valid compact action, non-success |
| `10.02` | 359 | one strict valid compact action reaching success |

The `-0.2` rows included forbidden `stay`/`stop`, invalid spellings, and a valid action name inside a non-strict envelope; all had `environment_extracted_actions=[]` and `done=0.0`. The `0.0` rows contained two otherwise valid compact actions such as `moveback,moveright`; they also had no extracted action and `done=0.0`. This resolves the reconstruction rule:

1. exactly one strict valid compact action → execute it and add `0.02`;
2. if that action reaches the 1.5 m threshold → add `10.0`, total `10.02`, and terminate;
3. invalid/forbidden/typo action or malformed strict envelope → execute no action and return `-0.2`;
4. too many otherwise valid actions → execute no action and return `0.0`;
5. failed AI2-THOR movement is not an invalid parser action; its returned reward remains the valid-action format reward and `last_action_success` records the physical failure.

Source training logs independently expose `dense_invalid_action_penalty` and separate `format/error/too_many_actions` from `format/error/invalid_action_name`, consistent with the table classes. The smoke must recheck these cases through the actual batch service.

Reward provenance for reconstruction is therefore `step_rewards`. Any proposed fallback to a terminal aggregate is a material replan.

## Sampling invocation evidence

Source log:

`/project/peilab/hligb/vagen-navigation/logs/navigation_vagen1_native_8gpu_rmb4_ppo16_val5_save5_lightckpt_48h_20260813T011326Z.log`

The printed resolved config has no `actor_rollout_ref.rollout.val_kwargs`. It contains one rollout configuration at `actor_rollout_ref.rollout`:

- `do_sample=true`
- `temperature=0.7`
- `top_p=0.95`
- `top_k=-1`
- `n=1`
- `response_length=256`
- `ignore_eos=false`
- `max_model_len=6144`
- `limit_mm_per_prompt=6`

Runtime log lines print the same vLLM kwargs during both the run and archived validation generation. Therefore the current task is not selecting optimization kwargs over a distinct source `val_kwargs`; this source run exposes no separate validation sampling block. This distinction must remain explicit because known error E0016 applies to runs that do have separate `val_kwargs`.

The archived W&B `requirements.txt` pins vLLM `0.8.5.post1`, Transformers `4.49.0` and PyTorch `2.6.0`. In vLLM `0.8.5.post1`, `CompletionOutput.stop_reason` is documented as `None` for EOS and a string/token ID for a custom stop; `StopChecker` marks EOS as stopped without setting `stop_reason`, while custom stop tokens/strings set it explicitly. Reconstruction therefore persists package versions, tokenizer/config hashes, EOS ID, generated token IDs, `finish_reason` and `stop_reason`, requires empty custom stop lists plus `ignore_eos=false`, and accepts only `(finish_reason="stop", stop_reason=null)`. Source references:

- `https://github.com/vllm-project/vllm/blob/v0.8.5.post1/vllm/outputs.py`
- `https://github.com/vllm-project/vllm/blob/v0.8.5.post1/vllm/engine/output_processor/stop_checker.py`

For the human-selected executable vLLM `0.8.2`, read-only inspection of canonical `.venv` found the same relevant contract: `CompletionOutput.stop_reason` is `None` for EOS, and `StopChecker` checks `last_token_id == eos_token_id`, sets `FINISHED_STOPPED`, and returns without setting a custom stop reason. Exact installed-source hashes are:

- `vllm/outputs.py`: `047d469792ba4b332fd6bc6837af03340135cb49798e1ddfd2ffa730ead436f8`
- `vllm/engine/output_processor/stop_checker.py`: `5ed39ad2df9912b7a4b9ff52168c50bfe9d937675d3f1122148c0824450afa28`

This verifies the metadata interpretation for 0.8.2; actual model generation/tokenization remains smoke-gated.

## Accessible reconstruction lineage

Approved base candidate:

`3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a`

It descends from `44be18c` and preserves the legacy Flask batch endpoints, compact-action compatibility scaffolding and 0.5 m navigation dynamics. It does not contain this task's exact strict prompt or invalid-action reward semantics. A new isolated mode and patch commit are therefore required; the base alone is never a launchable substitute.

The base's environment assets are hash-bound as `base.json` SHA256 `6b575621a6b15e90e1040dd86d661a5e1ee70134f42fd7f3d61706347449c55a` and `common_sense.json` SHA256 `3e7d2cb4246b6e2edaeaabd318dba93e4dbbff114c8368ed0c862e64f417afcf`, each with 60 tasks. Eleven sampled source evaluator rows across both categories (source indices 0, 1, 2, 100, 999, 1000, 9999, 10000, 10001, 11000 and 13000) matched the accessible asset instruction exactly under the runtime rule `task = dataset[seed % 60]`. Formal smoke still validates actual scene/image behavior; these instruction samples do not by themselves prove image parity.

A full read-only replay of the reconstruction classifier over all 7,351 archived W&B turns produced zero reward-class mismatches: 556 invalid action-name `-0.2`, 40 malformed-envelope `-0.2`, 35 too-many-action `0.0`, 6,361 valid `0.02`, and 359 valid-success `10.02`.

Invalid/no-action feedback is demonstrably stale source behavior rather than physical-action evidence: among `-0.2` turns, 491 report “Last action is executed successfully” and 105 report failure; among too-many-action `0.0` turns, 30 report success and 5 failure. The reconstructed environment therefore preserves the legacy last-event feedback while separately persisting `environment_extracted_actions=[]`; it must not reinterpret that feedback as execution of an invalid action.
