# Runtime path audit — VAGEN step60 batch-1 collection

Date: 2026-09-01

## Source checkpoint and source code availability

- The requested step60 checkpoint is a lightweight world-size-8 actor checkpoint with no HF model weights under `actor/huggingface`.
- The source training log identifies VAGEN commit `fee3ffac036a599b0ae979a6dd1ce2b21f7dec49` and source checkout `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505`.
- That home checkout is not readable by the current remote user, and the commit object is not present in the current local `external/VAGEN` object database. Therefore the task cannot claim to run the exact source tree unless that object becomes available through an authorized repository source.
- The checkpoint format is standard VERL FSDP shards. Current `external/VAGEN/verl` contains both `verl.model_merger` and `scripts/legacy_model_merger.py`; the legacy merger is the candidate because the lightweight checkpoint has no modern `fsdp_config.json`. Compatibility still requires an actual merge/load preflight.

## Existing source-lineage evaluation evidence

A prior full evaluation of the same training run's step50 checkpoint exists at:

- `/project/peilab/hligb/vagen-navigation/eval/full_516668_step50_full_20260814T182512Z_train_all/results.jsonl`
- `/project/peilab/hligb/vagen-navigation/eval/full_516668_step50_full_20260814T182512Z_train_all/summary.json`

Verified properties:

- the evaluator merged the actor into a temporary HF directory and loaded two safetensor shards with vLLM;
- the first raw transcript establishes the actual source prompt/action protocol: `grounding_worldmodeling`, strict `<think>...<answer>one_action</answer>`, legacy action names, one action per response;
- persisted rows contain `source_index`, `output_str`, metrics and `num_images`, but not image files/paths or the terminal observation;
- the run processed 13,744 of 20,000 rows before ending, so it cannot be reused as the requested dataset;
- existing output is valuable as a golden prompt/transcript fixture only.

## Current Nimloth reusable path

Current production code already provides:

- `VAGENNavigationSession` and batch client lifecycle;
- rollout collectors that save every observation image, including the terminal observation;
- `AgentRuntime.terminal_state()` semantics: generate terminal response but execute no draft action;
- current `nimloth_trajectory_v1` storage and SFT2 transition validation.

The existing collectors are bound to current Nimloth prompt/action policies and sequential synthetic seeds. Batch1 needs explicit source row identities/seeds and source `<answer>` response generation. The design therefore adds a bounded source-policy collection adapter and deterministic conversion rather than reusing the legacy VAGEN validation dump blindly.

## Source prompt/runtime recheck after server access

Read-only inspection of the same-run step50 evaluator and raw result established:

- evaluator entrypoint: private source checkout `vagen.inference.navigation_chunked_eval`; vLLM loaded the merged Qwen2.5-VL actor and the legacy batch service;
- source training resolved rollout: `max_model_len=6144`, `response_length=256`, `temperature=0.7`, `top_p=0.95`, `top_k=-1`, `n=1`, `window_size=5`, `limit_mm_per_prompt=6`, `max_turns=20`;
- raw `output_str` is a full `System:/User:/Assistant:` audit transcript with strict `<think>...<answer>one_action</answer>` responses; it is not itself the processor-rendered Qwen token prompt;
- archived source system prompt SHA256 is `d691e077a5a4204386d3958a81d08f4322d6618dbee0f740b2c4848ddf2bc99a`; after replacing only the `Human Instruction:` value, the initial user prompt template SHA256 is `95d3469f8d076ab788b3d100407d0200541fcb33fe006af941f224f69a7757e2`;
- both hashes matched sampled source indices across `base` and `common_sense`, including indices 0, 9999, 10000 and 13000;
- a later VPN-restored read-only check normalized only extracted action, feedback, reward, done and instruction values in 71 post-step user turns from seven sampled rows across both categories. After `rstrip()` to remove the evaluator transcript splitter's optional trailing blank line, every turn had the same SHA256 `c0d89b9a3949ef747676ba00d10b488a91b03fa80c2beb90d488d7de316824e7`;
- the prior evaluator persisted all transcript images in its count but configured a bounded multimodal request; the source training contract confirms five completed history turns plus the current observation, at most six images.

The exact source checkout `/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505` remains permission-denied, and commit `fee3ffac...` is absent from the accessible Nimloth VAGEN object databases. Accessible commits `f7aefd...` and `44be18c...` do not reproduce the archived strict prompt/config exactly and are therefore not accepted as silent substitutes. The collector now requires a clean runtime at the exact `fee3ffac...` commit and a hash-bound source-runtime contract that records the resolved step length, success reward, action order, three archived prompt hashes, legacy batch API and 500-second service timeout. It fails closed until that object/worktree and evidence become accessible. This is a launch-preflight blocker, not permission to approximate the environment.

A subsequent read-only lineage audit established what an evidence-backed reconstruction could and could not reuse:

- accessible legacy commit `44be18c` implements the expected Flask batch routes (`/health`, `/environments`, `/batch/reset`, `/batch/step`, `/batch/reward`, `/batch/system_prompt`, `/batch/close`), compact action dispatch, `step_length=0.5`, `success_threshold=1.5`, success reward `10.0`, and per-step format reward;
- its default format reward is `0.5`, while the source parquet/runtime config pins `0.02`; its code does not expose the source run's `invalid_action_penalty=-0.2` field, so it is not exact without reviewed adaptation;
- later accessible commits `db59a11`, `dda9239`, and `3003c2e` are prior evidence-driven compatibility reconstructions for other archived VAGEN runs. They demonstrate that isolated prompt/parser modes are feasible, but their golden prompt hashes do not equal this task's archived step60 hashes;
- the committed Nimloth collector independently checks this task's exact three prompt hashes and exact clean `fee3ffac...` HEAD, so none of those accessible commits can pass by relabeling metadata.

The human subsequently approved replanning to this bounded reconstruction route. The reviewed base is now exact `3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a` (a descendant of `44be18c` with isolated compatibility-mode scaffolding). Implementation still requires a fresh gate and must produce a separately named VAGEN patch commit plus a Nimloth contract that reports reconstruction honestly; this planning approval does not authorize code changes, commits, GPU or launch.

## Terminal CoT decision

Human decision:

- the same VAGEN step60 policy generates full CoT + draft action for the terminal observation;
- the draft action is never executed and creates no transition/reward/success;
- terminal CoT is not an SFT1 assistant target and is not an SFT2 LLM-backbone target;
- its only training-time role is to condition the final state encoder input;
- conversion persists a Nimloth terminal prefix ending at `action_start`, plus separate audit evidence for the raw source terminal response/draft action.

## Data staging decision

The task context records the approved staging shape:

- deterministically partition source train rows into ten disjoint batches;
- each batch has 1,000 `base` + 1,000 `common_sense` rows, preserving category-local parquet order;
- only batch1 is in this task's launch scope; batches2–10 need new run/output identities and fresh launch approvals;
- source test overlaps train on every `(eval_set, seed)` key and is not held-out generalization evidence.

## Resource observation (not approval)

At the 2026-09-01 read-only query:

- normal: 6 free GPUs across three nodes, including one node with 4 free;
- preempt: 7 free GPUs across two nodes, including one node with 4 free.

Availability is transient and must be queried again immediately before launch. The human selected `normal`, one node / four GPUs as the preparation direction: two policy TP ranks plus two environment GPUs, with complete-shard resume. Exact CPU/memory/walltime/device binding and the full command remain subject to the separate launch approval.

## Prompt/chat dual-view decision

The human requires SFT1/SFT2-compatible training format while preserving the original real prompt and chat. The selected contract is dual-view:

- raw/source audit: verbatim step60 rendered prompt, system/user/assistant messages, ordinary responses and terminal full response, with canonical hashes;
- converted training view: current K16 Nimloth prompt/response/action format, with only format-level rewriting and no change to task content, observations or real CoT;
- manifest: source/converted hashes and conversion-contract version bind both views; neither view overwrites or impersonates the other.

## Internal held-out decision

The human selected a deterministic 10% held-out within batch1. For each category, `category_local_ordinal % 10 == 9` is held out. The source audit established that `base` and `common_sense` share the same ordered seed sequence, so the two rows for one seed are always assigned together. Expected outputs are 1,800 train and 200 internal held-out rows with zero bare-seed overlap. This split is only internal unseen-seed evidence and is not an unseen environment-distribution generalization claim.
