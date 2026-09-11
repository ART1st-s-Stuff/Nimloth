# Research: Early-stage environment evaluation

- Query: Reuse formal evaluation entrypoint for VAGEN, Stage 1 B, and Stage 2 without planner; identify protocol and launch blockers.
- Scope: internal source inspection; no remote execution or source edits
- Date: 2026-09-11

## Findings

### Reuse and boundaries

`src/nimloth/training/sft/evaluation/{config,cli,rollout}.py` is the existing formal entrypoint. `cli.run_evaluation` writes and validates immutable evaluation contract plus checkpoint fingerprints before calling rollout. Reuse this dispatch, extend explicit stage/protocol selection, and add thin stage1/eval.py and stage2/eval.py wrappers. Preserve WM branch and existing training collector behavior; current user scope excludes Stage 3/RL modifications.

`EvaluationConfig` currently models direct|wm, explicit held-out sets/count/seeds and generation settings; it has no checkpoint family or stage discriminator. `cli.build_rollout_argv` always selects vLLM and action-only credit. This is inadequate to identify VAGEN semantic vs B action-token vs query protocol.

### Actual blockers, not hypothetical

1. `backbone/qwen25vl/policy.py:62-73` validates positive K and inject only. Explicit None from format checkpoints raises even during int conversion. Stage2 generate is rejected. Do not relax this existing planner-facing validator globally: add stage-specific evaluation checkpoint validation.
2. `backbone/qwen25vl/vllm_policy.py:502-552` constructs a controlled turn protocol injecting latent queries AND action_start after think, constrains action tokens, ignores EOS, and stops at action_end. That is unsuitable for assessing Stage1's learned format and unsuitable for Stage2 generate. Early eval needs ordinary raw generation retaining special tokens. Inject mode may insert only the actual trained query interval, with provenance; never manufacture CoT/action envelope or valid actions after malformed generation.
3. `agent/templates/nimloth.py:19-53` rejects absent K and constructs latent/action prefixes. `build_response_policy_prompt` is a planner-oriented prefix; do not force Stage1 through this template. Use shared `agent.action_prompt.format_action_prompt` from B implementer on both system and observation prompts; Stage2 needs same B wording with explicit ordered query slots matching saved K/mode.
4. `environment/navigation/vagen.py:186-225` always configures prompt_format=nimloth and requires positive K. `reset:261-296` optionally substitutes an older custom prompt in vagen_eval profile. `step:299-305` validates action_index locally but forwards response TEXT unchanged to server; action_index does not execute the action. Therefore local parsed action and remote parser protocol must agree. Either select a verified matching service protocol or explicitly serialize a validated action into legacy semantic service input, persisting model response and actual service response separately. Invalid raw generation must remain invalid; do not repair or silently drop trials.
5. `navigation_environment_config` profile changes dynamics: current threshold1.5/step0.5 versus vagen_eval threshold1.0/step0.3 and success_reward1.0. Prompt protocol must be independent of dynamics choice. `step:310-312` success fallback reward>=10 is inconsistent with the latter reward profile if service omits task_success. Use actual environment success information/metrics with explicit provenance, not arbitrary reward threshold inference.
6. `training/sft/stage1/checkpoint_export.py:146-204` already exports format/query stage, query mode, latent count, format objective, and copies query projector metadata. Validate exact saved stage before model load; no silent K1 defaults. Stage2 eval does not require loading DINO teacher or external WM/projector to select actions, but checkpoint projector metadata remains part of checkpoint identity and should be checked, not deleted.
7. `stage1/cli.py:70-79,238-249` supports query inject and generate and uses None for format. Both query modes need explicit tests. Old vLLM injection depends on patched runtime TurnGenerationSpec; a plain vLLM install cannot be assumed to implement it.

### Minimal implementation allocation

- Evaluation config/CLI: explicit VAGEN/stage1/stage2 protocol selector; stage-specific checkpoint acceptance; immutable resume captures protocol, dynamics, prompt identity and token IDs, all generation options. Existing WM route remains unchanged.
- Agent-owned pure protocol helper: raw prompt adaptation, ordered query rendering, strict response parsing; no imports from training. Keep raw model text and diagnostic parse failure.
- Backbone-owned direct generator: plain raw generation for VAGEN and Stage1 and Stage2 generate; Stage2 inject follows actual query token policy without constraining the learned action response. Preserve image placement and actual history. Validate token IDs and trim only model EOS/padding, never skip all special tokens.
- Environment-owned direct runner: use real session/client, explicit legacy semantic service contract where applicable; preserve genuine invalid-action outcomes and success provenance. Stage1/2 should not use planner state capture or require latent-state rollout schema fields that do not exist.
- Persist atomic episode records with set/seed/checkpoint identity, raw response, executed response/action, finish reason, format result, env success; count successes from these records. Resume validates identities and prevents duplicate seed counting. Standard heldout is base60+common_sense60, seeds1..60; smoke results must be named as smoke.
- Tests: stage/config rejection before model load, semantic/action/query parsing including wrong order/duplicates/trailing garbage, no forced formatting, injection only at real boundary, raw-special-token preservation, service payload fidelity, explicit success metadata, resume mismatch and immutable checkpoint identity. CPU tests do not establish real rollout quality.

### Active launcher inventory / path hazards

`experiments/training/sft/evaluation/run_step79_stage1_stage2_eval.sh` calls formal evaluator at line363, but is a historic mixed pipeline. `environment_arm.sh:14-26` hardcodes AI2THOR and Vulkan under /project/peilab/atst/flower; lines40-49 override HOME and start `vagen.envs.navigation.serve`. Replace deployment assumptions with explicit validated runtime paths/options; do not copy these to a100.

Competing legacy launchers to remove or make thin delegates for the scoped early-stage eval:
- experiments/training/sft1/eval_greedy_valtest.slurm (vagen.trainer.main_ppo:226)
- experiments/training/sft1/run_parent_checkpoint_eval_arm.sh (vagen.trainer.main_ppo:352)
- experiments/training/sft2/eval_greedy_valtest.slurm (vagen.trainer.main_ppo:230; inspect semantics: historical sft2 often means current stage3)
- experiments/navigation_baseline/sft1_eval_one_model_valtest.slurm and sft1_eval_vagen79_greedy_valtest.slurm (vagen.main_ppo)
- experiments/navigation_baseline/submit_sft1_eval_{vagen48,epoch17_nimloth,epoch17_vs_step79,vagen79_after_train}.sh, resubmit_sft1_eval_vagen48.sh
- experiments/navigation_baseline/{run,submit}_sft1_lora_ckpt_eval_watcher.sh and sft1_lora_ckpt_eval_watcher.slurm
- experiments/navigation_baseline/summarize_sft1_eval_rollouts.py, compare_sft1_eval_summaries.py, backfill_wandb_eval_metrics.py need caller audit; retain offline historical record compatibility if needed, but no competing live evaluation driver.

Do not delete rollout collection scripts merely because they invoke main_ppo: rollouts_greedy_parallel.slurm and rollout_source_runtime_parity.slurm are collection/reproduction responsibilities. Do not change experiments/training/sft2/eval_mcts_rollout.py or RL eval launchers in this early-stage scope.

## Related specs and references

Read `.trellis/spec/domains/sft-stages.md`, `agent-rollout-and-state.md`, `reconstruction-and-evaluation.md`, `.trellis/spec/python/configuration-and-interfaces.md`, `agent/README.md`, `stage2/README.md`, workflow and trellis-start skill. They require typed validated configuration, no hidden fallback, agent/environment ownership of online evaluation, explicit raw response provenance, distinct heldout evidence and planner semantics.

No external web references needed: claims derive from current source. Memory registry MEMORY.md:340 was consulted only for historical heldout scope; verify actual assets when launching.

## Caveats / Not Found

Dedicated worktree external/VAGEN source files were absent during audit; server parser and exact dependency checkout API must be verified before integration/remote launch. No remote preflight or model generation performed. B agent is concurrently adding action_prompt; current references indicate inspected pre-edit state. This document is an implementation plan, not proof the new pipeline runs.
