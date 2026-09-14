# Research: epoch7 objective-change continuation

- Query: smallest faithful model continuation with action16 and action-boundary/EOS16, fresh output and convergence.
- Scope: internal source audit; no remote access or source mutation.
- Date: 2026-09-12

## Findings

All source paths below are relative to `.worktree/fix-sft1-termination-retrain`, inspected at supplied 913ea614 source. Latest PRD/design/implement continuation paragraphs supersede old evaluation-only scope.

### Existing path is insufficient

- `src/nimloth/training/sft/stage1/workflow.py:116,157-185`: only fresh initialization or exact-manifest `--resume`. Fresh initializes semantic rows and owns data/base/cache; resume compares entire override list and input identities. No objective-change continuation flag.
- `stage1/initialization.py:193-196`: rejects sources already registering action tokens. Thus passing merged epoch7 as source model cannot be the standard fresh workflow workaround.
- `stage1/trainer.py:955-1019`: only same-output resume loads saved base plus adapter. Ordinary `--model` with `--lora` creates a new adapter, losing prior parameterization if merged model used.
- `stage1/trainer.py:1123-1168`: weighted identity mismatch rejected, optimizer/scheduler/convergence/RNG restored together. `checkpoint.py:79-87` identity comparison normalizes old absent action weight to1, otherwise exact. Keep this strict.

### Minimal explicit extension

Add a workflow/trainer **separate** epoch-boundary continuation input (suggested spelling `--continue-from-checkpoint PATH`, not currently implemented). Source must be committed complete epoch checkpoint, Stage1 current recognized schema/scope, same base/data/tokenizer/LoRA/precision/world/hyperparameters except explicitly approved loss weights and new output location. Reject optimizer-step snapshots, partial checkpoints, source/destination overlap, resume+continuation, unsupported stage, missing saved rank RNG or lineage. Bind original source checkpoint fingerprint and source identity in new workflow/checkpoint metadata.

Reuse original workflow base/data/cache after fingerprint verification, without semantic reinitialization or running successful records through preparation again. Output training state and new manifest live in a fresh unique run. Loading uses original base + same LoRA config + `checkpoint.load_lora_adapter_state` (line403), which verifies all saved tensor values including full saved embeddings/head. Do not merge LoRA then apply fresh LoRA.

Retain source completed global step126 and next global epoch8 (first update127), sampler `set_epoch(8)` (`trainer.py:1229`); no mid-epoch cursor. Separate global numbering from objective-phase counters. New optimizer and scheduler are the clean objective-change default; state explicitly that Adam moments are reset and constant scheduler warmup restarts in local update time. Alternatively preserving moments is technically possible but is a different declared continuation policy, not mathematically required. Do not call fresh optimizer a faithful optimizer resume.

Restore saved per-rank epoch RNG at the same existing epoch-boundary point (`trainer.py:1313`) to avoid accidentally changing dropout/random sequence due to model construction. This does not import old loss history. Fresh seeded RNG is possible but unnecessarily changes one more experimental factor. Require same world size.

Critical convergence trap: `convergence.py:49` requires `epoch == last_epoch+1`, state loading rejects nonzero last_epoch with absent losses. `trainer.py:1174,1227,1421` assumes convergence epoch equals global epoch. Use explicit saved `convergence_epoch_offset=7`: observe local epoch1 at global8, compare cursor using offset, and map terminal epoch back to global. Do not fabricate last_epoch7 with empty state or import old best/previous loss. New epoch8 establishes baseline; two later insufficient improvements can stop at epoch10. Persist offset so resumed new-objective run works. Reset best_val and old W&B/run identity as appropriate. Existing validation remains unweighted LM loss.

Suggested future official command shape (requires implementation): `experiments/training/sft1/train.py <existing required source/data/config flags> --output-dir <new-root> --continue-from-checkpoint <old-root>/train/epoch_007 --action-token-loss-weight 16 --action-boundary-loss-weight 16 --until-converged`. Flag names must follow final implementation. Do not execute now as an existing command.

### Weight identity consumers

- `stage1/loss.py:14`: action scope and exact selected weight mask; preserve existing action-number scope and add distinct boundary/EOS scope to avoid silently changing old meaning.
- `stage1/config.py:48`, `cli.py:218-219`: YAML mapping, finite weight validation, reject nondefault boundary weight for Stage2 as currently action weight.
- `trainer.py:517,620-679,716,1330`: W&B config, resume identity, resolved IDs diagnostics, actual weighted forward. Preserve unweighted validation.
- `checkpoint.py:79-87,164-202,254-281,305-334`: missing boundary field should normalize only to documented legacy1; both resume-step and epoch saves write identity plus HF config metadata; stage validation accepts known old/new scope explicitly and rejects unknown legacy schemas.
- `checkpoint_export.py:185-208`: propagate boundary weight/scope into merged HF config, do not relabel old adapter as new objective. Export's verifier checks saved adapters (`:127`) and restores untied embeddings (`:54`).
- `evaluation/early_checkpoint.py:35-51`: currently **hardcodes action weight8**; must validate supported scope and declared finite weights rather than one experiment setting, and understand explicit legacy boundary1/new boundary metadata. Stage1 and Stage2 eval share this entry; Stage2 branch checks query/projector contract, not action training weight.
- `stage2/__main__.py:1-3` calls Stage1 shared trainer with stage=query. Ensure boundary CLI/default and checkpoint writer do not leak Stage1 weighted protocol into Stage2. No separate Stage2 trainer.
- Source-wide search found no other Python consumer of action loss weight/scope. `backbone/qwen25vl/checkpoint.py` and `eval/sft_checkpoint_state_matrix.py` contain no such fields; no reason to broaden into WM/RL.
- Workflow manifest raw CLI overrides already bind the new weight. Unified eval contract fingerprints HF artifact, so exported config field changes are captured; review tests rather than duplicating loss metadata into every record.

## Files found

- `stage1/workflow.py`: official preparation/cache/launch contract.
- `stage1/trainer.py`: load, optimizer, scheduler, data cursor and convergence lifecycle.
- `stage1/checkpoint.py`: atomic checkpoints, identity matching, verified adapter loading.
- `stage1/convergence.py`: sequential local epoch plateau state.
- `stage1/checkpoint_export.py`: verified adapter merge and full HF metadata.
- `evaluation/early_checkpoint.py`: full HF stage and objective acceptance.
- `stage2/__main__.py`: shared trainer query dispatch.

## Related specs

Read `.trellis/workflow.md`, `.trellis/spec/domains/sft-stages.md`, `.trellis/spec/experiments/outputs-checkpoints-and-evidence.md`. Preserve recognized stage identity, unique output ownership, faithful versus objective-change continuation distinction, and recorded optimizer/scheduler/RNG semantics.

## External references

None needed; current source establishes supported interface. No memory-derived claim used.

## Caveats / Not Found

Remote epoch7 state contents, exact original command and base/data fingerprints are parent-owned verification. No remote access, source edits, tests or GPU execution performed. Research role intentionally did not read implement/check JSONL manifests. Fresh optimizer is a recommendation, not a proof that moment reuse is invalid. Continuation needs focused tests for new-run save/resume and convergence offsets, not merely initial loading.
