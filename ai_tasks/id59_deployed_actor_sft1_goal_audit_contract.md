# ID59 deployed actor + SFT1 projector visual/goal audit

Status: human authorized a read-only `normal 1×H800` check, expected within 20 minutes. No training or checkpoint is authorized.

## Question

Can the SFT1 `SharedSlotProjector` remain the canonical state anchor when fed hidden states from the deployed pre-RL ID176 actor?

## Inputs

- SFT1 merged backbone/projector checkpoint.
- ID176 deployed pre-RL actor checkpoint.
- ID176/ID74 `state_proj.pt` as the drifted-projector comparison.
- All 3,211 archived pre-RL train trajectories and 355 archived validation trajectories, first decision state only.
- Frozen original-observation DINO grid cache.

Every state uses the archived observation and its actual recorded assistant response/CoT. This is a controlled checkpoint forward, not newly generated same-generation ID176 behavior-time CoT; the result must not be presented as a replacement for ID188 rollout evaluation.

## Grounded goal labels

The migrated archive's declared `data_source/eval_set` does not always match the actual source environment config. ID59 therefore:

1. reads the referenced source JSONL row and parses its `NavigationEnvConfig(eval_set=...)`;
2. extracts the exact archived `Human Instruction`;
3. matches it against that actual asset;
4. requires the instruction to map to exactly one `targetObjectType`.

No heuristic or generated label is permitted. Declared-versus-actual mismatch counts are reported.

## Metrics

For three states—SFT1+SFT1 projector, ID176+SFT1 projector, and ID176+ID74 projector:

- validation state/DINO RMSE, cosine, token-centered cosine and scale;
- target-object retrieval from the 3,211-row train gallery to validation queries using slot-mean and flattened-K16 cosine; queries whose grounded target type is absent from the gallery are excluded and reported explicitly;
- top1, top5, MRR and macro-top1;
- exact-image train candidates are excluded;
- a visual-controlled retrieval metric first restricts candidates to 64 DINO-nearest train images;
- natural exact-image pairs with different real goals are audited separately;
- SFT1→ID176 backbone drift under a fixed SFT1 projector and projector drift under fixed ID176 hidden.

The goal audit is diagnostic because the source archive's actual task-disjoint split identity is uncertain after the metadata mismatch. It proves retained target information only if state retrieval exceeds DINO and majority baselines without exact-image leakage.

## Freeze and output boundaries

- All model modules are frozen under inference mode.
- No optimizer, backward, generation, action execution, parameter update, resume, or model checkpoint.
- Fresh output:
  `/project/peilab/atst/nimloth/outputs/experiments/evaluation/state_alignment/2026-08-23/59_id176_sft1_goal_audit_train3211_val355`.
- W&B project/name/ID: `nimloth-recon` / `59_id176_sft1_goal_audit_train3211_val355_k16` / `nimloth-recon-id59-id176-sft1-goal-audit`.
- Slurm: normal 1×H800, 16 CPUs, 128 GiB, hard walltime 30 minutes; excludes `dgx-09,dgx-13,dgx-32,dgx-51`.
- Retry after failure requires fresh output, W&B identity and production worktree.
