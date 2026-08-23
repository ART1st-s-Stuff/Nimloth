# ID59 deployed actor + SFT1 projector visual/goal audit

Status: **completed**. Job `528906` ran read-only on `normal/dgx-10` with one H800, `COMPLETED 0:0`, elapsed `00:13:23`. No training or checkpoint occurred.

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

## Result and decision

- Runtime commit: `12fa3df3f06156fe24f4e330a2643eeeadfb9861`.
- Fixed-SFT1-projector SFT1→ID176 state RMSE is `0.018796` on validation. ID176+SFT1 validation DINO RMSE/cosine is `0.846426/0.642177`, versus SFT1+SFT1 `0.846767/0.641844`; the visual non-inferiority gate passed.
- Replacing only the projector on ID176 hidden causes state RMSE `0.885311` and degrades DINO RMSE/cosine to `1.119524/0.402734`.
- Grounded ID176+SFT1 flattened-K16 retrieval top1/top5/MRR is `0.086455/0.288184/0.196890`, below frozen DINO `0.109510/0.371758/0.242340`; the goal-above-DINO gate failed. Majority top1 is `0.070423`.
- Retrieval evaluates 347 represented-label validation rows. Eight grounded `Footstool` rows are excluded and reported because no train-gallery row has that target. One exact-image candidate is excluded.
- Declared category differs from actual source config for 587/3211 train and 67/355 validation rows. The grounded labels remain exact instruction→actual-asset mappings, but task-disjoint split identity is not trustworthy enough for a final learned-probe claim.
- Natural train same-image pairs show larger state change under different goals than same goals (`0.020433` vs `0.003009` RMSE), but sample counts are only 21 vs 4 and actual CoT also changes; this cannot override the failed retrieval gate.

Decision: retain SFT1 projector as the visual anchor candidate, but do not start T1 residual-WM training. A stronger controlled goal probe/counterfactual gate is required first. Any learned diagnostic readout, projector calibration or WM training needs a separate authorization.

Integrity: `result.json` SHA256 `9b821d0b09581e407ce80ae4871663b7a37823911ed1b92a8dd904270293d25e`; float32 NPZ SHA256 `32077f60e749d9b10fd8ff76d0bd2cdd2a290dafbfeb2b04f8413733e20c5bb3`. All eight arrays were re-opened and validated for exact shape, dtype and finiteness. W&B: `https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/nimloth-recon/runs/nimloth-recon-id59-id176-sft1-goal-audit`.
