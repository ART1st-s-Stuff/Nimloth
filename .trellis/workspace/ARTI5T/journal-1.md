# Journal - ARTI5T (Part 1)

> AI development session journal
> Started: 2026-08-25

---



## Session 1: Migrate Nimloth AI workflow into Trellis

**Date**: 2026-08-25
**Task**: Migrate Nimloth AI workflow into Trellis
**Branch**: `chore/trellis-init`

### Summary

Initialized Trellis 0.6.15 for Pi, Claude Code, and Codex; migrated Nimloth safety, workflow, experiment, progress, memory, worktree, and known-error contracts into source-backed Trellis specs and repository-owned skills; archived legacy rules losslessly; fixed Pi Desktop project-root resolution; completed cross-platform validation without product, experiment, data, checkpoint, TaskTree, or remote changes.

### Git Commits

| Hash | Message |
|------|---------|
| `aa258e44` | (see git log) |
| `80301f46` | (see git log) |

### Status

[OK] **Completed**


## Session 2: Integrate ID185 and submodules into dev

**Date**: 2026-08-25
**Task**: Integrate ID185 and submodules into dev
**Branch**: `merge/id185-trellis-dev`

### Summary

Semi-linearly integrated the complete ID185 history onto the Trellis dev baseline, pinned VAGEN/VERL recursively, validated the candidate tree, and kept the result local without pushing.

### Main Changes

- Rebased 358 ID185 commits onto d92b76a4 and merged them with explicit two-parent commit 33c37ca3.
- Updated VAGEN to 9f1e89e and nested VERL to 494f264 while preserving RCDM and le-wm pins.

### Git Commits

| Hash | Message |
|------|---------|
| `33c37ca372eefa96b5f24fb6295f01701dd3add4` | (see git log) |

### Testing

- [OK] Range-diff: 357 equal commits and one Trellis progress-context-only difference.
- [OK] Static checks: 1295 Python syntax files, 570 shell/Slurm files, structured config parsing, submodule and diff gates passed.

### Status

[OK] **Completed**

### Next Steps

- Human may review pending memories M0015-M0017 separately; no push was performed.


## Session 3: Complete SFT1 state interface v2 code canary

**Date**: 2026-08-26
**Task**: Complete SFT1 state interface v2 code canary
**Branch**: `feat/state-interface-v2-sft`

### Summary

Implemented and locally validated the strict DeepSight-style K16 SFT1-v2 code canary; archived the completed task without launching training.

### Main Changes

- Added same-forward Qwen K16/action output, unified seven-term state objective, strict data/manifest/DataProto contracts, complete-root FSDP worker, checkpoint/export, and non-launching canary config.
- Committed feature and test changes in the feature worktree after human approval.

### Git Commits

| Hash | Message |
|------|---------|
| `c4b2a357` | (see git log) |
| `8df9b853` | (see git log) |

### Testing

- [OK] Focused plus adjacent CPU structural gate: 48 passed in 3.71s; AST/config/task/diff and submodule cleanliness checks passed.

### Status

[OK] **Completed**

### Next Steps

- Create a dedicated experiment task and source-verify the real-data teacher/cache, checkpoint, metrics, resources, outputs, and exact launch command before requesting launch approval.


## Session 4: 完成项目 Trellis prompts 中文重写

**Date**: 2026-08-28
**Task**: 完成项目 Trellis prompts 中文重写
**Branch**: `dev`

### Summary

将9个项目维护的Trellis workflow/operational skill prompts重写为中文，保持machine contracts与审批门禁；focused validator、独立trellis-check和主会话全范围复核全部通过。

### Git Commits

| Hash | Message |
|------|---------|
| `7989667e` | (see git log) |
| `da51915d` | (see git log) |

### Status

[OK] **Completed**


## Session 6: 收尾 a100-1/a100-2 SFT 实验系列
<!-- trellis-session: v=2 fp=402d9c2248970000 -->

**Date**: 2026-09-14
**Task**: 收尾 a100-1/a100-2 SFT 实验系列
**Branch**: `dev`

### Summary

归档8个近期a100实验task，保留部分/失败边界、checkpoint和分支；Projector对照完成，暂停监控，保留a100-2共享服务。

### Main Changes

Closed eight recent a100 experiment tasks at the user's request. Main-checkout archives supersede stale historical task snapshots retained in experiment worktrees. No source branches were merged, no worktrees removed, and no additional weights/data were deleted by finish-work.

Archived scope: rollout-vagen-step60-sft1-sft2; sft1-rollout2000; fix-sft1-termination-retrain; batch-sft1-format-generation; eval-sft1-epoch5-6-test120; sft2-dino-information-test; eval-stage2-epoch3-deepsight-rollout10; sft2-deepsight-full. Each contains closeout-20260914.md and explicit meta.closeout.outcome. Archive completed status means the work item is closed, not that failed/partial original experiments passed acceptance. Original-test128 evaluation remains incomplete (16/128 epoch18, baseline not run). Early K16 Stage1 and other superseded failures remain failures.

Latest a100-1 full-language epoch12/324: validation total1.8815873563, LM.4116116464, DINO.7349878550. Query embedding epoch7 relativeL2 drift14.25%, cosine.989899. Latest a100-2 LoRA finished requested epoch5/135: total3.9405367076 (DINO4), LM.3679549396, DINO.8931454420. Different objective weights and other row-policy differences preclude simple causal comparison of raw totals.

Frozen-Qwen/Query Projector-only diagnostic finished at epoch12 by the agreed patience rule; best epoch12. Original1709train/193validation trajectories, 2152 validation answer states. FP32 original/final MSE.7361404896->.5597344041, cosine.7020128965->.7837318182, observation gain.1250385642->.2901100516, wrong-pair advantage normalized by training-mean MSE.2805801503->.6878266574. Supports previously unused readout capacity on the original split. Seed disjointness does not imply novel-scene validation; current local asset mapping shows scene/task overlap and original collection asset version was not independently established. No new rollout success evidence from projector-only fitting. Checkpoint base unchanged; best projector remains a separate artifact, not deployed into policy.

Projector result: a100-1 /mnt/nimloth/outputs/experiments/sft2-projector-probe/20260914_epoch12_original_split/fit/{finished.json,best_projector.pt,metrics.jsonl}. Final JSON copied into archived09-13-sft2-deepsight-full/projector-final-20260914.json. Baseline production-padded BF16 reproduction .7349898815 vs expected.7349878550 passed1% gate. Seven focused CPU tests and real first-record collator preflight passed before GPU execution.

Both hosts currently have no series training/evaluation GPU processes. a100-2 SAM3 PID631008 and ContactGraspNet631009 services remain. Projector automation paused on completion; no remaining ACTIVE a100 automations found. Checkpoint retention manifests and exact run lineage are referenced by .local/handoff-fresh-dino2-lr2e6.md and .local/sft2-checkpoint-retention.md.

Archive helper emitted pathspec warnings for formerly untracked task source paths; final auto-commit included all eight archive destinations. Verified final archive content rather than treating warnings as loss of records. Unrelated config, memory, submodule, RL spec/report and other Trellis changes remain untouched. Retained experiment branches have unmerged implementation commits; future integration is a separate task and was not silently done here.


### Git Commits

| Hash | Message |
|------|---------|
| `12620453` | Add frozen-state projector readout diagnostic |
| `cdac7ef6` | Import JSON for projector continuation regression |
| `ab8e2163` | Scope rollout readiness to assigned physical GPUs |
| `9f8bca39` | chore: record journal |

### Status

[OK] **Completed**
