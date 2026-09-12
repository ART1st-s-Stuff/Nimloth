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


## Session 5: SFT1 FP32 token训练与epoch2/3评估收尾
<!-- trellis-session: v=2 fp=8d6ab799b8f601a6 -->

**Date**: 2026-09-12
**Task**: SFT1 FP32 token训练与epoch2/3评估收尾
**Branch**: `codex/fix-sft1-termination-retrain`

### Summary

完成FP32 embedding/head主参数与BF16前向支持及PEFT标记识别修复。LoRA/token LR均5e-5，动作权重2、边界EOS权重8，从原始语义初始化训练5epoch。五轮格式32/32。epoch2标准120测试成功38/120，epoch3为31/120。训练和评估进程结束，GPU释放，监控暂停。固定5轮不代表证明收敛。

### Main Changes

- 产物根：/mnt/nimloth/outputs/experiments/sft1-rollout2000/20260912T130307Z_token_fp32_lr5e5_action2_boundary8_epoch5。a100-1保存训练及epoch3_success_test120；a100-2保存epoch2_success_test120；各控制目录保留launch/end日志。
- 验证LM loss五轮：0.381900,0.371837,0.371460,0.370622,0.370495；加权：0.323109,0.299800,0.299270,0.298537,0.298436。

### Git Commits

| Hash | Message |
|------|---------|
| `1627d618` | feat(sft1): retain FP32 embedding masters with BF16 forward |
| `96f5972e` | fix(sft1): identify FP32 master leaves without PEFT forwarding |

### Testing

- [OK] 16项独立CPU回归、Ruff及diff检查通过；真实8GPU验证首轮格式32/32、epoch1 COMMITTED并进入epoch2。
- [OK] FP32 master及Adam状态和实际更新已核验；导出698张量核验、真实AI2THOR渲染通过；两次完整120episode评估。

### Status

[OK] **Completed**

### Next Steps

- 未授权新增训练或评估；代码尚未在本轮合并或push。保留epoch2为本次已评估checkpoint中成功率较高者，单次采样比较不证明统计显著性。
