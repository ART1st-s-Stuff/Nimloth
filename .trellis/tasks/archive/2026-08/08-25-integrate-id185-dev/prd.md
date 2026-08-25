# Integrate ID185 and submodules into dev

## Goal

将最新 Trellis dev 基线与完整 `feat/id185-rollout-visualization` 历史按 semi-linear 策略集成到本地 `dev`，并把根仓库、VAGEN 与嵌套 VERL 固定到该分支已经验证的精确提交；本任务不 push。

## Requirements

- 保留本地 `dev` 的未推送提交 `44af540e`。
- 先纳入 `chore/trellis-init` 的四个提交，最终 Trellis 基线为 `d92b76a4`。
- 完整纳入 `feat/id185-rollout-visualization` 在共同基线 `0111d0de` 之后的 358 个提交，不改写其功能语义。
- 使用 semi-linear merge：在临时源分支上把 ID185 提交线性重放到 Trellis dev 基线，再以显式 merge commit 集成。
- 冲突只按双方真实内容解决；已知重叠路径 `AI_branch_progress.md` 必须同时保留 `44af540e`/Trellis 迁移里程碑与 ID185 后续记录。
- 根仓库最终固定：
  - `external/RCDM = 71daaf10a73bb2012864f0827c68d209fc92b0a5`
  - `external/VAGEN = 9f1e89eb8c9839a406b6e62aa75703494a79e5b5`
  - `external/le-wm = 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`
- VAGEN 必须递归固定 `verl = 494f264494b2525f2c13595f63ac4912963e6d2f`，并能从配置的远端获取精确提交。
- 保留 `dev` worktree 中无关的 `.pi/task-tree/` 文件及 `external/le-wm/__pycache__/`，不得删除、覆盖或提交。
- 不修改实验产物、模型或checkpoint。人类已在集成中明确批准原样纳入ID185通过memory skill历史创建的M0015--M0017；它们继续保持`pending-human-verification`，本任务不手工编辑、不upvote且不把它们表述为已审批。
- 不启动实验、Slurm任务或远程GPU工作。
- 不 push 根仓库或任何submodule。

## Acceptance Criteria

- [x] Trellis task已通过规划审查并处于`in_progress`后才执行集成。
- [x] 临时重放分支包含与原ID185提交序列等价的358个提交；`git range-diff`只允许已审查的冲突解决差异。
- [x] 最终集成提交同时包含Trellis基线和重放后的ID185父线，符合semi-linear策略。
- [x] `AI_branch_progress.md`同时保留两侧不重叠记录，没有整侧覆盖。
- [x] 根仓库和递归submodule提交与Requirements列出的SHA完全一致，且工作树除已知无关文件外clean。
- [x] `.memory/memories.jsonl`相对dev只原样增加M0015--M0017，三条level仍为`pending-human-verification`。
- [x] `git diff --check`通过。
- [x] 受影响Python源码与测试文件语法编译通过。
- [x] 新增/修改shell与Slurm入口全部通过`bash -n`。
- [x] Trellis task/context校验通过，JSON/TOML可解析；PyYAML不可用已明确记录。
- [x] 明确报告本地缺少pytest/torch导致未运行的测试，不以静态检查替代测试结论。
- [x] 经完整diff与验证审查、取得最终commit批准后，完成本地merge commit；Trellis收尾等待最终接受。
- [x] 本地`dev`已fast-forward到集成结果；`origin/dev`保持不变。

## Exclusions

- 不挑选或重写ID185内部算法设计。
- 不修复集成过程中发现但不阻塞合并的既有代码问题。
- 不更新submodule到ID185未固定的新提交。
- 不把旧VAGEN legacy历史与新的upstream VAGEN历史强行合并；根仓库采用ID185已固定的upstream VAGEN提交。
- 不运行模型质量评估或GPU验证。

## Authorization

人类已明确选择完整ID185分支、授权创建并执行Trellis集成task，并要求先只完成本地合并、不push。
