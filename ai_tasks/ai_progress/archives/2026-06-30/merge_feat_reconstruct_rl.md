# merge feat/reconstruct + feat/rl into dev

## 目标

- 审查 `feat/reconstruct` 与 `feat/rl` 的变更。
- 将两条分支 squash 后合并到当前 dev 目标分支。
- 尽量把参数/config 变更与代码变更分离；参数变更尽量集中到同一个提交。
- 审查发现的问题先记录到 `ai_tasks/merge_dev.md`，不直接修复。

## 初始状态

- 当前工作区：`/workspace/remote2/nimloth-dev`
- 当前分支：`dev`
- 注意：本地不存在 `feat/dev` 分支，也未发现 `Jun21-SFT2` tag；当前可用目标分支为 `dev`。
- 开始前已有未提交/未跟踪文件：`AGENTS.md`、`.agents/skills/git-worktree`、`.agents/skills/slurm`、`.local`、`ai_rules/known_errors/`。这些看起来不是本任务产生，合并时不应覆盖。

## 计划

1. 查看两条 feature 分支相对 `dev` 的变更与提交。
2. 审查 diff，记录问题到 `ai_tasks/merge_dev.md`。
3. 在 `dev` 上按代码/参数分组 squash 提交两条分支变更。
4. 运行可行的静态验证/测试。
5. 更新进度并汇报。

## 已完成步骤

- 读取项目规则与当前进度。
- 检查 git 分支/worktree/初始 dirty 状态。

## 文件修改

- 本进度文件。

## 验证

- 待执行。

## 待确认/风险

- `feat/dev` 分支与 `Jun21-SFT2` tag 在本地均未找到；暂按当前 `dev` 分支作为目标处理。

## 完成状态

- 已在 `dev` 上完成 squash 合并，新增提交：
  - `240f6bc feat(rl): squash online RL training pipeline`
  - `e95757f config: squash RL and baseline launch parameters`
  - `b78b3ee feat(reconstruction): squash decoder training and diagnostics`
  - `c9291a4 config: add reconstruction training parameters`
  - `c617da8 docs: record merge review issues`
- 审查问题已记录到 `ai_tasks/merge_dev.md`。
- 冲突处理：保留当前 dev 的 `.memory/memories.jsonl` 与 `AI_branch_progress.md`；`external/VAGEN` 指向 feature 分支的 `93c1124...`；`qwen_tuning.py` 同时保留 torchao workaround 和 RL 的空 `modules_to_save`。

## 验证命令和结果

- `python -m py_compile ...` 覆盖新增 RL/environment/agent/reconstruction/wm/eval 源码与相关测试文件：通过。
- `bash -n ...` 覆盖新增/修改的 baseline、RL、reconstruction shell/slurm 脚本：通过。
- `PYTHONPATH=src python -m pytest ...`：失败，系统 Python 无 `pytest`。
- `PYTHONPATH=src .venv/bin/python -m pytest ...`：失败，`.venv` 中 torch import 缺 `libstdc++.so.6`。

## 未解决风险

- 本地没有 `feat/dev` 分支，也没有发现 `Jun21-SFT2` tag；本次按当前 `dev` 执行。
- 本地 submodule 未初始化，未验证 `external/VAGEN` 新 pointer 的内容。
- `AGENTS.md` 和 `ai_rules/known_errors/` 有本任务开始前已有的未提交改动，已尽量保留；不计入本次 merge 提交。
