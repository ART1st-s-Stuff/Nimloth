# 2026-07-24 DINO SFT2 + online PPO merge review

## 目标

- 审查 `feat/sft2-dino-grid-ablation`。
- 审查 `exp/rl-dinogrid-ep1-online-ppo`（含本地提交 `bfa9c15`）。
- 将通过审查的内容按 semi-linear 策略合入分支 `dev`。

## 当前计划

1. 核对两个源分支的 ancestry、工作区和未推送提交。
2. 审查 DINO grid SFT2 与 online PPO 的正确性、checkpoint、split、分布式运行和测试。
3. 在隔离集成分支 `merge/dino-rl-online-ppo` 合并 DINO 分支的尾部提交并修复阻塞问题。
4. 运行静态、定向和可用的完整回归。
5. 先将集成分支以 `--no-ff` 合入审查分支 `fix/sft2-review-bugs`，再将该审查分支合入目标分支 `dev`，更新进度并整理工作区。

## 已完成

- 最终目标分支为 `dev`，DINO 分支为 `feat/sft2-dino-grid-ablation`。执行中曾错误地把工作区路径 `nimloth-dev` 解释成 `fix/sft2-review-bugs` 目标语义；人类纠正后已补做正式 `dev` 合并。
- 三个相关源工作区均 clean；`exp/rl-dinogrid-ep1-online-ppo` 比 origin 多本地提交 `bfa9c15`。
- RL 分支在 merge commit `f65ec2f` 已整合 DINO 分支到 `62b0120`；DINO 分支随后仅增加 `802c67e`、`309b4cc` 两个进度提交。
- 从 RL tip `bfa9c15` 创建隔离集成分支与 worktree。
- 以 merge commit `a7f973e` 合入 DINO 分支最后两个进度提交；冲突仅在 `AI_branch_progress.md`，已保留双方不重叠的 DINO resume 与 RL 记录。
- 审查确认 DINO cache lineage、terminal target、FP32 grid auxiliary、EMA target、checkpoint extras 和 RL fresh-policy manifest 均为 fail-closed；RL 每 rank 使用相同窗口以保持 FSDP forward 与未包装小模块梯度一致。
- 发现固定 `run_vllm_online_ppo_2node4.sh` 已与当前 TP4 config 和异构 per-node launcher 冲突，且会触发 diff-check；删除该废弃入口并把 README 统一到 config-driven launcher。
- 为本地未推送修复 `bfa9c15` 新增 checkpoint 恢复回归，证明 RL 直接从 SFT2 `state_proj.pt` 重建 slot projector，不依赖 HF 目录中的旧 SFT1 sidecar，并验证 grid auxiliaries 的冻结边界。
- 审查内容先在 `590d713` 合入 `fix/sft2-review-bugs`，随后以 merge commit `7cd290b` 合入并推送目标 `dev`。RL 源分支在审查期间新增的 `757bfcb` 也已审查并以 `8fdbe42` 纳入 `dev`；最终扩展回归仍为 `173 passed, 1 skipped, 1 expected warning`。

## 文件修改

- `AI_branch_progress.md`、`ai_tasks/ai_progress/2026-07-23_sft2_dino_grid_refactor.md`：合并双方最新实验进度。
- `experiments/training/rl/README.md`：移除废弃固定拓扑入口，说明当前 per-node 异构 rank 启动语义。
- `experiments/training/rl/run_vllm_online_ppo_2node4.sh`：删除已失效的固定 TP8 启动器。
- `tests/training/rl/test_grid_checkpoint.py`：新增 DINO-grid SFT2 → RL checkpoint 恢复契约。
- `ai_tasks/ai_progress/2026-07-24_dino_rl_merge_review.md`：本进度记录。

## 验证

- `bash -n`：当前 RL/SFT2 生产启动器通过。
- `python -m compileall`：相关源码和实验入口通过。
- 本地无 Torch/Pytest；已同步隔离服务器 worktree。首次 collection 因新 worktree 未初始化 `external/le-wm` 失败，初始化该 submodule 后正常收集。
- 服务器首轮定向回归为 `68 passed, 1 failed`；唯一失败来自新增测试用非生产 `ValueHead(hidden_dim=4)`，而 checkpoint 格式按生产默认 hidden 维恢复。测试 fixture 已改为生产构造；这不是源分支运行错误。
- 修正后同一组定向回归 `69 passed`。
- 扩展回归覆盖 `tests/wm`、SFT2、RL、rollout、Qwen backbone 与 SFT1 merge：`173 passed, 1 skipped, 1 expected warning`。skip 为既有环境门槛，warning 来自刻意验证单样本 unbiased std。
- 最终 `bash -n`、`compileall`、`git diff --check` 均通过。

## 待确认/风险

- `main` 工作区存在人类未提交改动，本任务未触碰 `main`；最终目标由人类明确为 Git 分支 `dev`。
- 三个只读子审查 agent 均异常退出，因此当前审查由主 agent 完成，不会把子 agent 失败当作审查结论。
