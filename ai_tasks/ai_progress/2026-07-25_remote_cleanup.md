# 2026-07-25 远程代码与环境清理

## 目标

- 只在 superpod 登录节点审计并清理不再使用的 Nimloth worktree、分支和 Python 环境。
- 不使用 Slurm，不启动训练、测试、Ray 或 vLLM。
- 删除前逐项确认目标，保留所有未提交内容与仍被当前代码引用的环境。

## 已完成：只读审计

- `origin/dev` 与本地 `dev` 均为 `467df6e`；当前远程测试 worktree
  `.worktree/dev-rl-planner-distill` detached 在同一提交且完全干净。
- 远程存在大量历史 worktree。以下目录含真实未提交内容，禁止直接删除：主 worktree、
  `.worktrees/nimloth-exp-k8-preprojection-recon`、`nimloth.qwen-bug-repro`、
  `.worktree/dev`、`.worktree/review-55c0a0a`、`.worktree/rollout-resolution255-dev`、
  `.worktree/rollout-resolution255-rerun`。
- 多个历史 worktree 只含 `external/le-wm/__pycache__`，但部分嵌套 VAGEN/le-wm
  submodule 还有修改；删除这些 worktree 仍需明确授权，不能把 submodule 状态当成空目录。
- 当前仓库脚本仍显式引用 `.venv`、`.venv-vagen-main` 和 `.venv_vllm128_tmp`；三者
  不能按目录名直接删除。`.venv_vllm` 在当前仓库未找到引用，是环境清理候选。
- 登录节点未发现当前 Nimloth/Ray/vLLM Python 进程。按人类要求未查询 Slurm，因而本次
  审计不声称排除了计算节点上的任务引用。

## 待人类确认

1. 是否先执行保守清理：删除完全干净且提交仍被远端分支引用的历史 worktree，保留当前
   `.worktree/dev-rl-planner-distill` 与所有脏 worktree。
2. 是否删除唯一未被当前仓库引用的 `/project/peilab/atst/nimloth/.venv_vllm`；其余环境保留。
3. 是否删除已经合并到 `dev` 的远端分支引用；建议与 worktree 清理分开决定。

## 尚未执行

- 未删除任何 worktree、分支、环境、代码或实验产物。
