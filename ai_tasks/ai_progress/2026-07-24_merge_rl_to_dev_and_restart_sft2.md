# 2026-07-24 合并 online RL 到 dev 并重启 SFT2

## 目标

1. 将 `exp/rl-dinogrid-ep1-online-ppo` 合并到 `dev`。
2. P0 无条件采用最新规则：禁止 fixed CoT；普通 state 使用真实 assistant response；
   terminal observation 使用已确认生成协议额外生成并持久化真实 CoT。
3. 手动验证 P1 的 DINO/SFT2 模块化改动与 RL/vLLM/分布式改动没有逻辑冲突。
4. 合并与回归通过后，以前一次因分支冲突暂停的相同 SFT2 参数重新启动，并监控到
   数据/cache、分布式训练和首批指标均健康。

## 当前计划

1. 核对两个分支、共同祖先、P0/P1 代码职责和实验历史。
2. 在 `dev` 手动合并 RL 分支并解决冲突。
3. 执行静态检查、定向测试和必要的远端回归。
4. 提交并推送合并后的 `dev`，同步服务器独立 worktree。
5. 按已确认的 terminal CoT 协议生成正式 train/val 数据、重建 cache，并启动新 ID
   SFT2；监控到健康启动。

## 已完成

- 已刷新远端引用并确认：`dev=ee0a636`，RL 分支当前为 `65f97d6`，两个 worktree
  均干净且与远端同步。
- 已确认上次 terminal CoT smoke 在 `ebc4d3b` 通过 8 项定向回归和一条真实 GPU
  样本；随后因分支 lineage 冲突取消 hold，未生成正式数据/cache，也未启动训练。
- 已核验本地 memory `M0001` 的服务器 Python 证据；当前 SFT2 应显式使用
  `/project/peilab/atst/nimloth/.venv-vagen-main/bin/python3`。该 memory 尚待人类审批，
  CLI 因此拒绝 upvote。
- 已人工核对 P1：merged tree 保留唯一 `SFT2Algorithm` 与可配置 `DINOGridLoss`；
  terminal-CoT 只扩展 transition/prompt/cache，不复制训练核心。
- 已执行 merge 并解决 4 个文本冲突；最新 P0 进一步删除 fixed thought active path。
  RL 的 current/terminal CoT state replay 与 PlanningPolicy 因未完成而 fail-fast，明确
  留作 TODO。
- 本地 `compileall`、`git diff --check` 通过；本机没有 `pytest` 命令，完整测试需在
  superpod 明确 Python 环境执行。

## 文件修改

- 本进度文件。

## 验证命令与结果

- `git fetch origin --prune`：成功。
- `git status --short --branch`：两个目标 worktree 均干净并跟踪同名远端分支。

## 待确认问题

- 无。实验使用上次暂停前已经由人类确认的 terminal CoT 与 SFT2 参数；若远端记录
  与仓库记录不一致，将停止并请人类确认，不会猜测。
