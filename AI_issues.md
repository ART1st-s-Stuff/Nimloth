# AI_issues.md — 需要人类确认的问题

## 2026-06-10 memory skill/CLI 待确认

1. **审批 pending memory**：当前 `.memory/memories.jsonl` 中有 `M0001`，记录 memory skill/CLI 的存在。请在需要时运行 `./skill human memory-approve` 审批或驳回。
2. **Task skill/CLI**：是否继续按类似方式实现 `task` skill/CLI？
3. **旧记忆系统迁移**：是否逐步废弃 `AI_branch_progress.md`、`AI_issues.md`、`ai_tasks/`，转为由 skill/CLI 生成或替代？

## 人类回答
1. 不需要进入记忆，已经在skill里了。
2. 是，也需要针对issue写类似的CLI
3. 是。

## 2026-07-24：`planning.horizon=2` 长时实验模式待确认

已确认：人类所说的“预测2轮”指 `agent.planning.horizon=2`，不是
`predictor.history_size=2`。尚未据此修改配置或启动作业。

当前实现会在加载模型前拒绝同时启用 `agent.planning.enabled=true` 和
`actor.enabled=true`，因为 planner behavior 尚不能执行精确 PPO replay。继续实验前需要
人类明确选择：

1. 关闭 PPO actor，直接用 `planning.horizon=2` 做较长的 planning rollout/evaluation；
2. 保持训练时 direct-policy online PPO，训练后再用 `planning.horizon=2` 做固定评估；
3. 先实现 planner-compatible PPO，再做 planning-enabled online PPO。

## 2026-07-24：planning rollout后的policy update定义待确认

人类提出的方向已确认包含三部分：真实rollout直接监督ValueHead；在同一次Qwen
state/action forward保留被planner选中动作的Qwen概率，并用真实return构造policy
advantage；WM是否训练由配置决定。

审查后仍需人类决定policy objective名称和严格定义：

1. planner作为teacher，Qwen对planner root distribution做policy distillation或
   advantage-weighted regression；这允许planner采集的数据，但不称on-policy PPO；
2. 定义包含Qwen logits和planner score的统一随机behavior policy，rollout和replay都
   使用这个完整分布，再计算严格匹配的old/new ratio。

两条路线都应先补逐步rewards、terminal/truncation和完整八动作Qwen/planner分布；只保存
selected action的Qwen概率不足以审计entropy、KL、support和behavior mismatch。

## 2026-07-25：远程代码与环境清理范围待确认

登录节点只读审计已完成，尚未删除任何内容。精确证据和目录分组见
`ai_tasks/ai_progress/2026-07-25_remote_cleanup.md`。

需要人类确认：

1. 是否删除除当前 `.worktree/dev-rl-planner-distill` 外所有**完全干净**、且 HEAD 仍被
   远端分支引用的历史 worktree；所有含未提交内容或 submodule 修改的目录保留。
2. 是否删除当前仓库无引用的 `.venv_vllm`；`.venv`、`.venv-vagen-main`、
   `.venv_vllm128_tmp` 仍有代码引用，默认保留。
3. 是否另行删除已合并到 `dev` 的远端分支引用，还是本轮只清理服务器 worktree 目录。

### 部分执行状态

人类已确认上述保守范围。7 个已合并远端分支已删除；普通 `git worktree remove` 因目标
含已初始化 submodule 被 Git 拒绝，后续步骤按 `set -e` 停止。7 个目标 worktree 复核仍
完全干净，`.venv_vllm` 和失效 worktree 注册也仍在。继续需要人类额外确认：是否仅对
这 7 个已列明路径使用 `git worktree remove --force`，随后按原确认范围清除失效注册和
`.venv_vllm`。

## 2026-07-25：planner-distillation RL 启动参数待确认

人类已授权开始 RL，但新路径没有可合法继承的完整旧配置。提交 GPU 任务前仍需明确：

1. 实验范围：先做一轮 correctness smoke，还是直接给出正式的 episodes、每条最大 steps、
   iterations 和停止条件。
2. 算法参数：`agent.planning.teacher_temperature`、
   `actor.planner_distillation_weight`、`agent.planning.device`、
   `predictor.train_wm`、`token_credit.gamma`、`token_credit.gae_lambda`、
   `token_credit.value_lr`、`token_credit.value_loss_weight`、
   `token_credit.hidden_dim`；当前 truncation 只能显式选择
   `rl.truncated_bootstrap=zero`。
3. rollout 参数：Qwen `temperature`、`top_p`、`max_reasoning_tokens`；terminal CoT 使用
   同一组数值的协议已经确认，但数值本身仍需固定。
4. 资源：partition、物理 GPU 总数、`rollout_tensor_parallel_size`、训练 `world_size` 和
   `gpus_per_rank`。总 GPU 必须由 config 推导，不能根据旧实验猜测为 8。
5. 模块边界：是否沿用 full Qwen language + ValueHead + TokenValueHead 训练、vision 和
   StateProjector 冻结；WM 由上面的 `predictor.train_wm` 决定，DINO 已确认关闭。
