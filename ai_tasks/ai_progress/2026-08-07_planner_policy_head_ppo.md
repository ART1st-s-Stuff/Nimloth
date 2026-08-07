# 2026-08-07: PlannerPolicyHead action-level PPO

## 任务目标

- 新增与 `ValueHead` 同构的 `PlannerPolicyHead`，输出离散动作 logits。
- 保留现有 `ValueHead` 作为 `Q(s_t,a_t)` critic。
- H=1 planner 训练 rollout 从 `PlannerPolicyHead` 的 categorical policy 采样动作，
  持久化真实 behavior log-prob，并对该 head 执行 clipped-ratio PPO。
- PPO 梯度继续通过冻结的 StateProjector 回传到 Qwen language body；不把 planner
  action token 误纳入 direct-Qwen token PPO。

## 当前计划

1. 核对 WorldModel、planner trace、fresh rollout、checkpoint 和 optimizer 契约。
2. 实现 PolicyHead、rollout sampling/provenance 和 H=1 fail-closed 约束。
3. 实现 action-level PPO actor loss与 state-only advantage baseline。
4. 补齐 checkpoint objective/version、配置和回归测试。
5. 运行 CPU 测试并更新分支进度。

## 已确认设计

- `PlannerPolicyHead` 结构与 `ValueHead` 一致，但输出语义为 action logits。
- `ValueHead` 不改为 logits；它继续承担 action-value critic，避免同一输出同时承担
  policy probability 与 return regression 两种冲突语义。
- 训练 behavior 必须从 PolicyHead categorical distribution 采样；evaluation 使用稳定
  argmax。沿用 deterministic `0/-inf` behavior distribution 不构成 on-policy PPO。
- advantage 使用完整 episode MC return 减去 frozen state-only baseline；具体计算在实现
  前继续核对现有 batch/DDP 契约。

## 文件修改

- 新增 `PlannerPolicyHead`，并接入 standard/grid `WorldModel` 的 logits forward、
  trainable/synchronized module 边界、optimizer 参数组和 checkpoint save/resume。
- 新增 `agent.planning.search_mode=policy`：H=1 训练 categorical sampling、evaluation
  argmax，trace 持久化完整 behavior action log-prob。
- 新增 action-level PPO clipped surrogate。old statistics 在更新前从 rollout 保存的
  decision state 计算并校验；advantage 为
  `G_t - sum_a pi_old(a|s_t) Q_old(s_t,a)`。
- PolicyHead PPO 的当前 state 由完整持久化 prefix 重算，因此梯度路径为
  `PlannerPolicyHead -> StateProjector -> Qwen language body`。ValueHead 同时使用普通
  executed-action MC regression，不再在新模式中承担 policy logits。
- 新 objective 为 `receding_horizon_planner_policy_ppo_v1`；checkpoint 保存完整
  `planner_policy_config` 和 PolicyHead。旧 objective/checkpoint 不能静默 resume。
- 新增显式、固定 seed 的 PolicyHead 初始化脚本；它只读取 ValueHead 结构，不复制
  critic 参数。formal runner 首轮要求单独 head artifact，后续轮次使用上轮 snapshot。
- 更新 fresh-manifest 指纹和单/多节点 rollout/train/eval runner 参数传递。
- 扩展现有 real-GPU mechanics gate，使其可校验 PolicyHead 梯度、参数更新和
  多 rank replica 同步；legacy ValueHead critic gate 仍可使用原配置运行。
- 新增 smoke config 和 head/config/trace/PPO/full-prefix gradient 回归测试。

## 验证

- 人类按授权在 `nimloth-dev` 中保全本地进度文件并将 `dev` 从 `2007c661`
  线性快进到 `969fc557`（相对 `origin/dev` ahead 164）。stash 恢复只在
  `AI_branch_progress.md` 产生两处内容冲突；解决结果完整保留快进后的 ID143
  历史和原 dev 独有的三个新增区域，且不含冲突标记。人类已在共享 Git index
  中标记解决并恢复为未暂存状态；该进度改动不会进入 PlannerPolicyHead 提交。
- PlannerPolicyHead 工作区差异已直接迁入 `nimloth-dev`：32 个修改文件、7 个新增文件
  与原 feature worktree 逐字节一致。dev 上 `git diff --check`、使用 `/tmp` pycache 的
  `compileall`、六个受影响 shell/Slurm 入口的 `bash -n` 全部通过。
- `compileall`（使用 `/tmp` pycache）通过。
- 四个受影响 shell runner 的 `bash -n` 通过。
- `git diff --check` 通过。
- 本机默认 Python 没有可用 torch/pytest；发现的本地 `.venv` 解释器链接已失效，
  直接复用 site-packages 也不能加载 PyTorch C extension，因此尚未运行 pytest。
- git 暂存的权限提升请求被环境策略拒绝；未使用 scp/patch 等方式绕过。需得到新的
  git 写权限后提交并同步到服务器，运行 CPU focused suite 和之后单独授权的 GPU gate。
- 人类随后明确授权 `git add/commit/push`，但执行层再次拒绝，说明当前审批策略禁止
  所有 `require_escalated` 请求，即使已有用户授权也不放行。需要人类直接提交，或由
  环境管理员开放该 worktree 的共享 Git index 写权限。
- 服务器 focused pytest、Git commit/push 和远端 dev worktree 更新仍未完成；当前没有
  运行 GPU gate、训练或评估。

## 待确认问题

- 无；人类已批准新增 `PlannerPolicyHead` 并要求其结构与 `ValueHead` 一致。
