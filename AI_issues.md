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
