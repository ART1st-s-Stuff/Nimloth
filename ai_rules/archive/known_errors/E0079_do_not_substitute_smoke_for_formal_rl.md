# E0079：不得用 RL smoke 代替人类要求的正式 RL

错误：人类已要求开始正式 RL，但实际只提交了
`rl.iterations=1`、4条 episode、1次 optimizer step 的 mechanics smoke，并在 smoke
正常结束后停止。

原因：把“正式训练前必须通过真实多卡门禁”误当成了人类请求的终点，
没有在门禁通过后继续提交正式的多iteration、fresh-rollout 训练。

正确做法：smoke 只证明 rollout、真实分布式 backward、optimizer和checkpoint机制可运行。
人类要求正式 RL 时，必须使用明确的 formal config、独占输出/W&B identity与
resume-safe outer loop继续训练；若仓库只有 smoke config，必须先新增并验证正式配置，
不得把smoke结果表述为正式 RL 已完成。

证据：`configs/training/rl/planner_greedy_h1_smoke_1x4.yaml`的`iterations=1`；
`AI_branch_progress.md`中ID113终态为7分42秒、4条trajectory和`global_step=1`。
