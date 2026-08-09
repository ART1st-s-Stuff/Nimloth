# RL 配置

本包负责 RL YAML 加载、schema 校验和命令行覆盖。RL 运行时代码不得自行解析 YAML。

planner训练每个fresh rollout global step只执行一个完整objective optimizer epoch。
legacy planner必须显式配置`value_head.ppo_clip_range>0`和
`value_head.ppo_epochs=1`；PlannerPolicyHead必须显式配置
`planner_policy.ppo_epochs=1`。schema拒绝其他epoch数。未启用planner的
sequence/direct-Qwen路线继续使用各自原有的value/actor配置。
