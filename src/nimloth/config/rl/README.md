# RL 配置

本包负责 RL YAML 加载、schema 校验和命令行覆盖。RL 运行时代码不得自行解析 YAML。

planner训练使用ValueHead的PPO clipped critic，必须显式配置
`value_head.ppo_clip_range>0`和`value_head.ppo_epochs>=2`。这两个字段只适用于planner；
未启用planner的sequence/direct-Qwen路线继续使用各自原有的value/actor配置。
